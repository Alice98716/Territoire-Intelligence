import os
import sys

# Windows' console defaults to cp1252, which can't encode characters this
# file's logging uses throughout (→, —, etc.) - without this, a print() deep
# inside a try/except (e.g. right after a successful Claude call) raises
# UnicodeEncodeError, gets swallowed by the surrounding except, and silently
# discards a good result in favor of a hardcoded fallback. Must happen before
# any print() below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import re
import time
import math
import shutil
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import anthropic
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

print("[1/4] Loading environment variables...", flush=True)
load_dotenv()

#Anthropic AI 
claude = anthropic.Anthropic()

app = FastAPI(title="Quebec Territorial Intelligence RAG Engine")

# ── Rate limiting ────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# TODO before deploying off localhost: lock allow_origins down to the real
# frontend domain(s) (e.g. ["https://your-frontend-domain.com"]) instead of
# "*" - wide open is fine for local dev, not once this is reachable publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (perfect for local testing)
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods like POST, GET, OPTIONS
    allow_headers=["*"], # Allows all headers
)

# ── Security headers ─────────────────────────────────────────────────────
# Applied to every response regardless of scheme. Strict-Transport-Security
# is inert until this is actually served over HTTPS (browsers ignore it over
# plain HTTP), so it's harmless to add now rather than waiting for the HTTPS
# rollout - the rest (nosniff/frame-options/referrer-policy) are useful
# immediately. No HTTP->HTTPS redirect middleware yet: this only runs on
# localhost today, and a scheme-check redirect would either break local
# testing (no TLS listener to redirect to) or, once behind a reverse proxy,
# loop forever unless it trusts X-Forwarded-Proto instead of request.url.scheme
# - add that once the deployment target (and its proxy setup) is decided.
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

cloud_uri = os.getenv("MONGO_URI") #connecting to MongoDB
if not cloud_uri:
    raise RuntimeError("MONGO_URI missing from environment variables.")

from spatial_rag_v1 import SpatialHybridRAG, NAICSClassifier, DB_NAME
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

#Declare the variable globally
geo_rag: Optional[SpatialHybridRAG] = None
naics_classifier: Optional[NAICSClassifier] = None

@app.on_event("startup")
async def startup_event():
    global geo_rag, naics_classifier
    print(f"[3/4] Connecting to MongoDB and initializing Spatial Engine (DB: {DB_NAME})...", flush=True)
    try:
        geo_rag = SpatialHybridRAG(mongo_uri=cloud_uri, db_name=DB_NAME)
        print("[4/4] Spatial Engine ready! Server is listening for requests.", flush=True)
    except Exception as e:
        print(f"Initialization failed: {e}", file=sys.stderr, flush=True)
        os._exit(1)

    # Separate try/except from the one above: a bad NAICS index build
    # shouldn't take the whole server down the way a failed Mongo connection
    # does above - match_sector_category already degrades to its Ollama-only
    # path whenever naics_classifier is None (see below), so this is safe to
    # fail non-fatally.
    try:
        naics_classifier = NAICSClassifier(geo_rag)
        print(f"[NAICS] Classifier ready — {len(naics_classifier.documents)} categories indexed "
              f"(one-time FAISS build, dense cosine similarity only - see NAICSClassifier's "
              f"docstring in spatial_rag_v1.py for why it skips BM25/hybrid search).", flush=True)
    except Exception as e:
        print(f"[NAICS] Classifier init failed — match_sector_category will fall back to "
              f"Ollama-only: {e}", flush=True)

# ─── GEOCODING ENDPOINT ──────────────────────────────────────────────────
# Strategy: try Nominatim (OpenStreetMap) first — free, no API key, no billing.
# Only fall back to Google Maps if Nominatim finds nothing, since Google
# handles ambiguous/local business names better but costs money per call.

class GeocodeRequest(BaseModel):
    query: str

# Candidates farther apart than this are treated as meaningfully different
# real places (not just two slightly different pins for the same address),
# e.g. Quebec's several "Rue de l'Église" streets in different boroughs.
GEOCODE_AMBIGUITY_DISTANCE_M = 5000
GEOCODE_MAX_CANDIDATES = 4


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _polygon_centroid(geometry: dict) -> Optional[list]:
    """Approximate [lon, lat] centroid of a GeoJSON Polygon - a plain average
    of the outer ring's points. Good enough for placing a map marker; not
    meant to be area-weighted precise."""
    try:
        ring = geometry["coordinates"][0]
        if not ring:
            return None
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return [sum(lons) / len(lons), sum(lats) / len(lats)]
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None


def _rank_geocode_candidates(raw_candidates: list) -> dict:
    """Takes a geocoder's raw ranked results (best match first) as a list of
    {"lat", "lng", "formatted"} dicts and decides whether they resolve to one
    place or several genuinely different ones. A geocoder returning 3 results
    that are all within a block of each other isn't ambiguous - it's just
    imprecise. Only flags ambiguity when 2+ results are far enough apart to
    plausibly be what the user meant by two different things."""
    if not raw_candidates:
        return {"ambiguous": False, "lat": None, "lng": None, "formatted": None}

    top = raw_candidates[0]
    distinct = [top]
    for c in raw_candidates[1:GEOCODE_MAX_CANDIDATES]:
        if _haversine_m(top["lng"], top["lat"], c["lng"], c["lat"]) > GEOCODE_AMBIGUITY_DISTANCE_M:
            distinct.append(c)

    if len(distinct) > 1:
        return {"ambiguous": True, "candidates": distinct, "lat": None, "lng": None, "formatted": None}

    return {"ambiguous": False, "candidates": [], **top}


def _geocode_with_nominatim(query: str) -> dict:
    """Returns _rank_geocode_candidates()'s shape, or the same shape with
    lat=None when nothing was found."""
    import requests as req
    try:
        res = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": GEOCODE_MAX_CANDIDATES,
                "countrycodes": "ca",
                "accept-language": "fr",
            },
            headers={
                # Nominatim's usage policy requires a real User-Agent identifying the app
                "User-Agent": "TerritoireIntelligence/1.0"
            },
            timeout=5,
        ).json()

        if not res:
            print(f"Nominatim found nothing for '{query}'", flush=True)
            return {"ambiguous": False, "candidates": [], "lat": None, "lng": None, "formatted": None}

        candidates = [
            {"lat": float(r["lat"]), "lng": float(r["lon"]), "formatted": r["display_name"]}
            for r in res
        ]
        ranked = _rank_geocode_candidates(candidates)
        if ranked["ambiguous"]:
            print(f"Nominatim found {len(ranked['candidates'])} distinct matches for '{query}' — ambiguous", flush=True)
        else:
            print(f"Nominatim resolved '{query}' → {ranked['formatted']}", flush=True)
        return ranked
    except Exception as e:
        print(f"Nominatim request failed: {e}", flush=True)
        return {"ambiguous": False, "candidates": [], "lat": None, "lng": None, "formatted": None}


def _geocode_with_google(query: str) -> dict:
    """Returns _rank_geocode_candidates()'s shape, or the same shape with
    lat=None when nothing was found or no key is set."""
    import requests as req
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("GOOGLE_MAPS_API_KEY not set — skipping Google fallback", flush=True)
        return {"ambiguous": False, "candidates": [], "lat": None, "lng": None, "formatted": None}

    try:
        res = req.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": query,
                "key": api_key,
                "region": "ca",
                "language": "fr",
            },
            timeout=5,
        ).json()

        if res.get("status") != "OK" or not res.get("results"):
            print(f"Google geocode failed: {res.get('status')} — {res.get('error_message', '')}", flush=True)
            return {"ambiguous": False, "candidates": [], "lat": None, "lng": None, "formatted": None}

        candidates = [
            {"lat": r["geometry"]["location"]["lat"], "lng": r["geometry"]["location"]["lng"], "formatted": r["formatted_address"]}
            for r in res["results"][:GEOCODE_MAX_CANDIDATES]
        ]
        ranked = _rank_geocode_candidates(candidates)
        if ranked["ambiguous"]:
            print(f"Google found {len(ranked['candidates'])} distinct matches for '{query}' — ambiguous", flush=True)
        else:
            print(f"Google resolved '{query}' → {ranked['formatted']}", flush=True)
        return ranked
    except Exception as e:
        print(f"Google geocode request failed: {e}", flush=True)
        return {"ambiguous": False, "candidates": [], "lat": None, "lng": None, "formatted": None}


def geocode_with_disambiguation(query: str) -> dict:
    """Shared entry point for every geocoding call site in this file (the
    /api/geocode endpoint, spatial_search, compare). Tries Nominatim first
    (free), falls back to Google only if Nominatim found nothing - same
    cost-conscious order as before, just candidate-aware now."""
    result = _geocode_with_nominatim(query)
    if result["lat"] is None and not result["ambiguous"]:
        print(f"Falling back to Google Maps for '{query}'", flush=True)
        result = _geocode_with_google(query)
    return result


def resolve_location(term: str, resolved: Optional["ResolvedLocation"]) -> dict:
    """Same return shape as geocode_with_disambiguation(), but skips the
    actual geocode call when the caller already resolved this exact term -
    a retry after the user picked a candidate from a needs_disambiguation
    response. Matching on the term (not just "any resolved_location present")
    means a compare with several locations only bypasses geocoding for the
    one the user actually disambiguated, not the others."""
    if resolved and resolved.term.strip().lower() == term.strip().lower():
        return {
            "ambiguous": False, "candidates": [],
            "lat": resolved.lat, "lng": resolved.lng, "formatted": resolved.formatted,
        }
    return geocode_with_disambiguation(term)


@app.post("/api/geocode")
@limiter.limit("20/minute")
def geocode_location(request: Request, payload: GeocodeRequest):
    result = geocode_with_disambiguation(payload.query)
    if result["lat"] is None and not result["ambiguous"]:
        print(f"Both geocoders failed for '{payload.query}'", flush=True)
    return result


# ─── DOCUMENT UPLOAD ENDPOINT ────────────────────────────────────────────
# Saves an uploaded file to disk and reports back its server-side path and
# detected file_type - the frontend then sends a chat message referencing
# this upload (e.g. "update the map with the document I provided"), and
# chat_turn_1 passes the path/file_type into run_agent_loop as context so it
# can call tools.parse_uploaded_document on that exact file.
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {
    ".csv": "csv",
    ".geojson": "geojson",
    ".json": "geojson",
    ".xlsx": "excel",
    ".xls": "excel",
}


@app.post("/api/upload-document")
@limiter.limit("10/minute")
def upload_document(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    file_type = ALLOWED_UPLOAD_EXTENSIONS.get(suffix)
    if not file_type:
        return {
            "status": "error",
            "message": f"Type de fichier non supporté '{suffix}' — attendu : .csv, .geojson, .json, .xlsx, ou .xls.",
        }

    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = UPLOAD_DIR / safe_name
    with dest_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    print(f"[Upload] Saved '{file.filename}' -> {dest_path} (file_type={file_type})", flush=True)

    return {
        "status": "success",
        "file_path": str(dest_path),
        "file_type": file_type,
        "original_filename": file.filename,
    }


# ─── AUTONOMOUS PIPELINE ENDPOINTS ──────────────────────────────────────

class ResolvedLocation(BaseModel):
    term: str
    lat: float
    lng: float
    formatted: str


class HistoryTurn(BaseModel):
    message: str
    action_type: Optional[str] = None
    location_query: Optional[str] = None
    panel_name: Optional[str] = None
    layers_to_activate: Optional[List[str]] = None
    naics_query: Optional[str] = None


class UploadedFileRef(BaseModel):
    file_path: str
    file_type: str


class ChatTurn1Request(BaseModel):
    message: str
    # Set when this request is a retry after the user picked one candidate
    # from a needs_disambiguation response. Geocoding a formatted address
    # (comma-separated: "Street, Borough, City, Region, Country") back through
    # LOCATION_PATTERN would just re-truncate at the first comma and loop
    # forever - so a resolved retry skips geocoding for `term` entirely and
    # uses these coordinates directly instead.
    resolved_location: Optional[ResolvedLocation] = None
    # Compact record of recent turns (frontend-maintained, capped client-side
    # to the last few) - lets the LLM resolve follow-ups like "show retail
    # there instead" that name no location of their own. Deliberately NOT the
    # raw report text - just the resolved intent, so the prompt stays small.
    history: Optional[List[HistoryTurn]] = None
    # Set when this message follows a POST /api/upload-document call - the
    # frontend passes back the exact path/file_type it got from that
    # response. Presence of this field routes straight to run_agent_loop
    # (bypassing fast_extract_intent/the taxonomy entirely), since an
    # uploaded file is a far stronger signal than any keyword match.
    uploaded_file: Optional[UploadedFileRef] = None
    # An arbitrary polygon the user drew or selected directly on the map
    # dashboard - a closed GeoJSON ring of [lon, lat] pairs, the same shape
    # polygon_filter/count_businesses_in_da already accept. Deliberately NOT
    # limited to a dissemination area or an isochrone: those are the only
    # two polygon sources run_agent_loop could previously reconstruct on its
    # own (see _last_resolved_polygon), so "list what's in the shape I just
    # drew" had no way to reach the agent loop at all before this field
    # existed. See _augment_message_with_polygon and run_agent_loop's
    # preselected_polygon parameter for how it's actually used.
    selected_polygon: Optional[List[List[float]]] = None

# ── Fast, deterministic intent extraction ──────────────────────────────────
#Turn 1 only needs 6 small, mostly-enumerated fields. Regex extracts fast and free. 
#
# Returns None (instead of a dict) when the message doesn't match a
# recognizable "from/near/around <place>" pattern — in that case we fall back
# to the LLM, since free-text location extraction is  hard to do
# reliably with regex.

# ═══════════════════════════════════════════════════════════════════════════
# INTENT TAXONOMY — single source of truth for Turn 1 keyword matching.
#
# THINGS SHOULD BE ADDED HERE IF THE PURPOSE OF THE MODEL EXPANDS 
# Every category that used to live in its own separate list (LAYER_KEYWORDS,
# PANEL_KEYWORDS, ANALYZE_KEYWORDS, ALL_ANALYSIS_KEYWORDS, SPATIAL_KEYWORDS,
# ZONE_KEYWORDS) now lives here, in one place, each entry documented with the
# example phrases it's meant to catch. 
# The individual lookup structures other code depends on (PANEL_KEYWORDS,
# LAYER_KEYWORDS, etc.) are derived from this table below, so nothing else
# in the file needs to change — this is a refactor, not a behavior change.
# ═══════════════════════════════════════════════════════════════════════════
INTENT_TAXONOMY = [
    # ── action_type ──────────────────────────────────────────────────────
    {"category": "action", "value": "analyze",
     "keywords": ["analy", "competit", "concurren", "retention", "rétention", "score", "implant", "viabilité", "feasib"],
     "examples": ["analyze the competition near X", "quelle est la rétention à X", "site score for X"]},
    {"category": "action", "value": "spatial_search",
     "keywords": ["closest", "nearest", "proche", "près", "top"],
     "examples": ["closest empty locals to X", "les 5 commerces les plus proches de X"]},
    {"category": "action", "value": "compare",
     "keywords": ["compar", "vs", "versus", "meilleur", "better", "lequel", "quel est le meilleur"],
     "examples": ["compare Bromont and Saint-Hyacinthe for a bakery",
                  "quel est le meilleur secteur entre Bromont et Saint-Hyacinthe"]},
    # "explore" is the default action_type when nothing above matches.

    # ── panel_name (only relevant when action_type == "analyze") ────────
    {"category": "panel", "value": "competitive",
     "keywords": ["competit", "concurren"],
     "examples": ["analyze the competition near X", "concurrence à X"]},
    {"category": "panel", "value": "retention",
     "keywords": ["retention", "rétention", "fuite"],
     "examples": ["retention analysis for X", "fuite commerciale à X"]},
    {"category": "panel", "value": "siteScore",
     "keywords": ["site score", "score d'emplacement", "emplacement"],
     "examples": ["site score for X", "score d'emplacement à X"]},
    {"category": "panel", "value": "implementation",
     "keywords": ["implementation", "implantation", "faisabilité", "feasib"],
     "examples": ["can I open a restaurant at X", "faisabilité d'implantation à X"]},
    {"category": "panel", "value": "all",
     "keywords": ["toutes les analyses", "analyse complète", "analyse globale", "rapport complet",
                  "tous les piliers", "4 piliers", "quatre piliers", "toute l'analyse",
                  "full analysis", "complete analysis", "all analyses", "everything"],
     "examples": ["give me the full analysis of X", "analyse complète de X"]},

    # ── layers_to_activate ───────────────────────────────────────────────
    {"category": "layer", "value": "commerces",
     "keywords": ["commerce", "business", "concurrent", "competitor", "competit"],
     "examples": ["show me businesses near X"]},
    {"category": "layer", "value": "locauxVacants",
     "keywords": ["vacant", "empty", "locaux vides", "available", "libre", "louer"],
     "examples": ["show me vacant locals in X", "locaux à louer à X"]},
    {"category": "layer", "value": "statcanDA",
     "keywords": ["population", "demographic", "income", "revenu", "démographie", "habitant",
                  "market", "marché", "purchasing power", "pouvoir d'achat", "wealthier",
                  "richer", "clientele", "clientèle", "buying power"],
     "examples": ["show me demographics for X", "which market is bigger, X or Y"]},
    {"category": "layer", "value": "zonage",
     "keywords": ["zoning", "permit", "zonage", "permis"],
     "examples": ["show me zoning for X"]},

    # ── zone drawing (applies regardless of action_type) ────────────────
    {"category": "zone", "value": True,
     "keywords": ["zone", "isochrone", "rayon", "chalandise", "à distance de",
                  "dans un rayon", "autour de", "within", "catchment", "radius"],
     "examples": ["vacant locals within 10 minutes of X", "commerces dans un rayon de 5 min de X"]},
]


def _hits(msg: str, keywords: list) -> int:
    """Counts how many keywords from a list appear in the message — used both
    to decide a match and to score confidence (0 hits, 1 hit, or several)."""
    return sum(1 for k in keywords if k in msg)


def _taxonomy_keywords(category: str, value=None) -> list:
    for entry in INTENT_TAXONOMY:
        if entry["category"] == category and (value is None or entry["value"] == value):
            return entry["keywords"]
    return []


# Derived lookup structures — kept under their original names since other
# code in this file (e.g. the spatial-search interception block) already
# references PANEL_KEYWORDS directly.
ANALYZE_KEYWORDS = _taxonomy_keywords("action", "analyze")
SPATIAL_KEYWORDS = _taxonomy_keywords("action", "spatial_search")
COMPARE_KEYWORDS = _taxonomy_keywords("action", "compare")
PANEL_KEYWORDS = {e["value"]: e["keywords"] for e in INTENT_TAXONOMY if e["category"] == "panel" and e["value"] != "all"}
ALL_ANALYSIS_KEYWORDS = _taxonomy_keywords("panel", "all")
LAYER_KEYWORDS = {e["value"]: e["keywords"] for e in INTENT_TAXONOMY if e["category"] == "layer"}
ZONE_KEYWORDS = _taxonomy_keywords("zone", True)

LOCATION_PATTERN = re.compile(
    r'(?:from|near|around|of|de|près de|à côté de|autour de)\s+'
    r'([A-Za-zÀ-ÿ0-9\s\-\']+?)(?:\s+within|\s+in\s+\d|\.|$|,)',
    re.IGNORECASE,
)
MINUTES_PATTERN = re.compile(r'(\d+)\s*(?:min|minute)', re.IGNORECASE)

# Matches a monthly budget cap, e.g. "under $2000/month", "moins de 2000$/mois",
# "budget de 1500 $ par mois", "max 3000$/mois". Requires an explicit monthly
# unit so a plain number elsewhere in the message (a NAICS code, a radius in
# meters) is never misread as a price.
PRICE_PATTERN = re.compile(
    r'(\d[\d\s,]*(?:\.\d+)?)\s*\$?\s*(?:/|par\s+|per\s+)?\s*(?:mois|month)\b',
    re.IGNORECASE,
)

# Search radius for the spatial_search interception block below — same value
# the old raw $geoNear query used before it was replaced with geo_rag's
# hybrid (spatial + semantic) retrieval.
DEFAULT_SPATIAL_SEARCH_RADIUS_M = 5000

# Fields pulled from get_smart_demographics() for compare — only when the
# query's own layers_to_activate already includes statcanDA (population/
# income/market keywords), so demographics are fetched dynamically per
# query rather than on every comparison regardless of relevance.
DEMOGRAPHIC_FIELDS = ["Population_2021", "menages_total", "revenu_median", "age_moyen"]

# layers_to_activate name -> Mongo collection name, for spatial_search.
SPATIAL_SEARCH_COLLECTIONS = {
    "commerces": "quebec_businesses",
    "locauxVacants": "locaux_vacants",
}

# Trigger words LOCATION_PATTERN anchors on — stripped when isolating the
# "what" part of a spatial_search query from the "where" part below.
_LOCATION_TRIGGER_WORDS = re.compile(
    r'\b(?:from|near|around|of|de|près de|à côté de|autour de)\b', re.IGNORECASE,
)

# Generic words that carry no category signal on their own, so "closest
# vacant locals near X" (no real category named) isn't treated the same as
# "closest yoga studio near X" (a real category) when deciding whether to
# apply a free-text category boost in spatial_search.
CATEGORY_STOPWORDS = {
    "closest", "nearest", "proche", "proches", "plus", "les", "des", "le", "la",
    "de", "du", "vacant", "vacants", "empty", "locals", "locaux", "commerce",
    "commerces", "business", "businesses", "near", "around", "show", "me",
    "find", "trouve", "trouvez", "montre", "montrez", "a", "à", "of", "to",
    "the", "within", "minutes", "minute", "min", "km", "m", "radius", "rayon",
    "zone", "distance", "away", "naics", "code", "in", "dans",
    # compare-specific filler — trigger words and list separators that
    # survive location-stripping in a "compare X and Y for Z" query.
    "compare", "comparer", "comparez", "between", "entre", "and", "et", "or",
    "ou", "vs", "versus", "meilleur", "meilleure", "better", "which", "quel",
    "quelle", "est", "for", "pour", "is",
}


def _extract_category_phrase(raw_message: str, location_query) -> Optional[str]:
    """Best-effort free-text category phrase for boosting spatial-search
    results, e.g. "yoga studio" out of "closest yoga studio near Bromont".
    location_query may be a single string (spatial_search) or a list of
    strings (compare, which has several locations to strip). Returns None
    when nothing but generic search words is left, so callers skip category
    boosting instead of forcing a meaningless match."""
    text = raw_message or ""
    locations = [location_query] if isinstance(location_query, str) else (location_query or [])
    for loc in locations:
        if loc:
            text = re.sub(re.escape(loc), "", text, flags=re.IGNORECASE)
    text = _LOCATION_TRIGGER_WORDS.sub("", text)
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    meaningful = [w for w in words if w.lower() not in CATEGORY_STOPWORDS]
    phrase = " ".join(meaningful).strip()
    return phrase if len(phrase) > 2 else None


# Two natural phrasings this fast path covers: an explicit compare/between
# trigger followed by a location list ("compare X and Y", "between X and Y",
# "entre X et Y"), and a bare "X vs Y" with no trigger word at all. Anything
# less explicit (implicit lists, "which of these is better") escalates to
# the LLM, same as everywhere else in this taxonomy. "between/entre" is
# tried first since it's the more precise anchor when a message also has
# descriptive text before it (e.g. "compare the competition between X and Y").
_BETWEEN_SPAN_PATTERN = re.compile(
    r'(?:between|entre)\s+(.+?)(?:\s+for\b|\s+pour\b|\?|\.|$)', re.IGNORECASE,
)
_COMPARE_SPAN_PATTERN = re.compile(
    r'(?:compare|comparer|compar\w*)\s+(?:the\s+|les\s+)?(.+?)(?:\s+for\b|\s+pour\b|\?|\.|$)',
    re.IGNORECASE,
)
_VS_LOCATIONS_PATTERN = re.compile(
    r'([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']*?)\s+(?:vs\.?|versus)\s+'
    r'([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']*?)(?:\s+for\b|\s+pour\b|\?|\.|$|,)',
    re.IGNORECASE,
)
_COMPARE_LIST_SPLIT = re.compile(r'\s*,\s*|\s+(?:and|et|or|ou)\s+', re.IGNORECASE)


def _extract_comparison_locations(message: str) -> Optional[list]:
    """Pulls 2-4 location names out of a comparison query. Returns None when
    the phrasing isn't explicit enough to trust — the caller escalates to
    the LLM in that case rather than guessing."""
    for pattern in (_BETWEEN_SPAN_PATTERN, _COMPARE_SPAN_PATTERN):
        span_match = pattern.search(message)
        if span_match:
            parts = [p.strip() for p in _COMPARE_LIST_SPLIT.split(span_match.group(1)) if p.strip()]
            if len(parts) >= 2:
                return parts[:4]

    vs_match = _VS_LOCATIONS_PATTERN.search(message)
    if vs_match:
        loc1, loc2 = vs_match.group(1).strip(), vs_match.group(2).strip()
        if loc1 and loc2:
            return [loc1, loc2]

    return None


# Cross-collection queries ("areas with high income but low restaurant
# density") need to search across many sub-areas and rank by TWO independent
# criteria (one demographic, one business) at once - too open-ended for
# regex to extract reliably. This gate only decides WHETHER to force
# escalation to the Turn-1 LLM (which has its own JSON schema for this
# action_type below); it never tries to parse the criteria itself.
_CROSS_QUERY_CONTRAST_WORDS = {
    "but", "mais", "whereas", "alors que", "tandis que", "et peu", "et beaucoup",
    "with few", "with many", "with low", "with high", "avec peu", "avec beaucoup",
}
_CROSS_QUERY_AREA_WORDS = {
    "area", "areas", "zone", "zones", "secteur", "secteurs", "quartier",
    "quartiers", "neighborhood", "neighborhoods", "endroit", "endroits",
}


def _looks_like_cross_query(msg: str) -> bool:
    has_demographic_signal = _hits(msg, LAYER_KEYWORDS.get("statcanDA", [])) > 0
    has_business_signal = (
        _hits(msg, LAYER_KEYWORDS.get("commerces", [])) > 0
        or any(k in msg for k in ["density", "densité", "restaurant", "café", "boutique"])
    )
    has_contrast = any(k in msg for k in _CROSS_QUERY_CONTRAST_WORDS)
    has_area_phrase = any(k in msg for k in _CROSS_QUERY_AREA_WORDS)
    return has_demographic_signal and has_business_signal and (has_contrast or has_area_phrase)


PANEL_LABELS = {
    "competitive": "CONCURRENCE",
    "retention": "RÉTENTION COMMERCIALE",
    "siteScore": "SCORE D'EMPLACEMENT",
    "implementation": "IMPLANTATION",
}

# ── Pre-flight validation before calling the Turn-2 LLM ─────────────────────
# If a panel's data is mostly placeholder values ("N/A", "Donnée non
# calculée"...), that means the frontend sent the payload before the panel
# actually finished computing — the exact bug that produced a fully-written
# report about an implementation score that was never actually calculated.
# Rather than let the LLM narrate around missing data, catch it here and
# tell the caller plainly instead of spending a Claude call on it.
EMPTY_PLACEHOLDER_VALUES = {
    "n/a", "na", "donnée non calculée", "aucun point sélectionné",
    "non déterminé", "non disponible", "",
}


def _fraction_empty(d: dict) -> float:
    """Fraction of a metrics dict's values that are empty/placeholder."""
    if not d:
        return 1.0
    empty = 0
    for v in d.values():
        if v is None:
            empty += 1
        elif isinstance(v, str) and v.strip().lower() in EMPTY_PLACEHOLDER_VALUES:
            empty += 1
    return empty / len(d)


# The single field that most directly indicates "this panel actually
# finished computing" — mirrors isPanelReady() on the frontend, which gates
# on exactly these same headline metrics rather than an arbitrary field count.
PANEL_HEADLINE_FIELD = {
    "competitive": "saturation_marche_ism",
    "retention": "pouvoir_achat_disponible_pad",
    "siteScore": "score_global",
    "implementation": "score_composite_spc",
}


def _is_headline_empty(panel: str, d: dict) -> bool:
    field = PANEL_HEADLINE_FIELD.get(panel)
    if not field:
        return False
    v = d.get(field)
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in EMPTY_PLACEHOLDER_VALUES:
        return True
    return False


# Matches an explicit NAICS code mention, e.g. "NAICS 722", "code 4451", "naics 44-45".
# Free-text category descriptions ("restaurants", "food retail") are NOT
# handled by this regex — those go through keyword matching against the
# app's REAL sector/subsector list on the frontend first (findBestTaxonomyMatch),
# and only fall back to an LLM if that keyword match fails. See
# /api/match-sector-category below, which the frontend calls for that fallback.
NAICS_PATTERN = re.compile(
    r'naics\s*(?:code)?\s*[:#]?\s*(\d{2,6}(?:-\d{2})?)|code\s*naics\s*[:#]?\s*(\d{2,6}(?:-\d{2})?)',
    re.IGNORECASE,
)

# Sanity-check prefixes for a NUMERIC NAICS code the user explicitly typed.
# This is a coarse, 2-digit-sector check — not a full code lookup (Python
# doesn't have access to the app's real taxonomy, which lives in Mongo/the
# categories API) — but it's enough to catch a clearly-invalid code like
# "999" and tell the user plainly instead of silently doing nothing.
VALID_NAICS_PREFIXES = {
    "11": "Agriculture", "21": "Mining", "22": "Utilities",
    "23": "Construction", "31": "Manufacturing", "32": "Manufacturing",
    "33": "Manufacturing", "42": "Wholesale Trade",
    "44": "Retail Trade", "45": "Retail Trade", "48": "Transport",
    "49": "Transport", "51": "Information", "52": "Finance",
    "53": "Real Estate", "54": "Professional Services",
    "55": "Management", "56": "Admin Services", "61": "Education",
    "62": "Health Care", "71": "Arts/Entertainment",
    "72": "Accommodation/Food", "81": "Other Services",
    "92": "Public Administration",
}


def check_naics_code(code: str) -> Optional[str]:
    """
    Validates a numeric NAICS code the user explicitly typed.
    Returns None if it looks valid (2-digit prefix recognized), or a plain
    failure message if it doesn't match any known sector — so the frontend
    can show the user a clear "NAICS not found" toast instead of silently
    falling back to something else.
    """
    print(f"[NAICS] Numeric code detected: '{code}' — checking against known prefixes...", flush=True)
    prefix = code[:2] if len(code) >= 2 else code
    if prefix in VALID_NAICS_PREFIXES:
        print(f"[NAICS] '{code}' is valid (sector: {VALID_NAICS_PREFIXES[prefix]})", flush=True)
        return None  # valid, no error
    print(f" [NAICS] '{code}' not recognized — prefix '{prefix}' doesn't match any known sector", flush=True)
    return f"Code NAICS \"{code}\" introuvable — aucun secteur ne correspond à ce code."


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# Calibration reference: NAICSClassifier.classify_business_type's own
# docstring (spatial_rag_v1.py) measured a near-exact vocabulary match
# ("restaurant" -> NAICS 7225) at confidence ~0.6, a good-but-inexact one
# ("cafe" -> 7225) at ~0.44, and an outright wrong match still landing at
# ~0.30-0.35 - there is NO cleanly-separating threshold yet (that docstring
# says so explicitly, and warns against wiring one in without a larger
# labeled query set). 0.4 sits between the documented "wrong" and
# "good-but-inexact" bands as a first-pass floor, not a validated cutoff.
# The real safety net below is the candidates-grounding check, not this
# number: a semantic match is only ever accepted if it's literally one of
# the categories the frontend already considers valid for this request -
# the exact same hallucination guard the Ollama fallback further down
# already applies to its own matches.
_NAICS_SEMANTIC_CONFIDENCE_FLOOR = 0.4


@app.post("/api/match-sector-category")
@limiter.limit("30/minute")
def match_sector_category(request: Request, payload: dict):
    """
    Grounded category matching for free-text business descriptions. The
    frontend calls this ONLY after its own keyword match against the real
    sector/subsector taxonomy (findBestTaxonomyMatch) has already failed to
    find anything — this is a last resort, not the first attempt.

    Two-stage fallback from there:
    1. NAICSClassifier (spatial_rag_v1.py) - a semantic search over db.naics
       built ONCE at server startup (see its docstring for why it's dense
       cosine similarity only, never BM25/hybrid_semantic_search - that
       combination was measured to be both ~15-20s/call slower AND less
       accurate for this specific corpus). Free, instant, no LLM call.
       Only accepted when the matched label is literally one of
       `candidates` (see _NAICS_SEMANTIC_CONFIDENCE_FLOOR above) - never
       returned ungrounded.
    2. Ollama (llama3.2:3b, local - NOT the Claude API, so there is no
       per-call billing to reduce here) - unchanged from before, run only
       for whatever the semantic pass didn't confidently resolve.

    Critically, both stages are GROUNDED: the match always comes from the
    actual candidate list the frontend sends (the app's real sector +
    subsector labels), never invented from general knowledge. That's the
    same principle used everywhere else in this pipeline — never let a
    match answer with a category that doesn't actually exist in the app.

    Request body: {
        "message": "give me something in the food business",
        "candidates": ["Hébergement et restauration", "Restaurants", "Commerce de détail", ...]
    }
    Response: {"match": "Restaurants", "confidence": 0.8, "method": "semantic_search" | "ollama_fallback"}
    or {"match": null, "confidence": 0}
    """
    import requests

    message = payload.get("message", "").strip()
    candidates = payload.get("candidates", [])

    if not message or not candidates:
        print("[NAICS] Missing message or candidate list — skipping", flush=True)
        return {"match": None, "confidence": 0.0}

    # STEP 1: semantic search first - free, instant, no LLM call at all.
    if naics_classifier is not None:
        semantic_matches = naics_classifier.classify_business_type(message, top_k=5)
        accepted = next(
            (m for m in semantic_matches
             if m["label"] in candidates and m["confidence"] >= _NAICS_SEMANTIC_CONFIDENCE_FLOOR),
            None,
        )
        if accepted:
            print(f"[NAICS] Semantic match: \"{message}\" → \"{accepted['label']}\" "
                  f"(confidence={accepted['confidence']:.2f}, NAICS {accepted['code']})", flush=True)
            return {
                "match": accepted["label"],
                "confidence": accepted["confidence"],
                "method": "semantic_search",
            }
        print(f"[NAICS] No confident grounded semantic match for \"{message}\" among "
              f"{len(candidates)} candidate(s) — escalating to Ollama...", flush=True)

    print(f"[NAICS] Step 3/3 — keyword+semantic match failed, escalating to Ollama ({OLLAMA_MODEL})...", flush=True)
    print(f"[NAICS] Message: '{message}' | candidate count: {len(candidates)}", flush=True)

    # Cap candidate list size sent to the prompt — taxonomies can be large,
    # and this keeps the call fast.
    candidate_list = "\n".join(f"- {c}" for c in candidates[:300])

    prompt = f"""A user typed this request about a business search: "{message}"

Here is the exact list of real business categories available in this app:
{candidate_list}

Which ONE category from the list above (if any) best matches what the user is describing?
You MUST pick from the list exactly as written, or return null if nothing fits well.

Respond with ONLY valid JSON, nothing else:
{{"match": "<exact category from the list>" or null, "confidence": 0.0-1.0}}
"""

    try:
        print(f"[NAICS] Calling Ollama at {OLLAMA_URL} (model: {OLLAMA_MODEL})...", flush=True)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=15,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        print(f"[NAICS] Ollama raw response: {raw_text[:200]}", flush=True)

        result = json.loads(raw_text)
        match = result.get("match")
        confidence = result.get("confidence", 0.0)

        # Guard against hallucination: only accept the match if it's
        # actually in the candidate list we sent, verbatim.
        if match and match in candidates and confidence >= 0.5:
            print(f"[NAICS] Ollama matched \"{message}\" → \"{match}\" (confidence: {confidence:.0%})", flush=True)
            return {"match": match, "confidence": confidence, "method": "ollama_fallback"}
        elif match and match not in candidates:
            print(f"[NAICS] Ollama returned \"{match}\" but it's NOT in the candidate list — rejecting (hallucination guard)", flush=True)
            return {"match": None, "confidence": 0.0, "method": "ollama_fallback"}
        else:
            print(f"[NAICS] Ollama found no confident match for \"{message}\" (best guess: {match}, confidence: {confidence:.0%})", flush=True)
            return {"match": None, "confidence": confidence, "method": "ollama_fallback"}

    except requests.exceptions.ConnectionError:
        print(f"[NAICS] Ollama not reachable at {OLLAMA_URL} — is Ollama running locally? "
              f"(Start it with: ollama serve, and pull the model with: ollama pull {OLLAMA_MODEL})", flush=True)
        return {"match": None, "confidence": 0.0, "method": "ollama_fallback"}
    except requests.exceptions.Timeout:
        print(f"[NAICS] Ollama call timed out after 15s", flush=True)
        return {"match": None, "confidence": 0.0, "method": "ollama_fallback"}
    except json.JSONDecodeError as e:
        print(f"[NAICS] Ollama returned invalid JSON: {e} | raw: {raw_text[:200] if 'raw_text' in dir() else '(unavailable)'}", flush=True)
        return {"match": None, "confidence": 0.0, "method": "ollama_fallback"}
    except Exception as e:
        print(f"[NAICS] Ollama call failed unexpectedly: {e}", flush=True)
        return {"match": None, "confidence": 0.0, "method": "ollama_fallback"}


def fast_extract_intent(message: str) -> Optional[dict]:
    """
    Fast regex-based intent extraction with confidence-aware fallback to LLM.
    NAICS handling: numeric codes ("NAICS 722") are extracted and validated
    here; free-text category descriptions are NOT handled here at all — they
    go through the frontend's keyword-then-LLM matching against the real
    taxonomy (see match_sector_category above).
    """
    msg = message.lower()
    resolution_notes = []  # visible in terminal logs — ties resolution decisions to observability

    # ── Cross-collection queries ("areas with high income but low restaurant
    # density") need real reasoning to extract two independent ranking
    # criteria - always escalate to the LLM rather than guess at a shape
    # regex was never going to get right.
    if _looks_like_cross_query(msg):
        return None

    # ── Comparison intent: a fundamentally different shape (2+ locations,
    # no single "near X" trigger) - checked before the single-location gate
    # below, since LOCATION_PATTERN wouldn't match "compare X and Y" at all.
    if _hits(msg, COMPARE_KEYWORDS) > 0:
        locations = _extract_comparison_locations(message)
        if not locations or len(locations) < 2:
            return None  # ambiguous comparison — escalate to LLM

        layers_to_activate = [layer for layer, kws in LAYER_KEYWORDS.items() if _hits(msg, kws) > 0]

        naics_query = None
        naics_error = None
        naics_match = NAICS_PATTERN.search(message)
        if naics_match:
            code = naics_match.group(1) or naics_match.group(2)
            error = check_naics_code(code)
            if error:
                naics_error = error
                resolution_notes.append(f"NAICS code '{code}' not recognized")
            else:
                naics_query = code

        price_match = PRICE_PATTERN.search(msg)
        max_budget = None
        if price_match:
            raw_price = price_match.group(1).replace(" ", "").replace(",", ".")
            try:
                max_budget = float(raw_price)
            except ValueError:
                max_budget = None

        return {
            "action_type": "compare",
            "locations": locations,
            "layers_to_activate": layers_to_activate,
            "naics_query": naics_query,
            "naics_error": naics_error,
            "max_budget": max_budget,
            "raw_message": message,
            "resolution_notes": resolution_notes,
        }

    location_match = LOCATION_PATTERN.search(message)
    if not location_match:
        return None  # can't extract location reliably — let the LLM handle it

    # ── action_type, with conflict detection ─────────────────────────────
    # If the message has real signal for BOTH "analyze" and "spatial_search"
    # (e.g. "closest competitive analysis near X"), don't silently pick one —
    # that's exactly the kind of ambiguity regex is bad at resolving but an
    # LLM can reason through. Route to the LLM instead of guessing.
    analyze_hits = _hits(msg, ANALYZE_KEYWORDS)
    spatial_hits = _hits(msg, SPATIAL_KEYWORDS)

    if analyze_hits > 0 and spatial_hits > 0:
        return None  # conflicting signals — escalate to LLM verification

    if analyze_hits == 0 and spatial_hits == 0:
        # No real action_type signal at all - this used to silently default to
        # "explore" here, which meant messages like "find vacant offices within
        # 500m of X" or "list the businesses in this area" (a location, but no
        # "closest"/"analyze"-style keyword) resolved to a no-op pan+layer-toggle
        # instead of ever reaching the LLM classifier or run_agent_loop. Escalate
        # instead of guessing - the LLM prompt below still has "explore" as a
        # valid outcome for genuinely simple requests, it just isn't a silent
        # regex default anymore.
        return None

    action_type = "analyze" if analyze_hits > 0 else "spatial_search"

    # ── panel_name, with multi-panel detection ───────────────────────────
    panel_name = None
    if action_type == "analyze":
        if _hits(msg, ALL_ANALYSIS_KEYWORDS) > 0:
            panel_name = "all"
        else:
            matched_panels = [panel for panel, kws in PANEL_KEYWORDS.items() if _hits(msg, kws) > 0]
            if len(matched_panels) == 0:
                # No specific pillar named (e.g. "analyze Saint-Hyacinthe") — a
                # bare analysis request more plausibly means "show me
                # everything" than "show me only competition".
                panel_name = "all"
                resolution_notes.append("no specific pillar named — defaulted to full report")
            elif len(matched_panels) == 1:
                panel_name = matched_panels[0]
            else:
                # Multiple pillars explicitly named (e.g. "look at retention
                # and competition") — this is real multi-intent. Rather than
                # arbitrarily picking the first match and silently dropping
                # the rest, treat it as a request for the full report, which
                # covers everything the user actually asked for.
                panel_name = "all"
                resolution_notes.append(f"multiple pillars named ({', '.join(matched_panels)}) — treated as full report")

    layers_to_activate = [layer for layer, kws in LAYER_KEYWORDS.items() if _hits(msg, kws) > 0]

    minutes_match = MINUTES_PATTERN.search(msg)
    minutes = int(minutes_match.group(1)) if minutes_match else 10

    mode = (
        "walking" if any(k in msg for k in ["walk", "à pied", "marche"])
        else "cycling" if any(k in msg for k in ["bike", "vélo", "cycl"])
        else "driving"
    )

    # NAICS code, if explicitly stated (e.g. "NAICS 722"). If the user typed
    # a numeric code, validate it — an unrecognized code becomes a clear
    # failure message rather than silently being applied or dropped.
    naics_query = None
    naics_error = None
    naics_match = NAICS_PATTERN.search(message)
    if naics_match:
        print("[NAICS] Step 1/3 — numeric code found in message", flush=True)
        code = naics_match.group(1) or naics_match.group(2)
        error = check_naics_code(code)  # prints its own step-2 validation result
        if error:
            naics_error = error
            resolution_notes.append(f"NAICS code '{code}' not recognized")
        else:
            naics_query = code
    else:
        print("[NAICS] Step 1/3 — no numeric code in message; if this is a category "
              "request, the frontend will try keyword matching (step 2), then Ollama (step 3)", flush=True)

    # A zone should be drawn if the user gave an explicit time/distance
    # ("within 10 minutes") or used zone-implying language — regardless of
    # whether this turns out to be an "explore" or "analyze" request.
    needs_zone = bool(minutes_match) or _hits(msg, ZONE_KEYWORDS) > 0

    # Monthly budget cap, if explicitly stated ("under $2000/month",
    # "moins de 2000$/mois"). Only used today by spatial_search over vacant
    # locals — see PRICE_PATTERN's docstring for why it requires a monthly unit.
    price_match = PRICE_PATTERN.search(msg)
    max_budget = None
    if price_match:
        raw_price = price_match.group(1).replace(" ", "").replace(",", ".")
        try:
            max_budget = float(raw_price)
        except ValueError:
            max_budget = None

    return {
        "action_type": action_type,
        "location_query": location_match.group(1).strip(),
        "panel_name": panel_name,
        "layers_to_activate": layers_to_activate,
        "minutes": minutes,
        "mode": mode,
        "naics_query": naics_query,
        "naics_error": naics_error,
        "max_budget": max_budget,
        "raw_message": message,
        "needs_zone": needs_zone,
        "resolution_notes": resolution_notes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP — reason -> act -> observe -> reason again.
#
# The fixed action_types above (spatial_search, compare, cross_query, ...)
# each resolve in a single deterministic pass because the shape of the work
# is known in advance. Some questions don't have a fixed shape — "which
# address is best for a bakery near X" needs to geocode candidates, search/
# analyze each one, and compare, where each step's plan depends on the
# previous step's result. That can't be scripted as one more elif branch, so
# it's handed to Claude with the full tool set and let it drive.
# ═══════════════════════════════════════════════════════════════════════════
MAX_AGENT_ITERATIONS = 8
AGENT_MODEL = "claude-haiku-4-5-20251001"

AGENT_SYSTEM_PROMPT = (
    "You are a territorial intelligence assistant with tools to geocode places, "
    "search for businesses/vacant spaces, run pillar analyses, fetch demographics, "
    "and control the map UI. Call tools iteratively: run one, examine its result, "
    "then decide the next step - or give a final answer once you have enough "
    "information. Reply in the same language as the user's message. Never invent "
    "data you haven't actually retrieved through a tool call. When a tool's result "
    "contains a value you need for your NEXT tool call (e.g. a polygon_ring or "
    "geometry from get_dissemination_area_by_code/find_dissemination_area, to pass "
    "into count_businesses_in_da or polygon_filter), copy that exact value into the "
    "next call's arguments - never call a tool with an empty argument object or omit "
    "a required parameter."
)

# Fix for an observed failure mode: after resolving a DA's polygon, Claude
# sometimes calls count_businesses_in_da/polygon_filter with an EMPTY
# argument object instead of the polygon it just received - repeatedly,
# burning the whole iteration budget with no answer. The system prompt above
# now says not to do this, but that's not a guarantee, so this is a backstop:
# if one of these two tools is called with its geometry argument missing,
# auto-fill it from the most recent polygon this same loop actually
# resolved, rather than failing outright.
_POLYGON_ARG_NAME = {
    "count_businesses_in_da": "da_geometry",
    "polygon_filter": "polygon_ring",
}


def _last_resolved_polygon(partial_results: list):
    """Most recent polygon resolved by a prior call in THIS loop, in
    whichever shape its source tool produced - get_dissemination_area_by_code
    already returns a flat polygon_ring; find_dissemination_area returns a
    list of candidate areas, each with its own full 'geometry'. Both shapes
    are valid inputs to count_businesses_in_da/polygon_filter (see
    tools._as_polygon_geometry), so no reshaping is needed here."""
    for call in reversed(partial_results):
        result = call.get("result") or {}
        if call.get("tool") == "get_dissemination_area_by_code":
            ring = result.get("polygon_ring")
            if ring:
                return ring
        elif call.get("tool") == "find_dissemination_area":
            areas = result.get("areas") or []
            if areas and areas[0].get("geometry"):
                return areas[0]["geometry"]
    return None


# ChatTurn1Request.selected_polygon carries an arbitrary polygon the user
# drew/selected on the map dashboard - NOT a dissemination area or an
# isochrone, and not something run_agent_loop can reconstruct on its own the
# way _last_resolved_polygon reconstructs a DA boundary. Without this, "list
# what's in the shape I just drew" had no path into the agent loop at all:
# the model has no tool that geocodes a hand-drawn shape, and
# _last_resolved_polygon only ever looks at DA-sourced tool results.
def _augment_message_with_polygon(message: str, selected_polygon: Optional[list]) -> str:
    """Appends the same kind of system-context note the uploaded_file branch
    already prepends for parse_uploaded_document - tells Claude a custom
    polygon exists so it doesn't spend a turn geocoding a location or
    searching for a dissemination area for a request that names neither.
    The actual coordinates don't need to be restated in the note itself:
    run_agent_loop's preselected_polygon parameter auto-fills them into
    polygon_filter/count_businesses_in_da whenever the model leaves that
    argument out, exactly like the existing DA-fallback autofill below."""
    if not selected_polygon:
        return message
    return (
        f"{message}\n\n"
        f"[Contexte système : l'utilisateur a sélectionné/dessiné une zone personnalisée sur "
        f"la carte - PAS nécessairement une aire de diffusion (DA) ni une zone de chalandise. "
        f"Son contour sera fourni automatiquement à polygon_filter (polygon_ring) ou "
        f"count_businesses_in_da (da_geometry) si vous omettez cet argument. N'essayez PAS de "
        f"géocoder un lieu ou de chercher une aire de diffusion pour cette demande - utilisez "
        f"directement cette zone dessinée.]"
    )


# count_businesses_in_da/polygon_filter both default `collection` to
# "quebec_businesses" - fine when the model sets it deliberately, but the
# same empty-arguments failure above showed that default can mask a wrong
# answer: a "vacant locals" query got auto-filled a real polygon (by
# _last_resolved_polygon) and then silently counted BUSINESSES instead,
# because `collection` was never set either and nothing caught that. Only
# apply this when the model left `collection` out entirely - an explicit
# (even if wrong) choice is left alone.
_COLLECTION_ARG_TOOLS = {"count_businesses_in_da", "polygon_filter"}


def _infer_collection_from_message(user_message: str) -> Optional[str]:
    msg = user_message.lower()
    if _hits(msg, LAYER_KEYWORDS.get("locauxVacants", [])) > 0:
        return "locaux_vacants"
    return None


# Tools that operate on the spatial engine need the shared geo_rag instance
# injected - these are exactly the tools.py wrappers whose first parameter is
# geo_rag. The rest (run_analysis, and the map/UI tools) take none.
_GEO_RAG_TOOL_NAMES = {
    "geocode", "spatial_search", "polygon_filter", "compare_locations",
    "get_demographics", "find_dissemination_area", "get_dissemination_area_by_code",
    "count_businesses_in_da", "get_property_tax_info", "find_zone_at_location",
    "parse_uploaded_document",
}


def _execute_tool_call(tool_name: str, tool_input: dict) -> dict:
    """Runs one Anthropic tool_use block by dispatching to tools.TOOL_FUNCTIONS.
    Never raises - a bad call (unknown tool, wrong args, spatial engine not
    ready) becomes an {"error": ...} dict that goes back to Claude as the
    tool_result, so it can see the failure and adjust instead of crashing
    the whole loop."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool '{tool_name}'"}
    try:
        if tool_name in _GEO_RAG_TOOL_NAMES:
            if not geo_rag:
                return {"error": "Spatial engine is still initializing"}
            return fn(geo_rag, **tool_input)
        return fn(**tool_input)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {e}"}


def run_agent_loop(
    user_message: str,
    max_iterations: int = MAX_AGENT_ITERATIONS,
    preselected_polygon: Optional[list] = None,
) -> dict:
    """
    Full reason -> act -> observe -> reason again loop for queries that need
    multiple, DEPENDENT rounds of tool calls to answer (see module comment
    above). Sends the running conversation (including each tool_result) back
    to Claude every iteration; stops as soon as Claude replies with plain text
    (no more tool_use blocks). If max_iterations is hit first, returns a
    graceful "too complex" message plus whatever partial tool results were
    gathered, rather than silently truncating or crashing.

    preselected_polygon: an arbitrary polygon the user drew/selected on the
    map (ChatTurn1Request.selected_polygon) - NOT necessarily a dissemination
    area or isochrone. When set, it takes priority over
    _last_resolved_polygon in the auto-fill backstop below, since it's an
    explicit user selection rather than something this loop incidentally
    looked up.

    Returns {"status": "complete" | "incomplete" | "error", "final_text": str,
    "iterations": int, "partial_results": list}.
    """
    messages = [{"role": "user", "content": user_message}]
    partial_results = []

    for iteration in range(1, max_iterations + 1):
        t_iter = time.perf_counter()
        try:
            response = claude.messages.create(
                model=AGENT_MODEL,
                max_tokens=1024,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        except Exception as e:
            print(f"[Agent Loop] Iteration {iteration} — Claude call failed: {e}", flush=True)
            return {
                "status": "error",
                "final_text": "Erreur de connexion à l'API Claude pendant le raisonnement de l'agent.",
                "iterations": iteration - 1,
                "partial_results": partial_results,
            }

        messages.append({"role": "assistant", "content": response.content})
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            final_text = "".join(b.text for b in response.content if b.type == "text").strip()
            print(f"[Agent Loop] Iteration {iteration}/{max_iterations} — Claude answered "
                  f"directly ({time.perf_counter() - t_iter:.2f}s), stopping.", flush=True)
            return {
                "status": "complete",
                "final_text": final_text,
                "iterations": iteration,
                "partial_results": partial_results,
            }

        print(f"[Agent Loop] Iteration {iteration}/{max_iterations} — Claude requested "
              f"{len(tool_use_blocks)} tool call(s):", flush=True)

        tool_result_blocks = []
        for block in tool_use_blocks:
            tool_input = dict(block.input)

            arg_name = _POLYGON_ARG_NAME.get(block.name)
            if arg_name and not tool_input.get(arg_name):
                fallback_polygon = preselected_polygon or _last_resolved_polygon(partial_results)
                if fallback_polygon:
                    tool_input[arg_name] = fallback_polygon
                    source = "the user's selected/drawn map polygon" if preselected_polygon else "the most recently resolved boundary"
                    print(f"    (auto-filled missing '{arg_name}' for {block.name} from {source})", flush=True)

            if block.name in _COLLECTION_ARG_TOOLS and "collection" not in tool_input:
                inferred_collection = _infer_collection_from_message(user_message)
                if inferred_collection:
                    tool_input["collection"] = inferred_collection
                    print(f"    (defaulted 'collection' to '{inferred_collection}' based on the "
                          f"user's message - {block.name} left it unset)", flush=True)

            print(f"    -> {block.name}({json.dumps(tool_input, ensure_ascii=False)})", flush=True)
            result = _execute_tool_call(block.name, tool_input)
            partial_results.append({"tool": block.name, "input": tool_input, "result": result})

            result_str = json.dumps(result, ensure_ascii=False, default=str)
            preview = result_str if len(result_str) <= 200 else result_str[:200] + "..."
            print(f"       result: {preview} ({time.perf_counter() - t_iter:.2f}s)", flush=True)

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_result_blocks})

    print(f"[Agent Loop] Hit max_iterations ({max_iterations}) without a final answer.", flush=True)
    return {
        "status": "incomplete",
        "final_text": (
            "Cette question était trop complexe pour être entièrement résolue en "
            f"{max_iterations} étapes. Voici les résultats partiels obtenus jusqu'ici."
        ),
        "iterations": max_iterations,
        "partial_results": partial_results,
    }


# Coarse substring block-list for obvious prompt-injection / system-probing
# attempts (e.g. "ignore previous instructions", "show me your system prompt")
# before the message ever reaches Claude. This is a blunt first line of
# defense, not a guarantee - substring matching over the lowercased message
# is trivially evaded (synonyms, other languages, typos) and can false-positive
# on legitimate queries that happen to contain one of these words.
RED_FLAG_PATTERNS = [
    "system prompt", "show instructions", "reveal prompt",
    "ignore previous", "reset instructions", "bypass", "override",
    "debug mode", "internal", "configuration", "backend",
    "database", "API key", "secret", "password", "credentials",
    "jailbreak", "default instructions"
]

MAX_MESSAGE_LENGTH = 5000


@app.post("/api/rag/chat-turn-1")
@limiter.limit("30/minute")
def chat_turn_1(request: Request, payload: ChatTurn1Request):
    t0 = time.perf_counter()
    user_message = payload.message
    print(f"Turn 1: Received user message: '{user_message}'", flush=True)

    user_query_lower = user_message.lower()
    if any(pattern in user_query_lower for pattern in RED_FLAG_PATTERNS):
        print(f"Turn 1: Blocked — message matched a red-flag pattern.", flush=True)
        return {
            "status": "error",
            "message": "Je ne peux pas répondre à cette question. "
                       "Posez une question sur la recherche de commerces, "
                       "l'analyse de zones, ou les données immobilières."
        }

    if len(user_message) > MAX_MESSAGE_LENGTH:
        print(f"Turn 1: Blocked — message length {len(user_message)} exceeds {MAX_MESSAGE_LENGTH}.", flush=True)
        return {
            "status": "error",
            "message": "La requête est trop longue (max 5000 caractères)."
        }

    # An uploaded file is a stronger, more specific signal than any keyword
    # match in the taxonomy below - route straight to the full tool-use loop
    # rather than hoping fast_extract_intent/the LLM classifies phrasing like
    # "update the map with the document I provided" correctly.
    if payload.uploaded_file:
        print(f"Turn 1: Uploaded file present ({payload.uploaded_file.file_type}) — "
              f"routing straight to the agent loop.", flush=True)
        agent_message = (
            f"{payload.message}\n\n"
            f"[Contexte système : un fichier a été téléversé et est disponible à "
            f"file_path=\"{payload.uploaded_file.file_path}\", "
            f"file_type=\"{payload.uploaded_file.file_type}\". "
            f"Utilisez parse_uploaded_document avec ces valeurs exactes.]"
        )
        agent_message = _augment_message_with_polygon(agent_message, payload.selected_polygon)
        agent_result = run_agent_loop(agent_message, preselected_polygon=payload.selected_polygon)
        return {
            "status": "success",
            "tool_name": "map_command",
            "args": {
                "action_type": "explore",
                "location_query": "Montreal",
                "layers_to_activate": [],
                "message": agent_result["final_text"],
                "agent_status": agent_result["status"],
                "agent_iterations": agent_result["iterations"],
            }
        }

    intent_data = fast_extract_intent(payload.message)

    # Set further down (LLM escalation path only, since fast_extract_intent
    # never looks at history) when the most recent turn was a named-boundary
    # agent_task (case (b) - a DA code, a drawn zone) rather than a
    # geocodable place - read again in the agent_task branch below, as a
    # code-level safety net (see the comment there for why the prompt hint
    # alone isn't enough). Initialized here, before the fast/LLM branch, so
    # it's always defined by the time that branch runs regardless of which
    # path classified this message.
    last_boundary_message = None

    if intent_data:
        notes = intent_data.get("resolution_notes") or []
        notes_str = f" | notes: {'; '.join(notes)}" if notes else ""
        print(f" Turn 1 resolved via regex in {time.perf_counter() - t0:.3f}s "
              f"→ action_type={intent_data.get('action_type')}, panel_name={intent_data.get('panel_name')}, "
              f"location='{intent_data.get('location_query')}'{notes_str}", flush=True)
    else:
        print("Regex extraction inconclusive or ambiguous — escalating to LLM for verification...", flush=True)

        # Recent turns, compact (resolved intent only, never raw report text)
        # so the model can resolve follow-ups like "show retail there instead"
        # that name no location of their own - without this, every follow-up
        # would need to repeat the full context from scratch.
        history_block = ""
        if payload.history:
            lines = []
            for h in payload.history[-5:]:
                parts = [f'message="{h.message}"']
                if h.action_type: parts.append(f"action_type={h.action_type}")
                if h.location_query: parts.append(f"location={h.location_query}")
                if h.panel_name: parts.append(f"panel={h.panel_name}")
                if h.layers_to_activate: parts.append(f"layers={h.layers_to_activate}")
                if h.naics_query: parts.append(f"naics={h.naics_query}")
                lines.append("- " + ", ".join(parts))

            # Whichever came LAST between "a place was named" and "a named
            # boundary was resolved instead" - never both, since blending
            # stale signals from different points in the conversation would
            # be more confusing than picking the single most recent one. A
            # boundary turn is only detectable by its SHAPE
            # (action_type="explore" with location_query null) - per this
            # same prompt's own location_query rule below, that's the ONLY
            # way a case (b) resolution shows up in history, since it never
            # sets a location_query. The raw message text is the sole
            # surviving trace of what boundary the user actually named (e.g.
            # "aire de diffusion 24460257") - there's no dedicated field for
            # it, so this reuses h.message rather than inventing one.
            last_location = None
            for h in reversed(payload.history):
                if h.location_query:
                    last_location = h.location_query
                    break
                if h.action_type == "explore":
                    last_boundary_message = h.message
                    break

            history_block = (
                "\n        === CONVERSATION HISTORY (most recent turn last) ===\n"
                + "\n".join(f"        {l}" for l in lines) + "\n"
                + "        IMPORTANT: The history above is real and not empty. If the CURRENT message below "
                "names no place of its own, you MUST set location_query to the most recent location in "
                f"this history{f' (which is \"{last_location}\")' if last_location else ''} instead of "
                "defaulting to Montreal. Only ignore the history if the current message clearly names its "
                "own different place, OR is a named-boundary agent_task request (case (b) below - a "
                "dissemination area code, a specific zone, a drawn polygon reference) - the boundary "
                "itself is the location there, so neither history nor Montreal apply; leave location_query "
                "null instead.\n"
            )
            if last_boundary_message:
                history_block += (
                    "        IMPORTANT — NAMED BOUNDARY CONTINUATION: the most recently resolved "
                    "context was a NAMED BOUNDARY, not a geocodable place, from this exact message: "
                    f"\"{last_boundary_message}\". If the CURRENT message below names no place or "
                    "boundary of its own (e.g. it only asks to filter, count, or refine the same "
                    "results - \"filtre par catégorie\", \"combien sont des restaurants\"), it is "
                    "almost certainly still about THAT SAME boundary: classify it as agent_task case "
                    "(b) again and set location_query to null (do NOT default to Montreal, do NOT "
                    "reuse a place name instead) so the boundary is re-resolved from context. Only "
                    "skip this if the current message clearly names its own different place or "
                    "boundary.\n"
                )
            history_block += "=== END CONVERSATION HISTORY ===\n"

        # Told explicitly rather than left for the model to infer from wording
        # alone: a terse message like "list the vacant locals" or "combien de
        # commerces ici" carries none of case (b)'s usual boundary language
        # ("this DA", "the zone I drew") on its own, but selected_polygon
        # being non-empty means a real boundary already exists - the model
        # just isn't looking at it directly (it becomes real coordinates only
        # later, via run_agent_loop's preselected_polygon auto-fill). Without
        # this hint, such a message would misclassify as "explore" and never
        # reach run_agent_loop/polygon_filter at all.
        polygon_hint = ""
        if payload.selected_polygon:
            polygon_hint = (
                "\n        === ACTIVE MAP SELECTION ===\n"
                "        The user currently has a custom polygon drawn/selected on the map - NOT a "
                "dissemination area, NOT an isochrone. If the message asks to list, count, or find "
                "businesses, vacant locals, or properties and names no separate place of its own, treat "
                "this exactly as agent_task case (b) (a named-boundary request) with location_query "
                "null, even if the message doesn't explicitly say \"this zone\"/\"the area I drew\" - "
                "the drawn selection itself is that boundary. Only ignore this selection if the message "
                "clearly names its own different place instead.\n=== END ACTIVE MAP SELECTION ===\n"
            )

        # UPDATED PROMPT: Added spatial_search instructions
        prompt = f"""
        Extract the intent from the user's message regarding a geospatial map application.
        Determine the following fields, IN ORDER — field 1 is a scratchpad you must fill in
        before deciding action_type, not after, since your action_type choice needs to follow
        from this reasoning rather than the other way around:
        {history_block}
        {polygon_hint}
        1. action_type_reasoning: One short sentence: does the message combine TWO SEPARATE
           conditions that must BOTH hold — one about people/demographics (income, population,
           age) and one about businesses/places (a category, or density/count of something) —
           joined by a contrast like "but"/"mais"/"and"/"et"? (e.g. "areas with high income BUT
           low restaurant density near X", "zones avec fort revenu ET peu de commerces"). If yes,
           say so explicitly — that combination is what "cross_query" below means, and it applies
           EVEN THOUGH the message also mentions demographics and businesses individually, which
           might otherwise look like plain "explore". If no, say why it's a different action_type.

        2. action_type — must match what you just concluded in field 1:
           - "cross_query": the two-condition case identified above.
           - "explore": ONLY a general look at the map with a layer turned on and AT MOST ONE simple criterion, and NO named boundary to filter by (e.g. just "show me the demographics of X", or just "show me restaurants in X") - NOT two conditions combined, and NOT scoped to a specific named area (see agent_task below for that case).
           - "analyze": if the user wants a deep analysis.
           - "spatial_search": if the user explicitly asks for the closest items or a specific count based on distance from a POINT (e.g., "closest empty locals").
           - "compare": if the user wants to compare 2 or more named locations against each other (e.g., "which is better for a bakery, X or Y", "compare X and Y").
           - "agent_task": any of these three shapes - do NOT use "explore" for any of them:
               (a) an open-ended recommendation/optimization question that needs several DEPENDENT steps to answer, not a single lookup and not a comparison of already-named locations (e.g. "which address is best for a bakery near X", "where should I open a gym", "find me the best spot for a yoga studio"). Use "compare" instead if 2+ SPECIFIC locations are already named in the message.
               (b) a request to list, count, or find businesses, vacant locals, or properties WITHIN A NAMED BOUNDARY rather than a simple radius around a point - a dissemination area code, a specific zone, a drawn polygon reference, or any other named enclosed area (e.g. "list the businesses in dissemination area 24540318", "how many businesses are in this DA", "count the properties inside the zone I drew", "list the vacant locals in DA 24540318", "how many empty spaces are in this zone"). The defining signal is a BOUNDARY the user already has in mind (a code, "this zone", "the area I drew") rather than "near X" - that boundary has to be resolved and matched against, which "explore" (just toggling a layer) cannot do. This applies equally whether the user means businesses or vacant locals - pick whichever the message actually names.
               (c) a zoning/permitted-use question about a SPECIFIC address or place - what's allowed to be built or operated there, whether a given use is permitted or conditional, or what the zone code is (e.g. "is a café permitted at 50 Rue du Bourgmestre, Bromont", "what can I build at X", "is this address zoned for a restaurant", "can I open a daycare at X", "quels usages sont permis à cette adresse", "puis-je ouvrir un café à X"). This needs a real zoning lookup (find_zone_at_location) that resolves the zone and checks allowed uses - "explore" would just pan the map and flip on the zonage layer without ever answering the yes/no question actually asked. Don't be misled by "commerces"/"zonage"-style layer keywords appearing in the message - a permitted-use question about one address is agent_task even though it also implies those layers.
           - If the message asks for BOTH a general view AND a deep analysis (e.g. "show me restaurants and analyze the competition"), prefer "analyze" — the analysis pipeline also activates the requested map layers as a side effect, so nothing the user asked for is lost.

        3. location_query: The address, city, or place mentioned (for cross_query, the area to search WITHIN, e.g. a city). If the current message names no place at all, reuse the most recent location from the conversation history above (if any) rather than defaulting blindly. Only fall back to "Montreal" if there's truly no location anywhere. Null if action_type is "compare" (use "locations" instead). ALSO null if action_type is "agent_task" specifically because of a named boundary/area code (case (b) above, e.g. a DA code, "this zone", "the polygon I drew") - the boundary itself is what anchors the request (it gets resolved to its own geometry by a dedicated tool), so do NOT geocode a place name, and do NOT reuse a stale location from conversation history, in this case. Only set location_query normally (including reusing history) for a case (a) or case (c) agent_task request, both of which do name or imply a real place.
        4. locations: If action_type is "compare", a list of the 2+ location names being compared (e.g. ["Bromont", "Saint-Hyacinthe"]). Otherwise null.
        5. panel_name: If action_type is "analyze":
           - If the user wants ONLY ONE specific analysis, pick ["competitive", "retention", "siteScore", "implementation"].
           - If the user wants a complete/full report, OR names two or more specific pillars in the same message (e.g. "look at retention and competition"), use "all" — don't arbitrarily pick just one of several named pillars.
           - Otherwise null.
        6. layers_to_activate: ["commerces", "locauxVacants", "statcanDA", "zonage"].
        7. minutes: Isochrone time. Default 10.
        8. mode: Travel mode ("driving", "walking", "cycling"). Default "driving".
        9. naics_query: An explicit NAICS code if one is mentioned (e.g. "NAICS 722" → "722"). Otherwise null.
        10. needs_zone: true if the user implies a zone/isochrone should be drawn (mentions minutes, "within", "zone", "radius", "catchment"), even for exploration requests. Otherwise false.
        11. max_budget: A monthly rent/budget cap in dollars if one is mentioned (e.g. "under $2000/month", "moins de 2000$/mois" → 2000). Otherwise null.
        12. demographic_field: Only if action_type is "cross_query" — exactly one of "Population_2021", "revenu_median", "age_moyen", "menages_total" (pick whichever the message is actually about — income → revenu_median, population/size → Population_2021, age → age_moyen, households → menages_total). Otherwise null.
        13. demographic_direction: Only if action_type is "cross_query" — "high" or "low". Otherwise null.
        14. business_category: Only if action_type is "cross_query" — the free-text business/place type mentioned (e.g. "restaurant"). Otherwise null.
        15. business_direction: Only if action_type is "cross_query" — "high" or "low" (density/count). Otherwise null.
        16. resolution_notes: A list of short strings explaining any judgment calls you made (e.g. ["message named two pillars, treated as full report"]). Empty list if none.

        User message: "{payload.message}"
        Return ONLY a valid JSON object matching these keys.
        """
        try:
            t_llm = time.perf_counter()
            message = claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown fences if the model wraps the JSON
            raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            intent_data = json.loads(raw)

            # Claude Haiku is reliable at EXTRACTING demographic_field/
            # business_category (it always gets those right) but has proven
            # unreliable at also setting action_type="cross_query" itself -
            # observed misclassifying as "explore" even when its own
            # action_type_reasoning field explicitly concluded "cross_query".
            # Since _looks_like_cross_query() already deterministically
            # decided this message needed cross-query handling (that's the
            # only way this LLM branch fires for such a message), correct
            # the field here rather than trust a value the model contradicts
            # in its own reasoning.
            if (_looks_like_cross_query(payload.message.lower())
                    and intent_data.get("demographic_field")
                    and intent_data.get("business_category")
                    and intent_data.get("action_type") != "cross_query"):
                print(f"[Turn 1] Correcting action_type '{intent_data.get('action_type')}' → "
                      f"'cross_query' (model extracted the right criteria but mislabeled the action)", flush=True)
                intent_data["action_type"] = "cross_query"

            llm_notes = intent_data.get("resolution_notes") or []
            llm_notes_str = f" | notes: {'; '.join(llm_notes)}" if llm_notes else ""
            print(f"Turn 1 LLM took {time.perf_counter() - t_llm:.2f}s "
                  f"→ action_type={intent_data.get('action_type')}, panel_name={intent_data.get('panel_name')}, "
                  f"location='{intent_data.get('location_query')}'{llm_notes_str}", flush=True)
        except Exception as e:
            print(f"Turn 1 LLM Error: {e}", flush=True)
            intent_data = {
                "action_type": "analyze", "panel_name": "competitive",
                "location_query": "Montreal", "layers_to_activate": [],
                "minutes": 10, "mode": "driving", "naics_query": None,
                "resolution_notes": ["LLM call failed — used hardcoded fallback"],
            }

    # Guarantee these fields are always present (regex path already sets them;
    # the LLM path and the exception fallback above may not) — the frontend
    # uses raw_message to match free-text sector names against its live
    # sector list, and resolution_notes surfaces judgment calls in logs.
    intent_data.setdefault("raw_message", payload.message)
    intent_data.setdefault("naics_query", None)
    intent_data.setdefault("naics_error", None)
    intent_data.setdefault("max_budget", None)
    intent_data.setdefault("locations", None)
    intent_data.setdefault("demographic_field", None)
    intent_data.setdefault("demographic_direction", None)
    intent_data.setdefault("business_category", None)
    intent_data.setdefault("business_direction", None)
    # Populated below, only for action_type "explore" - see that block for
    # why this exists (business_category is scoped to cross_query only,
    # naics_query to explicit numeric codes, so a plain "show me the cinema
    # of Bromont" would otherwise carry no category signal at all).
    intent_data.setdefault("category_query", None)
    intent_data.setdefault("needs_zone", bool(MINUTES_PATTERN.search(payload.message)) or any(k in payload.message.lower() for k in ZONE_KEYWORDS))
    intent_data.setdefault("resolution_notes", [])

    # =====================================================================
    #  THE INTERCEPTION BLOCK FOR SPATIAL RAG 
    # =====================================================================
    if intent_data.get("action_type") == "spatial_search":
        print(" Spatial Search intercepted! Executing targeted RAG retrieval...", flush=True)

        target_landmark = intent_data.get("location_query", "Montreal")
        requested_layers = intent_data.get("layers_to_activate") or []
        target_collections = {
            SPATIAL_SEARCH_COLLECTIONS[layer]
            for layer in requested_layers if layer in SPATIAL_SEARCH_COLLECTIONS
        }
        if not target_collections:
            # Nothing specific named (e.g. "closest to X") — matches the old
            # behaviour of this block, which only ever searched vacant locals.
            target_collections = {"locaux_vacants"}

        # A numeric NAICS code and a free-text category name are alternate
        # ways of saying "what kind of place" - only look for one when the
        # other wasn't already given, so e.g. "closest NAICS 722" doesn't
        # also produce a spurious category phrase out of leftover words.
        naics_query = intent_data.get("naics_query")
        category_phrase = None if naics_query else _extract_category_phrase(
            intent_data.get("raw_message", ""), target_landmark
        )
        max_budget = intent_data.get("max_budget")
        if category_phrase:
            print(f"[Spatial Search] Free-text category detected: '{category_phrase}'", flush=True)
        if max_budget is not None:
            print(f"[Spatial Search] Monthly budget cap detected: ${max_budget:.0f}/mois", flush=True)

        try:
            if not geo_rag:
                raise RuntimeError("Spatial engine is still initializing")

            geocode_result = resolve_location(target_landmark, payload.resolved_location)
            if geocode_result["ambiguous"]:
                return {
                    "status": "success",
                    "tool_name": "map_command",
                    "args": {
                        "action_type": "needs_disambiguation",
                        "ambiguous_term": target_landmark,
                        "raw_message": intent_data.get("raw_message", ""),
                        "candidates": geocode_result["candidates"],
                    }
                }
            if geocode_result["lat"] is None:
                raise ValueError(f"Geocoding returned no results for '{target_landmark}'")

            target_lat = geocode_result["lat"]
            target_lng = geocode_result["lng"]
            print(f"Geocoded '{target_landmark}' → {target_lat}, {target_lng}", flush=True)

            # Hard spatial pre-filter ($geoNear per collection) — same radius
            # the old raw query used, now via geo_rag so the candidates carry
            # the metadata (lon/lat/type/NAICS/etc.) hybrid_semantic_search needs.
            candidates = []
            for collection in target_collections:
                candidates.extend(
                    geo_rag.hard_spatial_filter(
                        collection, target_lng, target_lat, DEFAULT_SPATIAL_SEARCH_RADIUS_M
                    )
                )
            print(f"[Spatial Search] {len(candidates)} candidate(s) within "
                  f"{DEFAULT_SPATIAL_SEARCH_RADIUS_M}m of '{target_landmark}'", flush=True)

            if not candidates:
                return {
                    "status": "success",
                    "tool_name": "map_command",
                    "args": {
                        "action_type": "spatial_results",
                        "location_query": target_landmark,
                        "spatial_data": [],
                        "layers_to_activate": requested_layers or ["locauxVacants"],
                        "message": f"Aucun résultat trouvé près de {target_landmark}.",
                    }
                }

            # Fuse dense (FAISS) + sparse (BM25) semantic relevance with the
            # spatial distance already on each candidate, then boost NAICS
            # matches if the user named a code. Falls back to distance-only
            # ordering if the embedding model isn't reachable, rather than
            # dropping the spatial_search intent entirely.
            try:
                # top_k=len(candidates): hybrid_semantic_search's own default
                # (top 15) would silently drop candidates before pareto_rank's
                # NAICS/category/budget boosts ever got a chance to run on them
                # - a real match ranking outside the top 15 on raw semantic
                # similarity to the full (noisy) message would simply vanish.
                ranked = geo_rag.hybrid_semantic_search(
                    query=intent_data.get("raw_message", target_landmark),
                    spatial_docs=candidates,
                    top_k=len(candidates),
                )
                ranked = geo_rag.pareto_rank(
                    ranked, DEFAULT_SPATIAL_SEARCH_RADIUS_M,
                    target_naics=naics_query,
                    target_category=category_phrase,
                    max_budget=max_budget,
                )
            except Exception as e:
                print(f"[Spatial Search] Semantic ranking unavailable ({e}) — "
                      f"falling back to distance-only order", flush=True)
                ranked = sorted(candidates, key=lambda d: d.metadata.get("distance_m", DEFAULT_SPATIAL_SEARCH_RADIUS_M))

            results_to_send = []
            for doc in ranked[:10]:
                md = doc.metadata
                results_to_send.append({
                    "id": md.get("id"),
                    "distance": round(md.get("distance_m", 0)),
                    "coordinates": [md.get("lon"), md.get("lat")],
                    "label": doc.page_content,
                    "source": md.get("source_collection"),
                    "match_status": md.get("match_status", "Standard Rank"),
                    "monthly_price": md.get("monthly_price"),
                    "price_match": md.get("price_match"),
                    # Additive: same values already baked into "label" above,
                    # now also broken out as their own fields. nom/secteur/
                    # code_naics are only set in metadata for quebec_businesses
                    # (see _doc_from_mongo) - md.get(...) naturally returns None
                    # for a vacant local, rather than a placeholder string.
                    "nom": md.get("nom"),
                    "adresse": md.get("adresse"),
                    "secteur": md.get("secteur"),
                    "code_naics": md.get("naics_code") or None,
                })

            # Return the ranked points to React
            return {
                "status": "success",
                "tool_name": "map_command",
                "args": {
                    "action_type": "spatial_results",
                    "location_query": target_landmark,
                    "spatial_data": results_to_send,
                    "layers_to_activate": requested_layers or ["locauxVacants"]
                }
            }

        except Exception as e:
            print(f"Spatial query failed: {e}")
            # Fallback to standard explore if geocoding or the DB failed
            intent_data["action_type"] = "explore"

    # =====================================================================
    # THE INTERCEPTION BLOCK FOR OPEN-ENDED AGENTIC TASKS
    # =====================================================================
    elif intent_data.get("action_type") == "agent_task":
        print(" Agent task intercepted! Running full tool-use loop...", flush=True)
        agent_message = _augment_message_with_polygon(payload.message, payload.selected_polygon)

        # Named-boundary continuation (see history_block/last_boundary_message
        # above): the classifier prompt asks the model to re-classify a
        # location-less follow-up as case (b) again when the last resolved
        # context was a boundary, but that instruction only affects THIS
        # classification call - it doesn't put the boundary text in front of
        # run_agent_loop, which is a SEPARATE Claude call that only ever sees
        # agent_message. Without this, the classifier could correctly decide
        # "still about that DA" while the agent itself still has no idea
        # which one, since "filtre les commerces par categorie" alone never
        # named any boundary. Re-inject the original phrasing (e.g. "aire de
        # diffusion 24460257") so the SAME get_dissemination_area_by_code
        # tool call the first turn made fires again, instead of the agent
        # guessing or defaulting to Montreal.
        #
        # Guarded so this never fires when the current message plausibly
        # names its own boundary: a real DA code / zone reference is
        # numeric, so any digit in the message is treated as "this message
        # has its own boundary, don't override it" - crude, but avoids the
        # much worse failure mode of clobbering a fresh, self-contained DA
        # reference with a stale one from several turns back. A live
        # selected_polygon also always wins (it already provides its own
        # boundary via preselected_polygon autofill below, no history needed).
        if (
            not payload.selected_polygon
            and not intent_data.get("location_query")
            and last_boundary_message
            and not any(ch.isdigit() for ch in payload.message)
        ):
            print(f"    (re-anchoring location-less follow-up to last boundary: \"{last_boundary_message}\")", flush=True)
            agent_message = f"{last_boundary_message}. {agent_message}"

        agent_result = run_agent_loop(agent_message, preselected_polygon=payload.selected_polygon)

        location_query = intent_data.get("location_query")
        centroid = None
        resolved_polygon_ring = None
        result_items = None

        # Scan every tool call this loop actually made, most recent first
        # ("last call wins" if the agent looked up more than one boundary or
        # list). These are two independent things to recover - a boundary/
        # centroid, and a list of matched businesses/vacant locals - so no
        # early break once one is found; keep scanning for the other.
        for call in reversed(agent_result.get("partial_results") or []):
            tool_name = call.get("tool")
            call_result = call.get("result") or {}

            if not location_query and resolved_polygon_ring is None and tool_name == "get_dissemination_area_by_code":
                # Named-boundary agent_task (case (b) in the classifier prompt) -
                # location_query is deliberately null since a boundary code isn't
                # a geocodable place name. Recover a map center AND the actual
                # boundary from whatever DA the agent loop resolved, instead of
                # defaulting to Montreal with nothing drawn.
                ring = call_result.get("polygon_ring")
                if ring:
                    resolved_polygon_ring = ring
                    # get_dissemination_area_by_code returns a flat ring, not a
                    # full geometry dict - wrap it the same way
                    # _as_polygon_geometry (tools.py) does, since
                    # _polygon_centroid expects geometry["coordinates"][0].
                    centroid = _polygon_centroid({"coordinates": [ring]})

            if result_items is None and tool_name == "polygon_filter":
                # For "list X in this DA/zone" queries, the agent uses
                # polygon_filter - confirmed empirically, not assumed:
                # count_businesses_in_da only ever returns a count, never
                # individual items. polygon_filter's results already carry
                # nom/adresse/secteur/code_naics (tools.py mirrors this same
                # spatial_search block's results_to_send shape), so no field
                # remapping is needed here - both read off the same
                # _doc_from_mongo metadata.
                items = call_result.get("results")
                if items:
                    result_items = items

        # A selected_polygon request never calls get_dissemination_area_by_code
        # (the agent already has the boundary via preselected_polygon autofill),
        # so centroid/resolved_polygon_ring stay None above even though a real,
        # already-on-screen boundary exists - falling back to "Montreal" here
        # would recenter the map away from wherever the user actually drew
        # their shape. Treat payload.selected_polygon as an equally valid
        # "a boundary exists" signal, and echo it back as polygon_ring too
        # (harmless - the frontend already has it - but keeps this response
        # self-consistent for anything reading polygon_ring unconditionally).
        has_boundary = centroid is not None or bool(payload.selected_polygon)
        args = {
            "action_type": "explore",
            "location_query": location_query or (None if has_boundary else "Montreal"),
            "centroid": centroid,
            "polygon_ring": resolved_polygon_ring or payload.selected_polygon,
            "layers_to_activate": [],
            "message": agent_result["final_text"],
            "agent_status": agent_result["status"],
            "agent_iterations": agent_result["iterations"],
        }
        # Omit entirely (not an empty list) when the agent never produced a
        # list - e.g. a get_dissemination_area_by_code + count_businesses_in_da
        # run (a pure count, no items) has nothing to attach here.
        if result_items is not None:
            args["result_items"] = result_items

        return {
            "status": "success",
            "tool_name": "map_command",
            "args": args,
        }

    # =====================================================================
    # THE INTERCEPTION BLOCK FOR LOCATION COMPARISON
    # =====================================================================
    elif intent_data.get("action_type") == "compare":
        print(" Compare intercepted! Running per-location retrieval...", flush=True)
        import requests

        locations = intent_data.get("locations") or []
        requested_layers = intent_data.get("layers_to_activate") or []
        target_collections = {
            SPATIAL_SEARCH_COLLECTIONS[layer]
            for layer in requested_layers if layer in SPATIAL_SEARCH_COLLECTIONS
        }
        if not target_collections:
            target_collections = {"locaux_vacants"}
        kind_label = (
            "commerces et locaux vacants" if len(target_collections) > 1
            else ("commerces" if "quebec_businesses" in target_collections else "locaux vacants")
        )

        naics_query = intent_data.get("naics_query")
        category_phrase = None if naics_query else _extract_category_phrase(
            intent_data.get("raw_message", ""), locations
        )
        max_budget = intent_data.get("max_budget")
        if category_phrase:
            print(f"[Compare] Free-text category detected: '{category_phrase}'", flush=True)

        location_results = []

        for loc_name in locations:
            entry = {"name": loc_name, "geocoded": False}
            try:
                if not geo_rag:
                    raise RuntimeError("Spatial engine is still initializing")

                geocode_result = resolve_location(loc_name, payload.resolved_location)
                if geocode_result["ambiguous"]:
                    # Pause the whole comparison on the first ambiguous name -
                    # resolving it out of order would compare against a place
                    # the user didn't actually mean. The frontend resubmits
                    # the original message with this name disambiguated once
                    # the user picks, and the loop starts over.
                    return {
                        "status": "success",
                        "tool_name": "map_command",
                        "args": {
                            "action_type": "needs_disambiguation",
                            "ambiguous_term": loc_name,
                            "raw_message": intent_data.get("raw_message", ""),
                            "candidates": geocode_result["candidates"],
                        }
                    }
                if geocode_result["lat"] is None:
                    raise ValueError(f"Geocoding returned no results for '{loc_name}'")

                lat, lng = geocode_result["lat"], geocode_result["lng"]
                entry["geocoded"] = True
                entry["coordinates"] = [lng, lat]
                print(f"[Compare] Geocoded '{loc_name}' → {lat}, {lng}", flush=True)

                # Only pulled when the query itself implied it (statcanDA
                # keyword hit - population/income/market/pouvoir d'achat/etc)
                # rather than on every comparison regardless of relevance.
                entry["demographics"] = None
                if "statcanDA" in requested_layers:
                    try:
                        demo_docs = geo_rag.get_smart_demographics(
                            loc_name, lat, lng, DEFAULT_SPATIAL_SEARCH_RADIUS_M, DEMOGRAPHIC_FIELDS
                        )
                        if demo_docs:
                            entry["demographics"] = demo_docs[0].metadata.get("metrics")
                    except Exception as e:
                        print(f"[Compare] Demographics lookup failed for '{loc_name}': {e}", flush=True)

                candidates = []
                for collection in target_collections:
                    candidates.extend(
                        geo_rag.hard_spatial_filter(collection, lng, lat, DEFAULT_SPATIAL_SEARCH_RADIUS_M)
                    )
                entry["candidate_count"] = len(candidates)

                if candidates:
                    try:
                        ranked = geo_rag.hybrid_semantic_search(
                            query=intent_data.get("raw_message", loc_name),
                            spatial_docs=candidates,
                            top_k=len(candidates),
                        )
                        ranked = geo_rag.pareto_rank(
                            ranked, DEFAULT_SPATIAL_SEARCH_RADIUS_M,
                            target_naics=naics_query,
                            target_category=category_phrase,
                            max_budget=max_budget,
                        )
                    except Exception as e:
                        print(f"[Compare] Semantic ranking unavailable for '{loc_name}' ({e}) — "
                              f"falling back to distance-only order", flush=True)
                        ranked = sorted(candidates, key=lambda d: d.metadata.get("distance_m", DEFAULT_SPATIAL_SEARCH_RADIUS_M))

                    matched_statuses = ("Exact NAICS", "Broad NAICS", "Category Match", "Type Local Boosted")
                    matched = [d for d in ranked if d.metadata.get("match_status") in matched_statuses]
                    prices = [d.metadata.get("monthly_price") for d in ranked if d.metadata.get("monthly_price") is not None]

                    entry["matched_count"] = len(matched) if (naics_query or category_phrase) else None
                    entry["avg_monthly_price"] = round(sum(prices) / len(prices)) if prices else None
                    entry["top_results"] = [
                        {
                            "id": d.metadata.get("id"),
                            "distance": round(d.metadata.get("distance_m", 0)),
                            "coordinates": [d.metadata.get("lon"), d.metadata.get("lat")],
                            "label": d.page_content,
                            "match_status": d.metadata.get("match_status", "Standard Rank"),
                            "monthly_price": d.metadata.get("monthly_price"),
                        }
                        for d in ranked[:3]
                    ]
                else:
                    entry["matched_count"] = 0
                    entry["avg_monthly_price"] = None
                    entry["top_results"] = []

            except Exception as e:
                print(f"[Compare] Failed for '{loc_name}': {e}", flush=True)
                entry["error"] = str(e)
                entry.setdefault("candidate_count", 0)
                entry.setdefault("matched_count", None)
                entry.setdefault("avg_monthly_price", None)
                entry.setdefault("top_results", [])

            location_results.append(entry)

        # Grounded verdict — short, and built ONLY from the numbers just
        # computed above, the same "never invent data" rule Turn 2 follows.
        verdict = None
        succeeded = [r for r in location_results if r.get("geocoded")]
        if len(succeeded) >= 2:
            focus = category_phrase or (f"NAICS {naics_query}" if naics_query else kind_label)
            stats_lines = []
            for r in succeeded:
                price_str = f", prix moyen {r['avg_monthly_price']}$/mois" if r.get("avg_monthly_price") else ""
                matched_str = f", {r['matched_count']} correspondant précisément" if r.get("matched_count") is not None else ""
                nearest = r["top_results"][0]["distance"] if r["top_results"] else None
                nearest_str = f", le plus proche à {nearest}m" if nearest is not None else ""
                demo = r.get("demographics")
                demo_str = ""
                if demo:
                    demo_parts = []
                    if demo.get("Population_2021") is not None:
                        demo_parts.append(f"population {demo['Population_2021']}")
                    if demo.get("revenu_median") is not None:
                        demo_parts.append(f"revenu médian {demo['revenu_median']}$")
                    if demo.get("menages_total") is not None:
                        demo_parts.append(f"{demo['menages_total']} ménages")
                    if demo.get("age_moyen") is not None:
                        demo_parts.append(f"âge moyen {demo['age_moyen']} ans")
                    if demo_parts:
                        demo_str = " | " + ", ".join(demo_parts)
                stats_lines.append(f"- {r['name']} : {r['candidate_count']} {kind_label} trouvés{matched_str}{nearest_str}{price_str}{demo_str}")

            prompt = f"""Tu es un analyste en intelligence territoriale. Compare les emplacements suivants pour cette recherche : "{focus}".

{chr(10).join(stats_lines)}

Rédige un verdict court (2-3 phrases) recommandant lequel semble le plus prometteur et pourquoi, en te basant uniquement sur ces chiffres. N'invente aucune donnée non fournie. Si les chiffres sont trop proches pour trancher, dis-le clairement."""
            try:
                t_llm = time.perf_counter()
                verdict_msg = claude.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                verdict = verdict_msg.content[0].text.strip()
                print(f"[Compare] Verdict generated in {time.perf_counter() - t_llm:.2f}s", flush=True)
            except Exception as e:
                print(f"[Compare] Verdict generation failed: {e}", flush=True)

        return {
            "status": "success",
            "tool_name": "map_command",
            "args": {
                "action_type": "compare_results",
                "locations": location_results,
                "verdict": verdict,
                "layers_to_activate": requested_layers or ["locauxVacants"],
            }
        }

    # =====================================================================
    # THE INTERCEPTION BLOCK FOR CROSS-COLLECTION QUERIES
    # "areas with high income but low restaurant density" - finds and ranks
    # dissemination areas, joining donnees (demographics) and
    # quebec_businesses (business density) by geography, since the two
    # collections share no key.
    # =====================================================================
    elif intent_data.get("action_type") == "cross_query":
        print(" Cross-query intercepted! Searching areas by demographic + business criteria...", flush=True)

        location = intent_data.get("location_query") or "Montreal"
        demographic_field = intent_data.get("demographic_field")
        business_category = intent_data.get("business_category")
        demographic_direction = intent_data.get("demographic_direction") or "high"
        business_direction = intent_data.get("business_direction") or "low"

        if not demographic_field or not business_category:
            return {
                "status": "success",
                "tool_name": "map_command",
                "args": {
                    "action_type": "explore",
                    "location_query": location,
                    "layers_to_activate": [],
                    "message": (
                        "Je n'ai pas pu identifier clairement les deux critères à combiner "
                        "(un critère démographique et un critère commercial). Essayez par exemple : "
                        "« zones avec un revenu élevé mais peu de restaurants près de X »."
                    ),
                }
            }

        CROSS_QUERY_SEARCH_RADIUS_M = 8000
        CROSS_QUERY_DA_CANDIDATE_LIMIT = 60
        CROSS_QUERY_DA_SHORTLIST = 15
        CROSS_QUERY_RESULT_COUNT = 6

        try:
            if not geo_rag:
                raise RuntimeError("Spatial engine is still initializing")

            geocode_result = resolve_location(location, payload.resolved_location)
            if geocode_result["ambiguous"]:
                return {
                    "status": "success",
                    "tool_name": "map_command",
                    "args": {
                        "action_type": "needs_disambiguation",
                        "ambiguous_term": location,
                        "raw_message": intent_data.get("raw_message", ""),
                        "candidates": geocode_result["candidates"],
                    }
                }
            if geocode_result["lat"] is None:
                raise ValueError(f"Geocoding returned no results for '{location}'")

            lat, lng = geocode_result["lat"], geocode_result["lng"]
            print(f"[Cross-Query] Geocoded '{location}' → {lat}, {lng}", flush=True)

            # 1. Candidate DAs near the search point — proximity-only fetch,
            # capped, so the expensive per-DA business count below only ever
            # runs on a bounded shortlist rather than every DA in the radius.
            das = geo_rag.find_candidate_das(lng, lat, CROSS_QUERY_SEARCH_RADIUS_M, CROSS_QUERY_DA_CANDIDATE_LIMIT)
            print(f"[Cross-Query] {len(das)} dissemination area(s) within {CROSS_QUERY_SEARCH_RADIUS_M}m", flush=True)

            if not das:
                return {
                    "status": "success",
                    "tool_name": "map_command",
                    "args": {
                        "action_type": "cross_query_results",
                        "location_query": location,
                        "demographic_field": demographic_field,
                        "business_category": business_category,
                        "results": [],
                        "verdict": None,
                        "message": f"Aucune zone trouvée près de {location}.",
                    }
                }

            # 2. Rank by the demographic criterion first — cheap, already
            # fetched, no extra queries needed for this step.
            # StatCan suppresses small-population cells with a placeholder
            # string ("x") for privacy rather than omitting the field, so a
            # plain "is not None" check isn't enough - some real documents
            # have a string sitting where every other one has a number.
            das_with_value = [
                (da, da[demographic_field]) for da in das
                if isinstance(da.get(demographic_field), (int, float))
            ]
            das_with_value.sort(key=lambda pair: pair[1], reverse=(demographic_direction == "high"))
            shortlist = das_with_value[:CROSS_QUERY_DA_SHORTLIST]

            # 3. Only the shortlist pays for a business-count query each.
            results = []
            for da, demo_val in shortlist:
                geometry = da.get("geometry")
                if not geometry:
                    continue
                biz_count = geo_rag.count_businesses_in_da(geometry, business_category)
                centroid = _polygon_centroid(geometry)
                if centroid is None:
                    continue
                results.append({
                    "dauid": da.get("Geographie"),
                    "coordinates": centroid,
                    "demographic_value": demo_val,
                    "business_count": biz_count,
                })

            # 4. Re-rank that shortlist by the business criterion.
            results.sort(key=lambda r: r["business_count"], reverse=(business_direction == "high"))
            final_results = results[:CROSS_QUERY_RESULT_COUNT]

            # Grounded verdict — short, built ONLY from the numbers just
            # computed above, the same rule Turn 2 and compare follow.
            verdict = None
            if final_results:
                demo_label = "élevé(e)" if demographic_direction == "high" else "faible"
                biz_label = "peu de" if business_direction == "low" else "beaucoup de"
                stats_lines = [
                    f"- Zone {r['dauid']} : {demographic_field} = {r['demographic_value']}, "
                    f"{business_category} trouvé(s) = {r['business_count']}"
                    for r in final_results
                ]
                prompt = f"""Tu es un analyste en intelligence territoriale. L'utilisateur cherche, près de {location}, des zones avec un(e) {demographic_field} {demo_label} et {biz_label} {business_category}.

Voici les {len(final_results)} zones les plus prometteuses trouvées, déjà classées :
{chr(10).join(stats_lines)}

Rédige un verdict court (2-3 phrases) sur ces zones, en te basant uniquement sur ces chiffres. N'invente aucune donnée non fournie. Si les résultats ne correspondent pas bien à la demande, dis-le clairement."""
                try:
                    t_llm = time.perf_counter()
                    verdict_msg = claude.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    verdict = verdict_msg.content[0].text.strip()
                    print(f"[Cross-Query] Verdict generated in {time.perf_counter() - t_llm:.2f}s", flush=True)
                except Exception as e:
                    print(f"[Cross-Query] Verdict generation failed: {e}", flush=True)

            return {
                "status": "success",
                "tool_name": "map_command",
                "args": {
                    "action_type": "cross_query_results",
                    "location_query": location,
                    "demographic_field": demographic_field,
                    "business_category": business_category,
                    "results": final_results,
                    "verdict": verdict,
                }
            }

        except Exception as e:
            print(f"Cross-query failed: {e}")
            intent_data["action_type"] = "explore"

    # =====================================================================
    # THE INTERCEPTION BLOCK FOR GENERAL VACANCY QUERIES
    # =====================================================================
    elif "locauxVacants" in intent_data.get("layers_to_activate", []) and intent_data.get("panel_name") == "competitive":
        
        # Check if the user EXPLICITLY asked for a competitive analysis
        explicit_competitive = any(k in payload.message.lower() for k in PANEL_KEYWORDS.get("competitive", []))
        
        if not explicit_competitive:
            print("City-Wide Vacancy Request intercepted! Bypassing the isochrone...", flush=True)
            target_city = intent_data.get("location_query", "Montreal")

            try:
                # Reuse the long-lived connection pool from geo_rag instead of
                # opening (and never closing) a brand-new MongoClient per request.
                db = geo_rag.db

                # Search the DB for vacancies matching the requested city name
                query = {
                    "$or": [
                        {"address": {"$regex": target_city, "$options": "i"}},
                        {"city": {"$regex": target_city, "$options": "i"}},
                        {"ville": {"$regex": target_city, "$options": "i"}}
                    ]
                }
                city_vacancies = list(db["locaux_vacants"].find(query).limit(10))
                
                if city_vacancies:
                    vacancy_data = "\n".join([f"- {v.get('address', 'Adresse inconnue')}" for v in city_vacancies])
                    
                    # Run a hyper-fast LLM generation right here in Turn 1
                    prompt = f"""
                    L'utilisateur veut voir les locaux vacants à {target_city}.
                    Voici quelques locaux disponibles tirés de la base de données :
                    {vacancy_data}
                    
                    Rédigez un court message professionnel (2-3 phrases) pour présenter ces options. 
                    Ne faites aucune analyse mathématique ou concurrentielle.
                    """
                    t_llm = time.perf_counter()
                    message = claude.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=150,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    llm_message = message.content[0].text.strip()
                    print(f" Vacancy narrative took {time.perf_counter() - t_llm:.2f}s", flush=True)
                else:
                    llm_message = f"J'ai bien activé la couche, mais je n'ai trouvé aucun local vacant contenant '{target_city}' dans la base de données."

                # Send back as 'explore' so React pans the map but skips the heavy Turn 2 pipeline
                return {
                    "status": "success",
                    "tool_name": "map_command",
                    "args": {
                        "action_type": "explore",
                        "location_query": target_city,
                        "layers_to_activate": ["locauxVacants"],
                        "message": llm_message
                    }
                }
                
            except Exception as e:
                print(f" City vacancy query failed: {e}")
                # Fallback to standard explore if the DB fails
                intent_data["action_type"] = "explore"

    # =====================================================================

    # "Show me the cinema of Bromont"-style requests classify as "explore"
    # (see the classifier prompt above: business_category is deliberately
    # scoped to cross_query only, naics_query only to explicit numeric
    # codes) - "explore" just pans the map and toggles a layer, so without
    # this, a plain free-text category name vanishes entirely and the
    # frontend has nothing telling it WHICH businesses to filter to, only
    # that a layer should turn on. Reuses the same extraction
    # spatial_search/analyze already apply for the identical problem, just
    # for this action_type too. Only when "commerces" is actually one of
    # the activated layers - a category phrase only means something against
    # the businesses layer (e.g. an "explore" for demographics/zoning would
    # otherwise pick up leftover words like "demographics" itself as a bogus
    # category_query, which nothing downstream should act on). Also skipped
    # when naics_query is already set, same reasoning as those call sites:
    # an explicit numeric code and a free-text name are alternate answers to
    # the same question, not two independent filters to combine.
    if (intent_data.get("action_type") == "explore"
            and not intent_data.get("naics_query")
            and "commerces" in (intent_data.get("layers_to_activate") or [])):
        category_query = _extract_category_phrase(
            intent_data.get("raw_message", ""), intent_data.get("location_query")
        )
        if category_query:
            intent_data["category_query"] = category_query
            print(f"[Explore] Free-text category detected: '{category_query}'", flush=True)

    # If not a spatial search (or if spatial failed), return the normal payload to React
    return {
        "status": "success",
        "tool_name": "map_command",
        "args": intent_data
    }


def generate_pillar_report(
    advanced_metrics: dict,
    site_score: dict,
    retention: dict,
    implementation: dict,
    population,
    households,
    median_income,
    avg_age,
    businesses: list,
    analysis_scope: str = "full",
    focus_panel: Optional[str] = None,
) -> str:
    """
    Builds the 4-pillar (or single-pillar) territorial intelligence narrative
    from metrics the frontend has ALREADY computed, and asks Claude to write
    it up. This is the exact body that used to live inline in chat_turn_2 -
    extracted so it can be called directly (e.g. by tools.run_analysis)
    without going through the FastAPI request/response cycle. It does not
    compute ISM/PAD/SPC/etc. itself - those come in as arguments.
    """
    # ── Pre-flight validation ────────────────────────────────────────
    # Catch an incomplete payload BEFORE spending LLM call on it —
    # this is exactly the bug that produced a full report describing an
    # implementation score that was never actually calculated.
    PANEL_DATA = {
        "competitive": advanced_metrics,
        "retention": retention,
        "siteScore": site_score,
        "implementation": implementation,
    }

    if analysis_scope == "single" and focus_panel in PANEL_DATA:
        focus_data = PANEL_DATA[focus_panel]
        empty_frac = _fraction_empty(focus_data)
        headline_empty = _is_headline_empty(focus_panel, focus_data)
        if headline_empty or empty_frac >= 0.5:
            label = PANEL_LABELS.get(focus_panel, focus_panel)
            print(f"Pre-flight check failed: {label} is {empty_frac:.0%} empty — "
                  f"skipping Claude call.", flush=True)
            return (
                f"Les données du panneau **{label}** ne semblent pas encore calculées "
                f"(la plupart des valeurs sont manquantes). Assurez-vous que le panneau a "
                f"fini de calculer à l'écran avant de relancer l'analyse."
            )
    elif analysis_scope == "full":
        fractions = {k: _fraction_empty(v) for k, v in PANEL_DATA.items()}
        if all(f >= 0.9 for f in fractions.values()):
            print(f"Pre-flight check failed: all 4 pillars are essentially empty "
                  f"({fractions}) — skipping Claude call.", flush=True)
            return (
                "Aucune donnée n'a pu être récupérée pour cette analyse. "
                "Veuillez vérifier que la zone et les panneaux ont bien fini de "
                "calculer avant de relancer l'analyse."
            )

    # PRINT INCOMING DATA IMMEDIATELY TO TERMINAL
    print(" DATA SHARED FROM REACT TO PYTHON:")
    print(f"   Competitive Math: {json.dumps(advanced_metrics, indent=2, ensure_ascii=False)}")
    print(f"   Site Score State: {json.dumps(site_score, indent=2, ensure_ascii=False)}")
    print(f"   Retention State: {json.dumps(retention, indent=2, ensure_ascii=False)}")
    print(f"   Implantation State: {json.dumps(implementation, indent=2, ensure_ascii=False)}")
    print(f"   Demographics: Pop={population}, Households={households}, Income={median_income}, Age={avg_age}", flush=True)

    # SECTOR BREAKDOWN
    sector_summary = {}
    for b in businesses:
        sec = b.get("sector") or b.get("category", "Autre")
        sector_summary[sec] = sector_summary.get(sec, 0) + 1
    summary_str = "\n".join([f"- {sec}: {count}" for sec, count in sector_summary.items()])

    # Build each pillar's data block once, so single-scope can include
    # ONLY the requested one — no data for the other 3 means the model
    # literally has nothing to write about them, instead of relying on
    # instructions alone to keep it brief.
    competitive_block = f"""
    CONCURRENCE ({len(businesses)} commerces) :
    - ISM (Saturation) : {advanced_metrics.get('saturation_marche_ism', 'N/A')}
    - DCP (Distance concurrent) : {advanced_metrics.get('distance_concurrent_proche_dcp', 'N/A')}
    - Secteurs : {summary_str}
    """

    retention_block = f"""
    RÉTENTION COMMERCIALE :
    - Pouvoir d'achat disponible (PAD) : {retention.get('pouvoir_achat_disponible_pad', 'N/A')}
    - Diversité (Shannon) : {retention.get('diversite_shannon', 'N/A')}
    - Indice de potentiel de dépense (SPI) : {retention.get('indice_spi', 'N/A')}
    - Dépenses clés par ménage :
      * Alimentation : {retention.get('depenses_alimentation', 'N/A')}
      * Transport : {retention.get('depenses_transport', 'N/A')}
      * Loisirs : {retention.get('depenses_loisirs', 'N/A')}
      * Santé : {retention.get('depenses_sante', 'N/A')}
      * Ameublement : {retention.get('depenses_ameublement', 'N/A')}
      * Vêtements : {retention.get('depenses_vetements', 'N/A')}
      * Soins personnels : {retention.get('depenses_soins_personnels', 'N/A')}
      * Tabac & alcool : {retention.get('depenses_tabac_alcool', 'N/A')}
    """

    site_score_block = f"""
    SCORE D'EMPLACEMENT :
    - Score global : {site_score.get('score_global', 'N/A')}
    - Indice de saturation : {site_score.get('indice_saturation', 'N/A')}
    - Compatibilité réglementaire : {site_score.get('compatibilite_reglementaire', 'N/A')}
    - Score de demande estimée : {site_score.get('score_demande_estimee', 'N/A')}
    - Taux de vacance commercial : {site_score.get('taux_vacance', 'N/A')}
    - Écart offre-demande (DSG) : {site_score.get('ecart_offre_demande', 'N/A')}
    """

    implementation_block = f"""
    IMPLANTATION :
    - Mode ciblé : {implementation.get('mode_actuel', 'N/A')}
    - Score composite (SPC) : {implementation.get('score_composite_spc', 'N/A')}
    - Points clés : {implementation.get('points_cles', 'N/A')}
    - Compatibilité réglementaire : {implementation.get('compatibilite_reglementaire', 'N/A')}
    - Taux de vacance : {implementation.get('taux_vacance', 'N/A')}
    - Écart offre-demande (DSG) : {implementation.get('ecart_offre_demande', 'N/A')}
    - Concurrence directe : {implementation.get('concurrence_directe', 'N/A')}
    """

    ALL_BLOCKS = {
        "competitive": competitive_block,
        "retention": retention_block,
        "siteScore": site_score_block,
        "implementation": implementation_block,
    }

    if analysis_scope == "single" and focus_panel in PANEL_LABELS:
        # Only the requested pillar's data is included — the model has
        # nothing to write about the other 3, so it can't "still show everything."
        focus_label = PANEL_LABELS[focus_panel]
        pillars_text = f"=== ANALYSE DEMANDÉE : {focus_label} ===\n" + ALL_BLOCKS[focus_panel]
        instructions = f"""
    === INSTRUCTIONS OBLIGATOIRES ===
    L'utilisateur a demandé UNIQUEMENT une analyse approfondie du pilier "{focus_label}".
    Ne mentionnez PAS les 3 autres piliers — vous n'avez de toute façon pas leurs données.
    Rédigez 3 à 5 paragraphes détaillés, avec un titre principal et des sous-titres si utile :
    - Interprétez chaque indicateur en profondeur.
    - Comparez-les à des repères sectoriels usuels quand c'est pertinent.
    - Identifiez les risques et opportunités spécifiques à ce pilier.
    - Terminez par des recommandations stratégiques concrètes et actionnables.

    Ton hautement professionnel. N'inventez aucune donnée.
    """
    else:
        pillars_text = (
            "=== LES 4 PILIERS DE L'ANALYSE ===\n"
            + competitive_block + retention_block + site_score_block + implementation_block
        )
        instructions = """
    === INSTRUCTIONS OBLIGATOIRES ===
    Rédigez un rapport stratégique structuré avec 4 sous-titres clairs (un pour chaque pilier).
    Chaque section doit faire 2-3 phrases maximum — soyez concis et direct, pas de remplissage.
    - Dans la section Concurrence, analysez la saturation (ISM) et la distance au concurrent le plus proche.
    - Dans la section Rétention, analysez le PAD, la balance commerciale et comment les résidents dépensent leur argent (Alimentation, Transport, etc.).
    - Dans la section Score d'Emplacement, évaluez le risque lié au taux d'inoccupation et vérifiez la compatibilité du zonage.
    - Dans la section Implantation, donnez un verdict final (Go / No-Go) et recommandez le type de commerce idéal.

    Ton hautement professionnel. N'inventez aucune donnée.
    """

    # THE MASTER PROMPT
    prompt = f"""
    Vous êtes un expert analyste en géomarketing et intelligence territoriale.

    === CONTEXTE DÉMOGRAPHIQUE ===
    - Population : {population} | Ménages : {households} | Revenu médian : {median_income} CAD | Âge moyen : {avg_age} ans

    {pillars_text}
    {instructions}
    """

    # GENERATE THE REPORT WITH CLAUDE
    print(f"Claude is generating the report (scope={analysis_scope}, focus={focus_panel})...", flush=True)
    try:
        t_llm = time.perf_counter()
        max_tokens = 2048 if analysis_scope == "single" else 1024
        message = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        final_analysis = message.content[0].text.strip()
        print(f"Turn 2 LLM generation took {time.perf_counter() - t_llm:.2f}s "
              f"for {len(final_analysis)} characters", flush=True)
        print(f"\n LLM OUTPUT GENERATED SUCCESSFULLY:\n{final_analysis}\n", flush=True)
    except Exception as ai_error:
        print(f"Claude API Error: {ai_error}", flush=True)
        final_analysis = "Erreur de connexion à l'API Claude."

    return final_analysis


@app.post("/api/rag/chat-turn-2")
@limiter.limit("15/minute")
def chat_turn_2(request: Request, payload: dict):
    t0 = time.perf_counter()
    try:
        print("\nTurn 2: Received 4-Pillar Data!", flush=True)

        final_text = generate_pillar_report(
            advanced_metrics=payload.get("advancedMetrics", {}),
            site_score=payload.get("siteScoreMetrics", {}),
            retention=payload.get("retentionMetrics", {}),
            implementation=payload.get("implementationMetrics", {}),
            population=payload.get("Population_2021", 0),
            households=payload.get("menages_total", "Non disponible"),
            median_income=payload.get("revenu_median", "Non disponible"),
            avg_age=payload.get("age_moyen", "Non disponible"),
            businesses=payload.get("_businesses") or payload.get("businesses", []),
            analysis_scope=payload.get("analysisScope", "full"),
            focus_panel=payload.get("focusPanel"),
        )

        print(f" Turn 2 total (incl. payload processing): {time.perf_counter() - t0:.2f}s", flush=True)
        return {"final_text": final_text}

    except Exception as e:
        print(f"Critical Turn 2 Crash: {e}", flush=True)
        return {"final_text": "Erreur lors du traitement des indicateurs."}

if __name__ == "__main__":
    print(" [2/4] Launching Uvicorn process...", flush=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)