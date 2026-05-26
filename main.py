
"""
baseline_rag_v2.py — Territoire Intelligence RAG
==================================================
Run:  python baseline_rag_v2.py [--radius M] [--debug]

THREE-PATH ARCHITECTURE
───────────────────────
Every question is routed to exactly one of three paths:

  PATH 1 — SPATIAL  (Python, <2s including geocode)
    "top 5 closest to the bus terminal", "le local le plus proche de la gare"
    → Haversine distance computed in Python from MongoDB coordinates.
      Reference point extracted from question and geocoded if needed.

  PATH 2 — FILTER   (Python, <1ms)
    "liste tous les bureaux", "show all industrial locals", "combien de locaux"
    → Pure in-memory dict lookup, zero LLM.

  PATH 3 — QUALITATIVE  (LLM, 10-60s)
    "quel secteur convient à un restaurant?", "pourquoi tant de locaux vacants?"
    → Small prompt: demographics + NAICS only, NO full locals list.

Context (MongoDB query) is fetched ONCE per location and reused for all
questions — no re-embedding between questions.
"""

import os
import sys
import math
import re
import time
import argparse

from pymongo import MongoClient
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "DB_NAME":          "overture_maps",
    "MONGO_TIMEOUT_MS": 5000,
    "LOCAUX_LIMIT":     100,      # fetch generously; Python ranks them all
    "NAICS_LIMIT":      50,
    "EMBED_MODEL":      "nomic-embed-text",
    "LLM_MODEL":        "llama3.2:3b",
    "LLM_TEMPERATURE":  0,
    "NOTES_MIN_LENGTH": 50,
    "CHUNK_SIZE":       400,
    "CHUNK_OVERLAP":    40,
    "RETRIEVER_K":      6,
    "GEOCODE_DELAY_S":  1.0,
    "DEFAULT_TOP_N":    5,
}

NO_DATA_MSG   = "Information non disponible dans le contexte."
_EMPTY        = {None, "None", "", "N/A", "null", "n/a"}   # MongoDB null sentinel values

# =============================================================================
# HELPERS
# =============================================================================
def _field(label: str, value, suffix: str = "") -> str:
    """Emit '- Label : value\n' only when value is real data, never for nulls."""
    if value in _EMPTY:
        return ""
    return f"- {label} : {value}{suffix}\n"


# =============================================================================
# ENV + MONGO
# =============================================================================
def validate_env() -> None:
    if not os.getenv("MONGO_URI"):
        print("[STARTUP ERROR] MONGO_URI missing from .env")
        sys.exit(1)
    print("[✓] .env OK")


def get_mongo_client() -> MongoClient:
    return MongoClient(
        os.getenv("MONGO_URI", ""),
        serverSelectionTimeoutMS=CONFIG["MONGO_TIMEOUT_MS"],
    )


# =============================================================================
# GEOCODING  (used for both session location and question reference points)
# =============================================================================
_geocoder = Nominatim(user_agent="territoire_intelligence_rag_v2")  # single instance

def geocode_address(address: str, context: str = "Quebec, Canada") -> tuple[float | None, float | None]:
    """
    Convert a free-text address to (lon, lat).
    `context` is appended to the query to bias results toward Quebec.
    Returns (None, None) on failure.
    """
    query = f"{address}, {context}" if context not in address else address
    print(f"  Geocoding: '{address}'...")
    try:
        time.sleep(CONFIG["GEOCODE_DELAY_S"])
        loc = _geocoder.geocode(query, timeout=10)
        if loc:
            print(f"    → ({loc.longitude:.5f}, {loc.latitude:.5f})")
            return loc.longitude, loc.latitude
        print(f"    → Could not resolve.")
        return None, None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"    → Geocoder error: {e}")
        return None, None


# =============================================================================
# DATA RETRIEVAL  — called ONCE per location
# =============================================================================
def retrieve_territorial_context(
    db, user_lon: float, user_lat: float, radius_meters: int, debug: bool = False
) -> tuple[str, list[Document], list[dict], dict, list[str]]:
    """
    Returns
    -------
    qualitative_str   Compact LLM context: demographics + economic sectors only.
                      The full locals list is intentionally absent — it stays in
                      raw_locals / local_index and is answered by Python directly.
    unstructured_docs Long free-text notes → FAISS vector store.
    raw_locals        Raw MongoDB dicts (with geometry) for spatial ranking.
    demographics      Raw demographics dict.
    naics_lines       List of formatted NAICS sector strings.
    """
    qualitative_parts : list[str]      = []
    naics_lines       : list[str]      = []
    unstructured_docs : list[Document] = []
    raw_locals        : list[dict]     = []
    demographics      : dict           = {}

    geo_point = {"type": "Point", "coordinates": [user_lon, user_lat]}

    # Define locaux_query once here so NAICS can reuse it safely
    locaux_query = {
        "geometry": {
            "$near": {
                "$geometry": geo_point,
                "$maxDistance": radius_meters,
            }
        }
    }

    # ── 1. Demographics ───────────────────────────────────────────────────────
    print(" [1/3] Demographics...")
    try:
        city_data = db.donnees.find_one({
            "geometry": {"$geoIntersects": {"$geometry": geo_point}}
        })
        if city_data:
            demographics = city_data
            demo  = "### Données Socio-Démographiques\n"
            demo += _field("Population 2021",  city_data.get("Population_2021"))
            demo += _field("Population 2016",  city_data.get("Population_2016"))
            demo += _field("Logements privés", city_data.get("Logements_prives_total"))
            demo += _field("Densité pop/km²",  city_data.get("Densite_pop_km2"))
            demo += _field("Superficie km²",   city_data.get("Superficie_km2"))
            qualitative_parts.append(demo)
            print(f"   → Pop 2021: {city_data.get('Population_2021', '?')}")
        else:
            print("   → No demographic zone found at these coordinates.")
    except Exception as e:
        print(f"   ✗ Demographics error: {e}")

    # ── 2. Locaux vacants ─────────────────────────────────────────────────────
    # Raw docs kept in raw_locals for spatial engine; never sent to LLM.
    print(f" [2/3] Vacant locals within {radius_meters}m...")
    try:
        found_items = list(
            db.locaux_vacants.find(locaux_query).limit(CONFIG["LOCAUX_LIMIT"])
        )
        raw_locals = found_items
        print(f"   → {len(found_items)} locals loaded into Python index.")

        for local in found_items:
            adresse = local.get("adresse", "adresse inconnue")
            notes   = local.get("notes", "")
            if notes and len(str(notes)) > CONFIG["NOTES_MIN_LENGTH"]:
                unstructured_docs.append(Document(
                    page_content=f"Notes pour le local au {adresse}: {notes}",
                    metadata={"type": "local_notes", "adresse": adresse},
                ))
    except Exception as e:
        print(f"   ✗ Locaux error (check 2dsphere index on locaux_vacants.geometry): {e}")

    # ── 3. NAICS — text-match from nearby business categories ─────────────────
    print(" [3/3] Economic sectors...")
    try:
        nearby_businesses = list(
            db.quebec_businesses.find(locaux_query).limit(CONFIG["NAICS_LIMIT"])
        )
        if nearby_businesses:
            raw_categories: set[str] = set()
            for b in nearby_businesses:
                for fld in ("categorie", "category", "type", "secteur", "type_entreprise"):
                    val = b.get(fld)
                    if val and str(val).strip() not in _EMPTY:
                        raw_categories.add(str(val).strip().lower())

            if raw_categories:
                all_naics = list(db.naics.find({}))
                matched: list[dict] = []
                for sector in all_naics:
                    label = str(sector.get("label", "")).lower()
                    desc  = str(sector.get("description", "")).lower()
                    if any(cat in label or cat in desc or label in cat
                           for cat in raw_categories):
                        matched.append(sector)

                header = "\n### Secteurs Économiques dans la Zone"
                if matched:
                    qualitative_parts.append(header)
                    for s in matched:
                        line = f"- {s.get('label', 'N/A')}"
                        if s.get("code") not in _EMPTY:
                            line += f" (NAICS {s.get('code')})"
                        naics_lines.append(line)
                        qualitative_parts.append(line)
                    print(f"   → {len(matched)} NAICS sector(s) matched.")
                else:
                    qualitative_parts.append(header)
                    for cat in sorted(raw_categories):
                        line = f"- {cat.title()}"
                        naics_lines.append(line)
                        qualitative_parts.append(line)
                    print(f"   → No NAICS label match; {len(raw_categories)} raw categories used.")
            else:
                print("   → No category/type fields on business documents.")
        else:
            print("   → No businesses found in radius.")
    except Exception as e:
        print(f"   ✗ NAICS error: {e}")

    qualitative_str = (
        "\n".join(qualitative_parts)
        if qualitative_parts
        else "Aucune donnée de contexte disponible pour cette zone."
    )

    if debug:
        print("\n─── LLM CONTEXT (qualitative only) ───")
        print(qualitative_str)
        print(f"─── {len(raw_locals)} locals in Python index ───\n")

    return qualitative_str, unstructured_docs, raw_locals, demographics, naics_lines


# =============================================================================
# STRUCTURED INDEX  — clean dict per local, built from raw MongoDB docs
# =============================================================================
def build_structured_index(raw_locals: list[dict]) -> list[dict]:
    """
    Build a clean display dict for each local.
    Only fields with real data are kept (_field logic applied at query time).
    """
    index = []
    for doc in raw_locals:
        entry: dict = {}
        for mongo_key, display_key in [
            ("adresse",        "Adresse"),
            ("ville",          "Ville"),
            ("type_local",     "Type"),
            ("secteur",        "Secteur"),
            ("prix",           "Prix"),
            ("superficie_pi2", "Superficie"),
            ("contact",        "Contact"),
            ("url_source",     "URL"),
            ("notes",          "Notes"),
        ]:
            val = doc.get(mongo_key)
            if val is not None and val not in _EMPTY:
                entry[display_key] = val
        if doc.get("actif") is False:
            entry["Statut"] = "Inactif"
        if entry.get("Adresse"):
            index.append(entry)
    return index


def format_local(entry: dict, rank: int, dist_str: str = "") -> str:
    """Format one local entry. Shows distance when provided."""
    header = f"**#{rank}" + (f" — {dist_str}" if dist_str else "") + "**"
    lines  = [header]
    for key in ["Adresse", "Ville", "Type", "Secteur", "Prix",
                "Superficie", "Contact", "Statut"]:
        if key in entry:
            suffix = " pi²" if key == "Superficie" else ""
            lines.append(f"  - {key} : {entry[key]}{suffix}")
    return "\n".join(lines)


# =============================================================================
# SPATIAL ENGINE
# =============================================================================
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres. Error < 0.5% at city scale."""
    R    = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _coords_from_doc(doc: dict) -> tuple[float, float] | None:
    """Extract (lon, lat) from GeoJSON geometry or flat lon/lat fields."""
    geo = doc.get("geometry")
    if isinstance(geo, dict) and geo.get("type") == "Point":
        coords = geo.get("coordinates", [])
        if len(coords) == 2:
            try:
                return float(coords[0]), float(coords[1])
            except (TypeError, ValueError):
                pass
    for lon_k, lat_k in [("longitude", "latitude"), ("lon", "lat")]:
        lon_v = doc.get(lon_k)
        lat_v = doc.get(lat_k)
        if lon_v is not None and lat_v is not None:
            try:
                return float(lon_v), float(lat_v)
            except (TypeError, ValueError):
                pass
    return None


# =============================================================================
# INTENT PARSER
# Three-way classifier: SPATIAL → PATH 1, FILTER → PATH 2, else → PATH 3 (LLM)
# =============================================================================

# ── Qualitative overrides (always LLM regardless of other signals) ────────────
_QUALITATIVE_OVERRIDES = [
    # French
    "recommande", "recommandez", "convient", "conviendrait",
    "adapté", "suggère", "suggérez",
    "meilleur pour", "meilleure pour",
    "pourquoi", "analyse", "expliquez", "expliquer",
    "est-ce que", "est-ce qu",
    "serait-il", "devrait", "faut-il",
    "avis sur", "opinion", "pensez-vous",
    "secteur économique", "données démographiques",
    "revenu", "densité", "superficie",
    # English
    "recommend", "suitable for", "best for", "advise",
    "why", "explain", "analysis", "analyze",
    "should i", "would you suggest",
]

# ── Spatial triggers (require proximity/ranking intent + a reference entity) ──
# IMPORTANT: kept as multi-word phrases to avoid false positives on common words.
_SPATIAL_TRIGGERS = [
    # French — multi-word only
    "plus proche", "plus proches", "le plus proche", "les plus proches",
    "à proximité de", "près de ", "proche de ", "autour de ",
    "à moins de", "dans un rayon de",
    "classer par distance", "trier par distance",
    "classé par distance", "ordonné par distance",
    # English — multi-word only
    "closest to", "nearest to", "near the", "close to the",
    "within ", "sort by distance", "ranked by distance",
    "distance from", "km from", "meters from",
    "top 1", "top 2", "top 3", "top 4", "top 5",
    "top 6", "top 7", "top 8", "top 9", "top 10",
]

# ── Filter triggers (listing without spatial ranking) ─────────────────────────
_FILTER_TRIGGERS = [
    # French
    "liste", "lister", "listez", "liste-moi",
    "tous les locaux", "toutes les propriétés",
    "quels sont les locaux", "quels locaux",
    "locaux vacants", "locaux disponibles",
    "combien de locaux", "combien y a-t-il",
    "affiche les locaux", "montre les locaux",
    # English
    "list all", "list the", "show all", "show me all",
    "all vacant", "all available locals",
    "how many locals", "how many vacant",
    "vacant locals", "available locals",
]

# ── Price/sort triggers ───────────────────────────────────────────────────────
_PRICE_SORT_TRIGGERS = [
    "moins cher", "moins chère", "le moins cher", "les moins chers",
    "plus abordable", "prix le plus bas", "tarif le moins élevé",
    "cheapest", "lowest price", "most affordable", "sort by price",
]


def classify_question(question: str) -> str:
    """
    Returns one of: 'spatial', 'filter', 'price_sort', 'qualitative'.
    Qualitative overrides always win.
    """
    q = question.lower()

    # Qualitative veto — always goes to LLM
    if any(ov in q for ov in _QUALITATIVE_OVERRIDES):
        return "qualitative"

    # Price sort — Python handles, sorted by price
    if any(t in q for t in _PRICE_SORT_TRIGGERS):
        return "price_sort"

    # Spatial ranking — Python + haversine + geocode
    if any(t in q for t in _SPATIAL_TRIGGERS):
        return "spatial"

    # Plain filter / listing
    if any(t in q for t in _FILTER_TRIGGERS):
        return "filter"

    # Default: let the LLM try
    return "qualitative"


def _parse_top_n(question: str, default: int | None = None) -> int:
    """Extract requested N from question. Returns CONFIG default if not found."""
    default = default or CONFIG["DEFAULT_TOP_N"]
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bles\s+(\d+)\s+plus\s+proch",
        r"\bles\s+(\d+)\s+premiers?\b",
        r"\b(\d+)\s+plus\s+proch",
        r"\bgive\s+me\s+(\d+)\b",
        r"\bshow\s+(?:me\s+)?(\d+)\b",
        r"\bfind\s+(?:me\s+)?(\d+)\b",
        r"\b(\d+)\s+(?:closest|nearest)\b",
        r"\b(\d+)\s+locaux\b",
        r"\bles\s+(\d+)\b",
    ]
    for p in patterns:
        m = re.search(p, question.lower())
        if m:
            return max(1, min(int(m.group(1)), 50))
    return default


def _parse_type_filter(question: str) -> list[str]:
    """Extract property type(s) from question. Returns [] for no filter."""
    q = question.lower()
    type_map = {
        "bureau":      ["bureau", "bureaux", "office", "professionnel"],
        "commercial":  ["commercial", "commerce", "magasin", "retail", "boutique"],
        "industriel":  ["industriel", "industrial", "entrepôt", "warehouse", "usine"],
        "résidentiel": ["résidentiel", "residential", "logement", "appartement"],
    }
    return [
        canonical for canonical, kws in type_map.items()
        if any(kw in q for kw in kws)
    ]


def _apply_type_filter(scored: list, type_filter: list[str]) -> list:
    """Filter a scored list to only entries matching the requested type(s)."""
    if not type_filter:
        return scored
    return [
        item for item in scored
        if any(t in str(item[-1].get("Type", "")).lower() for t in type_filter)
    ]


# =============================================================================
# REFERENCE POINT EXTRACTOR
# =============================================================================

# Patterns that capture what comes after a locative preposition
_REF_PATTERNS = [
    # "les 5 plus proches du / de la / de / des <landmark>"
    r"(?:plus proches?|nearest|closest)\s+(?:du|de la|de l[''']|des|de|from|to|near)\s+(.+?)(?:\s*[?!.]|$)",
    # "à proximité de / près de <landmark>"
    r"(?:à proximité|près)\s+(?:du|de la|de l[''']|des|de)\s+(.+?)(?:\s*[?!.]|$)",
    # "autour de <landmark>"
    r"(?:autour de|around)\s+(.+?)(?:\s*[?!.]|$)",
    # "near / close to / from <landmark>"
    r"(?:near the|close to the|near|from)\s+(.+?)(?:\s*[?!.]|$)",
    # "within Xkm of <landmark>"
    r"(?:within\s+\S+\s+(?:km|m|meters|kilometres)\s+of)\s+(.+?)(?:\s*[?!.]|$)",
]

def _geocode_reference_point(
    question: str,
    session_lon: float,
    session_lat: float,
    city_hint: str = "Saint-Hyacinthe, Quebec, Canada",
) -> tuple[float, float, str]:
    """
    Extract and geocode the reference landmark from a spatial question.

    Tries each regex pattern in order. For each candidate extracted:
      1. Try geocoding with city hint for precision.
      2. If that fails, try without hint (handles full addresses).
    Falls back to session coordinates if nothing resolves.

    Returns (lon, lat, description).
    """
    seen: set[str] = set()

    for pattern in _REF_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(1).strip().rstrip("?!. ")
        if len(candidate) < 4 or candidate in seen:
            continue
        seen.add(candidate)

        # Try with city hint first, then bare
        for query in (f"{candidate}, {city_hint}", candidate):
            lon, lat = geocode_address(candidate, context=city_hint if "hint" in query else "")
            if lon is not None:
                return lon, lat, candidate

    print("   → No reference point found; using session location.")
    return session_lon, session_lat, "votre position actuelle"


# =============================================================================
# PATH 1 — SPATIAL ANSWER
# =============================================================================
def answer_spatial(
    question    : str,
    local_index : list[dict],
    raw_locals  : list[dict],
    session_lon : float,
    session_lat : float,
    city_hint   : str = "Saint-Hyacinthe, Quebec, Canada",
) -> str:
    """
    Rank all indexed locals by Haversine distance from the reference point
    extracted from the question. Returns top-N formatted string.
    Always called via PATH 1 — never returns None.
    """
    top_n       = _parse_top_n(question)
    type_filter = _parse_type_filter(question)

    ref_lon, ref_lat, ref_desc = _geocode_reference_point(
        question, session_lon, session_lat, city_hint
    )

    # Build address → raw doc lookup for coordinate access
    raw_by_adresse = {
        doc.get("adresse", ""): doc
        for doc in raw_locals
        if doc.get("adresse")
    }

    scored: list[tuple[float, dict]] = []
    no_coords = 0
    for entry in local_index:
        raw_doc = raw_by_adresse.get(entry.get("Adresse", ""))
        if not raw_doc:
            no_coords += 1
            continue
        coords = _coords_from_doc(raw_doc)
        if not coords:
            no_coords += 1
            continue
        dist = haversine_m(ref_lon, ref_lat, coords[0], coords[1])
        scored.append((dist, entry))

    if no_coords:
        print(f"   → {no_coords} local(s) skipped (no geometry in MongoDB).")

    if not scored:
        return (
            "Aucun local avec coordonnées géographiques trouvé.\n"
            "Assurez-vous que vos documents ont un champ 'geometry' de type GeoJSON Point."
        )

    # Apply optional type filter
    scored = _apply_type_filter(scored, type_filter)
    if not scored:
        return f"Aucun local de type '{', '.join(type_filter)}' trouvé dans le rayon."

    scored.sort(key=lambda x: x[0])
    results = scored[:top_n]

    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**Top {len(results)} local(aux){type_label} — classés par distance de {ref_desc} :**\n"]

    for rank, (dist_m, entry) in enumerate(results, 1):
        dist_str = f"{dist_m:,.0f} m" if dist_m < 1000 else f"{dist_m / 1000:.2f} km"
        lines.append(format_local(entry, rank, dist_str))
        lines.append("")

    remaining = len(scored) - top_n
    if remaining > 0:
        lines.append(
            f"*{remaining} autre(s) local/locaux non affichés. "
            f"Demandez \"top {top_n + 5}\" pour en voir plus.*"
        )

    return "\n".join(lines)


# =============================================================================
# PATH 2 — FILTER ANSWER  (plain listing, no ranking)
# =============================================================================
def answer_filter(
    question    : str,
    local_index : list[dict],
) -> str:
    """
    Return all locals matching an optional type filter, unordered.
    Always called via PATH 2 — never returns None.
    """
    type_filter = _parse_type_filter(question)
    filtered    = _apply_type_filter([(0, e) for e in local_index], type_filter)
    entries     = [e for _, e in filtered]

    if not entries:
        return "Aucun local de type demandé trouvé dans le rayon de recherche."

    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**{len(entries)} local(aux){type_label} dans le rayon :**\n"]
    for i, entry in enumerate(entries, 1):
        lines.append(format_local(entry, i))
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# PATH 2b — PRICE SORT ANSWER
# =============================================================================
def _parse_price(prix_str: str) -> float:
    """Extract a numeric value from a price string for sorting. Returns inf if unparseable."""
    try:
        digits = re.sub(r"[^\d.]", "", str(prix_str).replace(",", "."))
        return float(digits) if digits else float("inf")
    except (ValueError, TypeError):
        return float("inf")


def answer_price_sort(
    question    : str,
    local_index : list[dict],
) -> str:
    """Sort locals by price ascending. Used for 'show me the cheapest' questions."""
    type_filter = _parse_type_filter(question)
    top_n       = _parse_top_n(question, default=len(local_index))

    filtered = [e for e in local_index if e.get("Prix")]
    if type_filter:
        filtered = [
            e for e in filtered
            if any(t in str(e.get("Type", "")).lower() for t in type_filter)
        ]

    if not filtered:
        return "Aucun local avec prix disponible dans le rayon de recherche."

    filtered.sort(key=lambda e: _parse_price(e.get("Prix", "")))
    results = filtered[:top_n]

    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**{len(results)} local(aux){type_label} — classés par prix croissant :**\n"]
    for i, entry in enumerate(results, 1):
        lines.append(format_local(entry, i))
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# PATH 3 — QUALITATIVE CHAIN BUILDER
# =============================================================================
def build_qualitative_chain(
    qualitative_str  : str,
    unstructured_docs: list[Document],
):
    """
    LangChain RAG chain for qualitative/synthesis questions.
    Prompt is intentionally compact — only demographics + NAICS, no locals list.
    Built ONCE per location, reused for all qualitative questions.
    """
    print("\n[BUILD] Preparing qualitative LLM chain...")
    llm        = ChatOllama(model=CONFIG["LLM_MODEL"], temperature=CONFIG["LLM_TEMPERATURE"])
    embeddings = OllamaEmbeddings(model=CONFIG["EMBED_MODEL"])

    if unstructured_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
        )
        splits       = splitter.split_documents(unstructured_docs)
        vector_store = FAISS.from_documents(splits, embeddings)
        k            = min(len(splits), CONFIG["RETRIEVER_K"])
        print(f"[BUILD] Notes vector store: {len(splits)} chunks, k={k}.")
    else:
        vector_store = FAISS.from_texts(["Aucune note textuelle disponible."], embeddings)
        k = 1
        print("[BUILD] No notes — dummy vector store.")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        """You are a territorial analysis assistant for Quebec municipalities.

RULES:
1. Answer ONLY from the context below. Never invent data.
2. If the answer is absent, say exactly: "Information non disponible dans le contexte."
3. Respond in the same language as the question.
4. Be concise. Do not mention specific property addresses or prices \
(those are handled by the spatial engine separately).

[TERRITORIAL CONTEXT]
{structured_data}

[RETRIEVED NOTES]
{context}

Question: {input}
"""
    )
    prompt = prompt.partial(structured_data=qualitative_str)

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt),
    )
    print("[BUILD] Qualitative chain ready.\n")
    return chain


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Territoire Intelligence RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radius", type=int,  default=5000,
                        help="MongoDB fetch radius in metres")
    parser.add_argument("--debug",  action="store_true",
                        help="Print LLM context and routing decisions")
    args = parser.parse_args()

    validate_env()
    mongo_client = get_mongo_client()
    db           = mongo_client[CONFIG["DB_NAME"]]

    print(f"\nTerritoire Intelligence RAG — radius={args.radius}m")
    print("Commands: 'exit' to quit | 'change location' to switch area\n")

    try:
        while True:
            # ── Location input ────────────────────────────────────────────────
            location_input = input("Enter an address or city (Quebec): ").strip()
            if not location_input:
                continue
            if location_input.lower() in ("exit", "quit", "q"):
                print("Au revoir!")
                break

            lon, lat = geocode_address(location_input)
            if lon is None:
                print("Could not resolve location. Try a more specific address.\n")
                continue

            city_hint = f"{location_input}, Quebec, Canada"

            # ── Fetch context ONCE for this location ──────────────────────────
            print(f"\nFetching context for '{location_input}'...")
            t0 = time.perf_counter()

            qualitative_str, notes_docs, raw_locals, demographics, naics_lines = (
                retrieve_territorial_context(db, lon, lat, args.radius, debug=args.debug)
            )
            local_index = build_structured_index(raw_locals)
            llm_chain   = build_qualitative_chain(qualitative_str, notes_docs)

            elapsed = time.perf_counter() - t0
            print(f"[✓] {len(local_index)} locals indexed | ready in {elapsed:.1f}s")
            print(f"[✓] Session: '{location_input}' | radius: {args.radius}m")
            print("    Type your question or 'change location'.\n")

            # ── Q&A loop ──────────────────────────────────────────────────────
            while True:
                try:
                    user_q = input(f"Question ({location_input}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAu revoir!")
                    return

                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Au revoir!")
                    return
                if user_q.lower() == "change location":
                    print("\nChanging location...\n")
                    break

                t0     = time.perf_counter()
                intent = classify_question(user_q)

                if args.debug:
                    print(f"   [ROUTER] intent={intent}")

                # ── PATH 1: Spatial ranking ───────────────────────────────────
                if intent == "spatial":
                    answer  = answer_spatial(
                        user_q, local_index, raw_locals, lon, lat, city_hint
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.1f}s — spatial):\n{answer}\n")

                # ── PATH 2: Filter / listing ──────────────────────────────────
                elif intent == "filter":
                    answer  = answer_filter(user_q, local_index)
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.3f}s — filter):\n{answer}\n")

                # ── PATH 2b: Price sort ───────────────────────────────────────
                elif intent == "price_sort":
                    answer  = answer_price_sort(user_q, local_index)
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.3f}s — price sort):\n{answer}\n")

                # ── PATH 3: LLM — qualitative / synthesis ─────────────────────
                else:
                    try:
                        result  = llm_chain.invoke({"input": user_q})
                        answer  = result.get("answer", "").strip() or NO_DATA_MSG
                        elapsed = time.perf_counter() - t0
                        print(f"\nRÉPONSE ({elapsed:.1f}s — LLM):\n{answer}\n")
                    except Exception as e:
                        print(f"\n[ERROR] LLM inference failed: {e}\n")

    finally:
        mongo_client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    main()




