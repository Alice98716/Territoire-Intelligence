"""

baseline_rag_v3.py — Territoire Intelligence RAG
==================================================
3 paths
Added quebec_business data 
─────────────────────────────────────────────────────────
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#All parameters in one place 
CONFIG = {
    "DB_NAME":          "overture_maps",
    "MONGO_TIMEOUT_MS": 5000,
    "LOCAUX_LIMIT":     100,      
    "NAICS_LIMIT":      50,
    "EMBED_MODEL":      "nomic-embed-text",
    "LLM_MODEL":        "llama3.2:1b",
    "LLM_TEMPERATURE":  0,
    "NOTES_MIN_LENGTH": 50,
    "CHUNK_SIZE":       400,
    "CHUNK_OVERLAP":    40,
    "RETRIEVER_K":      6,
    "GEOCODE_DELAY_S":  1.0,
    "DEFAULT_TOP_N":    5,
    "CHAT_MEMORY_LIMIT": 6, #keeps the last 6 messages to avoid context overflow
}

NO_DATA_MSG   = "Information non disponible dans le contexte."
_EMPTY        = {None, "None", "", "N/A", "null", "n/a"}

#helper functions
def _field(label: str, value, suffix: str = "") -> str:

    if value in _EMPTY:
        return ""
    return f"- {label} : {value}{suffix}\n"


#CONNECTION A MONGODB
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


#geocoding position de l'utilisateur => plus tard polygone etc
_geocoder = Nominatim(user_agent="territoire_intelligence_rag_v2")

def geocode_address(address: str, context: str = "Quebec, Canada") -> tuple[float | None, float | None]:

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


#DATA RETRIEVAL 
def retrieve_territorial_context(
    db, user_lon: float, user_lat: float, radius_meters: int, debug: bool = False
) -> tuple[str, list[Document], list[dict], dict, list[str]]:
    
    qualitative_parts : list[str]      = []
    naics_lines       : list[str]      = []
    unstructured_docs : list[Document] = []
    raw_locals        : list[dict]     = []
    demographics      : dict           = {}

    geo_point = {"type": "Point", "coordinates": [user_lon, user_lat]}

    locaux_query = {
        "geometry": {
            "$near": {
                "$geometry": geo_point,
                "$maxDistance": radius_meters,
            }
        }
    }

    #1. Demographics
    print("[1/3] Demographics...")
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
        print(f" Demographics error: {e}")

    #2. Locaux vacants 
    print(f"[2/3] Vacant locals within {radius_meters}m...")
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
        print(f"Locaux error: {e}")

    # 3. NAICS & Local Businesses
    print("[3/3] Nearby businesses and economic sectors...")
    try:
        nearby_businesses = list(
            db.quebec_businesses.find(locaux_query).limit(CONFIG.get("BUSINESS_LIMIT", 50))
        )
        
        if nearby_businesses:
            raw_categories: set[str] = set()
            business_lines: list[str] = ["\n### Entreprises Locales (À Proximité)"]

            for b in nearby_businesses:
                # 1. Extract categories for NAICS mapping
                for fld in ("categorie", "category", "type", "secteur", "type_entreprise"):
                    val = b.get(fld)
                    if val and str(val).strip() not in _EMPTY:
                        raw_categories.add(str(val).strip().lower())
                
                # 2. Extract specific business data for the LLM context
                name = b.get("nom", b.get("name", "Nom inconnu"))
                address = b.get("adresse", b.get("address", ""))
                activity = str(b.get("description", b.get("activite", b.get("secteur", ""))))
                
                details = f"- **{name}**"
                if address and address not in _EMPTY:
                    details += f" ({address})"
                
                if activity and activity not in _EMPTY:
                    #Prevent context window overflow by sending long descriptions to FAISS
                    if len(activity) > 150:
                        unstructured_docs.append(Document(
                            page_content=f"Détails sur l'entreprise {name} située au {address} : {activity}",
                            metadata={"type": "business_info", "nom": name}
                        ))
                        details += f" : {activity[:150]}... (voir notes pour plus de détails)"
                    else:
                        details += f" : {activity}"
                    
                business_lines.append(details)
            
            #structured business part is added to qualitative data
            qualitative_parts.extend(business_lines)

            if raw_categories:
                all_naics = list(db.naics.find({}))
                matched: list[dict] = []
                for sector in all_naics:
                    label = str(sector.get("label", "")).lower()
                    desc  = str(sector.get("description", "")).lower()
                    if any(cat in label or cat in desc or label in cat for cat in raw_categories):
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
                    print(f"   {len(matched)} NAICS sector(s) matched.")
                else:
                    qualitative_parts.append(header)
                    for cat in sorted(raw_categories):
                        line = f"- {cat.title()}"
                        naics_lines.append(line)
                        qualitative_parts.append(line)
                    print(f"    No NAICS label match; {len(raw_categories)} raw categories used.")
            else:
                print("    No category/type fields on business documents.")
        else:
            print("   No businesses found in radius.")
    except Exception as e:
        print(f" NAICS / Businesses error: {e}")


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


#STRUCTURED INDEX
def build_structured_index(raw_locals: list[dict]) -> list[dict]:

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

    header = f"**#{rank}" + (f" — {dist_str}" if dist_str else "") + "**"
    lines  = [header]
    for key in ["Adresse", "Ville", "Type", "Secteur", "Prix",
                "Superficie", "Contact", "Statut"]:
        if key in entry:
            suffix = " pi²" if key == "Superficie" else ""
            lines.append(f"  - {key} : {entry[key]}{suffix}")
    return "\n".join(lines)



def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:

    R    = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _coords_from_doc(doc: dict) -> tuple[float, float] | None:

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


_QUALITATIVE_OVERRIDES = [
    "recommande", "recommandez", "convient", "conviendrait", "adapté", "suggère", 
    "meilleur pour", "meilleure pour", "pourquoi", "analyse", "expliquez", "expliquer",
    "est-ce que", "est-ce qu", "serait-il", "devrait", "faut-il", "avis sur", "opinion", 
    "pensez-vous", "secteur économique", "données démographiques", "revenu", "densité", 
    "superficie", "recommend", "suitable for", "best for", "advise", "why", "explain", 
    "analysis", "analyze", "should i", "would you suggest",
]

_SPATIAL_TRIGGERS = [

    "plus proche", "plus proches", "le plus proche", "les plus proches",
    "à proximité de", "près de ", "proche de ", "autour de ",
    "à moins de", "dans un rayon de", "classer par distance", "trier par distance",
    "closest to", "nearest to", "near the", "close to the",
    "within ", "sort by distance", "ranked by distance",
    "distance from", "km from", "meters from",

]


_FILTER_TRIGGERS = [
    "liste", "lister", "listez", "liste-moi", "tous les locaux", "toutes les propriétés",
    "quels sont les locaux", "quels locaux", "locaux vacants", "locaux disponibles",
    "combien de locaux", "combien y a-t-il", "affiche les locaux", "montre les locaux",
    "list all", "list the", "show all", "show me all", "how many locals",
]
_PRICE_INQUIRY_TRIGGERS = [
    "prix", "loyer", "coût", "tarif", "price", "rent", "cost", "how much", 
    "combien coûte", "moins cher", "plus abordable", "cheapest", "le plus bas"
]


def classify_question(question: str) -> str:

    q = question.lower()

    if any(ov in q for ov in _QUALITATIVE_OVERRIDES):
        return "qualitative"
    if any(t in q for t in _PRICE_INQUIRY_TRIGGERS):
        return "price_inquiry"
    if any(t in q for t in _SPATIAL_TRIGGERS):
        return "spatial"
    
    if any(t in q for t in _FILTER_TRIGGERS):
        return "filter"
    
    return "qualitative"


def _parse_top_n(question: str, default: int | None = None) -> int:
    default = default or CONFIG["DEFAULT_TOP_N"]
    q = question.lower()

    #Map text numbers to digits before processing regex
    text_to_num = {
        "un": "1", "one": "1",
        "deux": "2", "two": "2",
        "trois": "3", "three": "3",
        "quatre": "4", "four": "4",
        "cinq": "5", "five": "5",
        "six": "6", "six": "6",
        "sept": "7", "seven": "7",
        "huit": "8", "eight": "8",
        "neuf": "9", "nine": "9",
        "dix": "10", "ten": "10"
    }
    
    for word, num in text_to_num.items():
        #Match word boundaries to prevent replacing partial words
        q = re.sub(r'\b' + word + r'\b', num, q)

    patterns = [
        r"\btop\s+(\d+)\b", r"\bles\s+(\d+)\s+plus\s+proch", r"\bles\s+(\d+)\s+premiers?\b",
        r"\b(\d+)\s+plus\s+proch", r"\bgive\s+me\s+(\d+)\b", r"\bshow\s+(?:me\s+)?(\d+)\b",
        r"\bfind\s+(?:me\s+)?(\d+)\b", r"\b(\d+)\s+(?:closest|nearest)\b",
        r"\b(\d+)\s+locaux\b", r"\bles\s+(\d+)\b",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            return max(1, min(int(m.group(1)), 50))
    return default


def _parse_type_filter(question: str) -> list[str]:

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

    if not type_filter:
        return scored
    return [
        item for item in scored
        if any(t in str(item[-1].get("Type", "")).lower() for t in type_filter)
    ]


_REF_PATTERNS = [
    r"(?:plus proches?|nearest|closest)\s+(?:du|de la|de l['’]|des|de|from|to|near)\s+(.+?)(?:\s*[?!.]|$)",
    r"(?:à proximité|près)\s+(?:du|de la|de l['’]|des|de)\s+(.+?)(?:\s*[?!.]|$)",
    r"(?:autour de|around)\s+(.+?)(?:\s*[?!.]|$)",
    r"(?:near the|close to the|near|from)\s+(.+?)(?:\s*[?!.]|$)",
    r"(?:within\s+\S+\s+(?:km|m|meters|kilometres)\s+of)\s+(.+?)(?:\s*[?!.]|$)",
]

def _geocode_reference_point(question: str, session_lon: float, session_lat: float, city_hint: str) -> tuple[float, float, str]:

    seen: set[str] = set()
    for pattern in _REF_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if not m: continue
        candidate = m.group(1).strip().rstrip("?!. ")
        if len(candidate) < 4 or candidate in seen: continue
        seen.add(candidate)

        for query in (f"{candidate}, {city_hint}", candidate):
            lon, lat = geocode_address(candidate, context=city_hint if "hint" in query else "")
            if lon is not None:
                return lon, lat, candidate
            
    print("    No reference point found; using session location.")
    return session_lon, session_lat, "votre position actuelle"

def answer_spatial(question: str, local_index: list[dict], raw_locals: list[dict], session_lon: float, session_lat: float, city_hint: str) -> str:
    top_n       = _parse_top_n(question)
    type_filter = _parse_type_filter(question)
    ref_lon, ref_lat, ref_desc = _geocode_reference_point(question, session_lon, session_lat, city_hint)
    raw_by_adresse = {doc.get("adresse", ""): doc for doc in raw_locals if doc.get("adresse")}
    scored: list[tuple[float, dict]] = []
    
    for entry in local_index:
        raw_doc = raw_by_adresse.get(entry.get("Adresse", ""))
        if not raw_doc: continue
        coords = _coords_from_doc(raw_doc)
        if not coords: continue
        dist = haversine_m(ref_lon, ref_lat, coords[0], coords[1])
        scored.append((dist, entry))

    if not scored: return "Aucun local avec coordonnées géographiques trouvé."
    scored = _apply_type_filter(scored, type_filter)
    if not scored: return f"Aucun local de type '{', '.join(type_filter)}' trouvé."

    scored.sort(key=lambda x: x[0])
    results = scored[:top_n]

    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**Top {len(results)} local(aux){type_label} — classés par distance de {ref_desc} :**\n"]

    for rank, (dist_m, entry) in enumerate(results, 1):
        dist_str = f"{dist_m:,.0f} m" if dist_m < 1000 else f"{dist_m / 1000:.2f} km"
        lines.append(format_local(entry, rank, dist_str))
        lines.append("")

    return "\n".join(lines)

def answer_filter(question: str, local_index: list[dict]) -> str:
    type_filter = _parse_type_filter(question)
    filtered    = _apply_type_filter([(0, e) for e in local_index], type_filter)
    entries     = [e for _, e in filtered]
    if not entries: return "Aucun local de type demandé trouvé dans le rayon de recherche."

    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**{len(entries)} local(aux){type_label} dans le rayon :**\n"]
    for i, entry in enumerate(entries, 1):
        lines.append(format_local(entry, i))
        lines.append("")
    return "\n".join(lines)

def _parse_price(prix_str: str) -> float:
    if prix_str is None or str(prix_str).strip() in _EMPTY:
        return float("inf")
        
    p = str(prix_str).strip()
    
    #case 1: thousand seperators
    if re.search(r',\d{3}(?:\D|$)', p) and "." not in p:
        p = p.replace(",", "")
    else:
        #otherwise treat as european separator
        p = p.replace(",", ".")
        
    try:
        #keep decimal point and number
        digits = re.sub(r"[^\d.]", "", p)
        
        if digits.count(".") > 1:
            parts = digits.split(".")
            digits = parts[0] + "." + "".join(parts[1:])
            
        return float(digits) if digits else float("inf")
    except (ValueError, TypeError):
        return float("inf")
    
def answer_price_inquiry(question: str, local_index: list[dict]) -> str:
    q_lower = question.lower()
    addresses = sorted([e.get("Adresse", "") for e in local_index if e.get("Adresse")], key=len, reverse=True)
    matched_addr = next((a for a in addresses if a.lower() in q_lower and len(a) > 5), None)

    if matched_addr:
        results = [e for e in local_index if e.get("Adresse") == matched_addr]
        lines = [f"**Informations récupérées pour : {matched_addr}**\n"]
        for i, entry in enumerate(results, 1):
            lines.append(format_local(entry, i))
        return "\n".join(lines)

    type_filter = _parse_type_filter(question)
    top_n       = _parse_top_n(question, default=len(local_index))
    filtered = [e for e in local_index if e.get("Prix")]
    
    if type_filter:
        filtered = [e for e in filtered if any(t in str(e.get("Type", "")).lower() for t in type_filter)]
    if not filtered: return "Aucun local avec prix disponible dans le rayon de recherche."

    filtered.sort(key=lambda e: _parse_price(e.get("Prix", "")))
    results = filtered[:top_n]
    type_label = f" ({', '.join(type_filter)})" if type_filter else ""
    lines = [f"**{len(results)} local(aux){type_label} — classés par prix :**\n"]
    
    for i, entry in enumerate(results, 1):
        lines.append(format_local(entry, i))
        lines.append("")
    return "\n".join(lines)


#QUALITATIVE
def build_qualitative_chain(qualitative_str: str, unstructured_docs: list[Document]):
    print("\n[BUILD] Preparing qualitative LLM chain...")
    # NOTE: Set streaming=True in the LLM configuration
    llm        = ChatOllama(model=CONFIG["LLM_MODEL"], temperature=CONFIG["LLM_TEMPERATURE"], streaming=True)
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

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a territorial analysis assistant for Quebec municipalities.

RULES:
1. Answer ONLY from the context below. Never invent data.
2. NEVER guess, estimate, or hallucinate specific prices, rents, or property addresses. You do not have this data.
3. If the user asks for a price or address, say exactly: "Je n'ai pas accès aux prix individuels dans cette vue. Veuillez spécifier l'adresse exacte pour activer le moteur de recherche."
4. If the general answer is absent, say exactly: "Information non disponible dans le contexte."
5. Respond in the same language as the question.

[TERRITORIAL CONTEXT]
{structured_data}

[RETRIEVED NOTES]
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    prompt = prompt.partial(structured_data=qualitative_str)

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt),
    )
    print("[BUILD] Qualitative chain ready.\n")
    return chain


#MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Territoire Intelligence RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--radius", type=int,  default=5000)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    validate_env()
    mongo_client = get_mongo_client()
    db           = mongo_client[CONFIG["DB_NAME"]]

    print(f"\nTerritoire Intelligence RAG — radius={args.radius}m")
    print("Commands: 'exit' to quit | 'change location' to switch area\n")

    try:
        while True:
            location_input = input("Enter an address or city (Quebec): ").strip()
            if not location_input: continue
            if location_input.lower() in ("exit", "quit", "q"): break

            lon, lat = geocode_address(location_input)
            if lon is None:
                print("Could not resolve location. Try a more specific address.\n")
                continue

            city_hint = f"{location_input}, Quebec, Canada"

            print(f"\nFetching context for '{location_input}'...")
            t0 = time.perf_counter()

            qualitative_str, notes_docs, raw_locals, demographics, naics_lines = (
                retrieve_territorial_context(db, lon, lat, args.radius, debug=args.debug)
            )
            local_index = build_structured_index(raw_locals)
            llm_chain   = build_qualitative_chain(qualitative_str, notes_docs)
            
            # Initialize dynamic context window for the session
            chat_history = []

            elapsed = time.perf_counter() - t0
            print(f"{len(local_index)} locals indexed | ready in {elapsed:.1f}s")
            print(f" Session: '{location_input}' | radius: {args.radius}m")
            print("    Type your question or 'change location'.\n")

            while True:
                try:
                    user_q = input(f"Question ({location_input}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    return

                if not user_q: continue
                if user_q.lower() in ("exit", "quit", "q"): return
                if user_q.lower() == "change location":
                    print("\nChanging location...\n")
                    break

                t0     = time.perf_counter()
                intent = classify_question(user_q)

                if args.debug:
                    print(f"   [ROUTER] intent={intent}")

                #PATH 1: Spatial ranking
                if intent == "spatial":
                    answer  = answer_spatial(user_q, local_index, raw_locals, lon, lat, city_hint)
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.1f}s — spatial):\n{answer}\n")
                    
                    chat_history.extend([HumanMessage(content=user_q), AIMessage(content=answer)])

                #PATH 2: Filter / listing
                elif intent == "filter":
                    answer  = answer_filter(user_q, local_index)
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.3f}s — filter):\n{answer}\n")
                    chat_history.extend([HumanMessage(content=user_q), AIMessage(content=answer)])

                #PATH 2b: Price Inquiry (Exact Lookup or Sort)
                elif intent == "price_inquiry":
                    answer  = answer_price_inquiry(user_q, local_index)
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.3f}s — price/lookup):\n{answer}\n")
                    chat_history.extend([HumanMessage(content=user_q), AIMessage(content=answer)])

                #PATH 3: LLM — qualitative (STREAMING INTEGRATED)
                else:
                    try:
                        print(f"\nRÉPONSE (LLM en cours) :\n", end="")
                        full_answer = ""
                        
                       
                        for chunk in llm_chain.stream({"input": user_q, "chat_history": chat_history}):
                            if "answer" in chunk:
                                text_chunk = chunk["answer"]
                                print(text_chunk, end="", flush=True)
                                full_answer += text_chunk
                        
                        elapsed = time.perf_counter() - t0
                        print(f"\n[Temps de réponse: {elapsed:.1f}s]\n")
                        
                        chat_history.extend([HumanMessage(content=user_q), AIMessage(content=full_answer)])
                    except Exception as e:
                        print(f"\n[ERROR] LLM inference failed: {e}\n")

                #dynamic context window integration
                if len(chat_history) > CONFIG["CHAT_MEMORY_LIMIT"]:
                    
                    chat_history = chat_history[-CONFIG["CHAT_MEMORY_LIMIT"]:]

    finally:
        mongo_client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    main()