"""
baseline_rag_v2.py — Territoire Intelligence RAG
==================================================
Run:  python baseline_rag_v2.py [--radius M] [--debug]

KEY ARCHITECTURE DECISIONS
───────────────────────────
• Context is fetched once per location (this is to become the polygon drawn by the user), then reused for all subsequent questions.
  No re-embedding, no chain rebuild between questions.

• Structured data (addresses, prices, demographics) is not sent through
  the vector store. It goes directly into the prompt as a formatted string
  so the LLM sees every row verbatim — this is to reduce hallucinations.

• The vector store is used for long free-text notes (> 50 chars).
  If there are no notes, a dummy store is created so LangChain doesn't crash,
  but it is never meaningfully retrieved from.

• The retriever k is set to min(len(chunks), 6) — not a hardcoded 20 that
  causes slow FAISS searches and irrelevant chunk returns.
"""

import os
import sys
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

#CHANGING ALL PARAMS IN ONE PLACE
CONFIG = {
    "DB_NAME":           "overture_maps",
    "MONGO_TIMEOUT_MS":  5000,
    "LOCAUX_LIMIT":      50,       
    "NAICS_LIMIT":       50,
    "EMBED_MODEL":       "nomic-embed-text",
    "LLM_MODEL":         "llama3.2:3b",
    "LLM_TEMPERATURE":   0,
    "NOTES_MIN_LENGTH":  50,       #notes shorter than this go in the prompt directly 
    "CHUNK_SIZE":        400,
    "CHUNK_OVERLAP":     40,
    "RETRIEVER_K":       6,        
    "GEOCODE_DELAY_S":   1.0,
}

NO_DATA_MSG = "Information non disponible dans le contexte."

#values that mean no data in MongoDB to avoid the model hallucinating 
_EMPTY = {None, "None", "", "N/A", "null", "n/a"}

def _field(label: str, value, suffix: str = "") -> str:
    """
    Return none only when no value returned in database so the LLM never sees
    a field it could hallucinate a value for.
    """
    if value in _EMPTY:
        return ""
    return f"- {label} : {value}{suffix}\n"

#CREATION OF ENVIRONMENT + CONNECTION TO MONGODB
def validate_env() -> None:
    if not os.getenv("MONGO_URI"):
        print("[STARTUP ERROR] MONGO_URI missing from .env")
        sys.exit(1)
    print(".env OK")

def get_mongo_client() -> MongoClient:
    return MongoClient(
        os.getenv("MONGO_URI", ""),
        serverSelectionTimeoutMS=CONFIG["MONGO_TIMEOUT_MS"],
    )

#geocoding to get adress: this can be temporary 
def geocode_address(address: str) -> tuple[float | None, float | None]:
    """Convert a free-text address to (lon, lat). Returns (None, None) on failure."""
    print(f"Geocoding: '{address}'...")
    try:
        time.sleep(CONFIG["GEOCODE_DELAY_S"])   #Nominatim rate-limit
        loc = Nominatim(user_agent="territoire_intelligence_rag_v2").geocode(address, timeout=10)
        if loc:
            print(f"({loc.longitude:.5f}, {loc.latitude:.5f})")
            return loc.longitude, loc.latitude
        print(f"Could not resolve '{address}'.")
        return None, None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Geocoder error: {e}")
        return None, None

#DATA RETRIEVAL
def retrieve_territorial_context(
    db, user_lon: float, user_lat: float, radius_meters: int, debug: bool = False
) -> tuple[str, list[Document]]:
    """
    Returns:
      structured_str— Markdown table of hard facts (demographics, all locals,
                        dominant sectors). Injected verbatim into the prompt.
      unstructured_docs— Long free-text notes only, for FAISS retrieval.

    """
    structured_parts: list[str] = []
    unstructured_docs: list[Document] = []

    geo_point = {"type": "Point", "coordinates": [user_lon, user_lat]}

    #1. Demographics
    print("Demographics...")
    try:
        city_data = db.donnees.find_one({
            "geometry": {"$geoIntersects": {"$geometry": geo_point}}
        })
        if city_data:
            # _field() drops any field that is None/N/A in this census zone
            demo = "### Données Socio-Démographiques\n"
            demo += _field("Population 2021",      city_data.get("Population_2021"))
            demo += _field("Population 2016",      city_data.get("Population_2016"))
            demo += _field("Logements privés",     city_data.get("Logements_prives_total"))
            demo += _field("Densité pop/km²",      city_data.get("Densite_pop_km2"))
            demo += _field("Superficie km²",       city_data.get("Superficie_km2"))
            demo += _field("Code géographique",    city_data.get("Geographie"))
            structured_parts.append(demo)
            print(f"Demographics loaded (pop {city_data.get('Population_2021', '?')}).")
        else:
            print("No demographic zone found at these coordinates.")
    except Exception as e:
        print(f"Demographics error: {e}")

    #2. Locaux vacants
    print(f"Vacant locals within {radius_meters}m...")
    locaux_query = {
        "geometry": {
            "$near": {
                "$geometry": geo_point,
                "$maxDistance": radius_meters,
            }
        }
    }
    try:
        found_items = list(db.locaux_vacants.find(locaux_query).limit(CONFIG["LOCAUX_LIMIT"]))
        print(f"{len(found_items)} locals found.")

        if found_items:
            structured_parts.append(f"\n### Locaux Vacants à Proximité ({len(found_items)} résultats)")
            for i, local in enumerate(found_items, 1):
                # _field() silently drops any field whose value is None/null/empty.
                # The LLM only sees fields that have real data thus cannot hallucinate
                # a price or contact that doesn't exist in MongoDB.
                adresse = local.get("adresse") or "Adresse inconnue"
                local_line = f"\n**Local #{i} — {adresse}**\n"
                local_line += _field("Ville",      local.get("ville"))
                local_line += _field("Type",       local.get("type_local"))
                local_line += _field("Secteur",    local.get("secteur"))
                local_line += _field("Prix",       local.get("prix"))
                local_line += _field("Superficie", local.get("superficie_pi2"), suffix=" pi²")
                local_line += _field("Contact",    local.get("contact"))
                local_line += _field("URL",        local.get("url_source"))
                #actif is a boolean — only show it when explicitly False (still listed = available)
                if local.get("actif") is False:
                    local_line += "- Statut : Inactif\n"
                structured_parts.append(local_line)

                #Long notes only go to FAISS (they don't contain exact numbers)
                notes = local.get("notes", "")
                if notes and len(str(notes)) > CONFIG["NOTES_MIN_LENGTH"]:
                    unstructured_docs.append(Document(
                        page_content=f"Notes pour le local au {adresse}: {notes}",
                        metadata={"type": "local_notes", "adresse": adresse},
                    ))
        else:
            structured_parts.append("\n### Locaux Vacants\nAucun local vacant trouvé dans ce rayon.")
    except Exception as e:
        print(f"Locaux error (check 2dsphere index): {e}")

    #3. NAICS — match business categories to NAICS labels by text => NEED TO FIX 
    print("Economic sectors...")
    try:
        nearby_businesses = list(
            db.quebec_businesses.find(locaux_query).limit(CONFIG["NAICS_LIMIT"])
        )
        if nearby_businesses:
            #Collect every non-empty category or type string from the businesses
            raw_categories: set[str] = set()
            for b in nearby_businesses:
                for field in ("categorie", "category", "type", "secteur", "type_entreprise"):
                    val = b.get(field)
                    if val and str(val).strip() not in _EMPTY:
                        raw_categories.add(str(val).strip().lower())

            if raw_categories:
                #Load all NAICS entries once, then match by label substring
                all_naics = list(db.naics.find({}))
                matched: list[dict] = []
                for sector in all_naics:
                    label = str(sector.get("label", "")).lower()
                    description = str(sector.get("description", "")).lower()
                    #A NAICS sector matches if any business category appears in its label or description
                    if any(cat in label or cat in description or label in cat
                           for cat in raw_categories):
                        matched.append(sector)

                if matched:
                    structured_parts.append("\n### Secteurs Économiques Présents dans la Zone")
                    for s in matched:
                        line = f"- {s.get('label', 'N/A')}"
                        if s.get("code") not in _EMPTY:
                            line += f" (Code NAICS {s.get('code')})"
                        if s.get("description") not in _EMPTY:
                            line += f" — {s.get('description')}"
                        structured_parts.append(line)
                    print(f"{len(matched)} NAICS sector(s) matched from {len(raw_categories)} business category/type(s).")
                else:
                    # Fallback: just list the raw business categories we found
                    structured_parts.append("\n### Catégories d'Entreprises Présentes dans la Zone")
                    for cat in sorted(raw_categories):
                        structured_parts.append(f"- {cat.title()}")
                    print(f" No NAICS label match; listed {len(raw_categories)} raw business category/type(s) as fallback.")
            else:
                print("Businesses found but no category/type field present on documents.")
        else:
            print("No businesses found in radius (collection may not exist yet).")
    except Exception as e:
        print(f" NAICS error: {e}")

    structured_str = (
        "\n".join(structured_parts)
        if structured_parts
        else "Aucune donnée structurée disponible pour cette zone."
    )

    if debug:
        print("\n─── STRUCTURED DATA INJECTED INTO PROMPT ───")
        print(structured_str)
        print("────────────────────────────────────────────\n")

    return structured_str, unstructured_docs


#CHAIN BUILDER
def build_chain(structured_data_str: str, unstructured_docs: list[Document]):
    """
    Build the RAG chain.

    - structured_data_str is baked into the prompt via prompt.partial().
      The LLM always sees every local, every price, every address.
    - FAISS is only used for long free-text notes.
    """
    print("\n[BUILD] Preparing AI model")
    llm        = ChatOllama(model=CONFIG["LLM_MODEL"], temperature=CONFIG["LLM_TEMPERATURE"])
    embeddings = OllamaEmbeddings(model=CONFIG["EMBED_MODEL"])

    #vector store for notes only
    if unstructured_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
        )
        splits = splitter.split_documents(unstructured_docs)
        vector_store = FAISS.from_documents(splits, embeddings)
        k = min(len(splits), CONFIG["RETRIEVER_K"])
        print(f"[BUILD] Notes vector store ready ({len(splits)} chunk(s), k={k}).")
    else:
        #dummy store if no long notes
        vector_store = FAISS.from_texts(["Aucune note textuelle disponible."], embeddings)
        k = 1
        print("[BUILD] No free-text notes — using dummy vector store.")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    #PROMPY
    # {structured_data} is baked in via .partial() — never retrieved from FAISS.
    # {context}         is filled by the retriever (notes only).
    # {input}           is the user's question.
    prompt = ChatPromptTemplate.from_template("""You are a precise territorial analysis \
assistant for Quebec municipalities.

RULES — no exceptions:
1. Answer ONLY using facts explicitly written in [DONNÉES STRUCTURÉES] below.
2. NEVER add, infer, or invent any field (address, price, type, contact, size) \
that is not explicitly listed under a local's entry.
3. If a field is not listed for a local, it does not exist — do not mention it.
4. If the answer cannot be found at all, respond exactly: \
"Information non disponible dans le contexte."
5. Respond in the same language as the user's question (French or English).
6. When listing locals, list EVERY one from [DONNÉES STRUCTURÉES] that matches \
the question — do not stop early.

=========================================
[DONNÉES STRUCTURÉES — SOURCE DE VÉRITÉ]
{structured_data}
=========================================
[NOTES QUALITATIVES SUPPLÉMENTAIRES]
{context}
=========================================

Question: {input}
""")

    # Bake structured data into the prompt at chain-build time
    prompt = prompt.partial(structured_data=structured_data_str)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("[BUILD] Chain ready.\n")
    return chain


#MAIN
def main():
    parser = argparse.ArgumentParser(description="Territoire Intelligence RAG v2")
    parser.add_argument("--radius", type=int,  default=5000,
                        help="Search radius in metres (default: 5000)")
    parser.add_argument("--debug",  action="store_true",
                        help="Print full structured context before Q&A")
    args = parser.parse_args()

    validate_env()
    mongo_client = get_mongo_client()
    db = mongo_client[CONFIG["DB_NAME"]]

    print(f"\nTerritoire Intelligence RAG — radius={args.radius}m")
    print("Type 'exit' to quit, 'change location' inside a session to switch area.\n")

    try:
        while True:
            #location input loop
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

            #build context chain
            print(f"\nFetching territorial context for '{location_input}'...")
            t0 = time.perf_counter()
            structured_str, notes_docs = retrieve_territorial_context(
                db, lon, lat, args.radius, debug=args.debug
            )
            chain = build_chain(structured_str, notes_docs)
            elapsed = time.perf_counter() - t0
            print(f"Context + chain ready in {elapsed:.1f}s")
            print(f"Ask all your questions about '{location_input}' below.")
            print("    Type 'change location' to switch area.\n")

            #Q&A loop
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

                t0 = time.perf_counter()
                try:
                    result = chain.invoke({"input": user_q})
                    answer = result.get("answer", "").strip() or NO_DATA_MSG
                    elapsed = time.perf_counter() - t0
                    print(f"\nRÉPONSE ({elapsed:.1f}s):\n{answer}\n")
                except Exception as e:
                    print(f"\n[ERROR] Inference failed: {e}\n")

    finally:
        mongo_client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    main()