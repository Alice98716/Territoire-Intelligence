import os
import sys
import urllib.parse
import geopy
import argparse
from pymongo import MongoClient
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def validate_env() -> None:
    """Check required environment variables exist before doing any work."""
    missing = []
    if not os.getenv("MONGO_URI"):
        missing.append("MONGO_URI")
    if missing:
        print(f"[STARTUP ERROR] Missing required .env variables: {', '.join(missing)}")
        print("Create a .env file with: MONGO_URI=mongodb+srv://user:pass@host/...")
        sys.exit(1)
    print("Environment variables validated.")


def get_mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "")
    if not uri:
        raise ValueError("MONGO_URI is not set in .env file.")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)

def geocode_address(address: str) -> tuple[float | None, float | None]:
    """Convert an address or municipality name into longitude and latitude coordinates."""
    print(f"Geocoding location: '{address}'...")
    try:
        # Always use a unique user_agent to comply with Nominatim's usage policy
        geolocator = Nominatim(user_agent="territoire_intelligence_rag")
        location = geolocator.geocode(address)
        if location:
            return location.longitude, location.latitude
        else:
            print(f"Location '{address}' could not be resolved.")
            return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None
    

def retrieve_territorial_context(
    user_lon: float, user_lat: float, radius_meters: int
) -> list[Document]:
    """Query MongoDB and return LangChain Documents for RAG context."""
    client = get_mongo_client()
    rag_documents: list[Document] = []
 
    try:
        db = client.overture_maps
 
        # Module 1: Locaux vacants (geospatial) 
        print(f"Searching vacant locals near ({user_lon}, {user_lat}) within {radius_meters}m")
        try:
            locaux_query = {
                "geometry": {
                    "$near": {
                        "$geometry": {"type": "Point", "coordinates": [user_lon, user_lat]},
                        "$maxDistance": radius_meters,
                    }
                }
            }
            found_items = list(db.locaux_vacants.find(locaux_query).limit(10))
            print(f"{len(found_items)} vacant local(s) found.")
            for local in found_items:
                #added fallback string if nothing is found
                text = (
                    f"[IMMOBILIER VACANT - {local.get('type_local', 'type inconnu')}] "
                    f"Located at {local.get('adresse', 'adresse inconnue')}. "
                    f"Additional notes: {local.get('notes', 'aucune information complémentaire')}. "
                    f"Market price: {local.get('prix', 'prix non disponible')}."
                )
                rag_documents.append(Document(page_content=text, metadata={"type": "local"}))
        except Exception as e:
            print(f"Locaux error (check 2dsphere index on locaux_vacants.geometry): {e}")
 
        #Module 2: City demographics 
        try:
            print(f"  → Searching demographics data...")
            # Run the spatial intersection on 'donnees' where the geometry polygon lives
            city_data = db.donnees.find_one({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": {"type": "Point", "coordinates": [user_lon, user_lat]}
                    }
                }
            })

            if city_data:
                text = (
                    f"[DONNÉES SOCIO-DÉMOGRAPHIQUES] "
                    f"Code Géographie: {city_data.get('Geographie', 'N/A')} | "
                    f"Population 2021: {city_data.get('Population_2021', 'N/A')} | "
                    f"Âge médian: {city_data.get('age_median', 'N/A')} ans | "
                    f"Revenu médian: {city_data.get('revenu_median', 'N/A')} $ | "
                    f"Total logements privés: {city_data.get('Logements_prives_total', 'N/A')} | "
                    f"Superficie: {city_data.get('Superficie_km2', 'N/A')} km²"
                )
                rag_documents.append(Document(page_content=text, metadata={"type": "city"}))
                print(f"  → Demographics data loaded successfully.")
            else:
                print("  → No demographic data found for these coordinates in 'donnees'.")
        except Exception as e:
            print(f"City error (check 2dsphere index on donnees.geometry): {e}")
 
        #Module 3: NAICS sectors
        print("Searching NAICS sectors")
        try:
            sectors = list(db.naics.find({}).limit(10))
            for sector in sectors:
                # Pull every useful field that might exist in the collection
                text = (
                    f"[NAICS code: {sector.get('code', 'N/A')}] "
                    f"Sector: {sector.get('label', 'Unknown sector')} | "
                    f"Category: {sector.get('category', 'N/A')} | "
                    f"Description: {sector.get('description', 'N/A')}"
                )
                rag_documents.append(
                    Document(page_content=text, metadata={"type": "naics"})
                )
            print(f"{len(sectors)} NAICS sector(s) loaded.")
        except Exception as e:
            print(f"NAICS error: {e}")
 
    finally:
        #for security reasons close the connection once the search is finished 
        client.close()
        print("MongoDB connection closed.")
 
    return rag_documents


def build_chain(docs: list[Document]):
    """Build the FAISS vector store and RAG chain from documents."""
    print("\nBuilding embeddings and vector store")
    embeddings = OllamaEmbeddings(model="nomic-embed-text") #swtiched to specialized embedding model 
    vector_store = FAISS.from_documents(docs, embeddings)

    llm = ChatOllama(model="llama3.2:3b", temperature=0) #switched to ollama 3b (used to be 1b)

    #WORK ON PROMPT FOR BETTER RESULTS => could add formatting prompt expected
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a precise territorial analysis assistant for Quebec municipalities.

Rules:
- Answer ONLY using the provided context. Never invent data.
- If the answer is not in the context, respond exactly: "Information non disponible dans le contexte."
- Always respond in the same language as the user's question (French or English).
- Be concise and structured. Use bullet points for lists of items.
- When mentioning properties or businesses, include all available details (address, type, price).
- Do not speculate or add general knowledge beyond what is in the context.

Context:
{context}""",
        ),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 6 #add filter if later docs are not of known type (not probable)
            }
        ),
        combine_docs_chain=combine_docs_chain,
    )
    return chain


def main():
    validate_env()
    
    parser = argparse.ArgumentParser(description="Territoire Intelligence RAG Baseline")
    parser.add_argument("--radius", type=int, default=15000, help="Search radius in meters")
    args = parser.parse_args()

    print("Type 'exit' at any time to turn off the platform.\n")

    while True:
        #get ocation dynamically => to be changed in the future
        location_input = input("Enter an address, neighborhood, or city name in Quebec: ").strip()
        if not location_input:
            continue
        if location_input.lower() in ("exit", "quit", "q"):
            print("Au revoir!")
            break

        #Geocode text to coordinates 
        lon, lat = geocode_address(location_input)
        if lon is None or lat is None:
            print("Could not establish spatial context for this location. Please try another address.\n")
            continue

        # Step 3: Query MongoDB live for this explicit point location
        docs = retrieve_territorial_context(user_lon=lon, user_lat=lat, radius_meters=args.radius)
        if not docs:
            print("No data chunks pulled from MongoDB for this sector. Moving to next search.\n")
            continue

        print(f"\n Context constructed with {len(docs)} live documents.")
        
        # Step 4: Build transient FAISS database matching this sector
        chain = build_chain(docs)

        # Step 5: Question loop specifically tied to this location context
        print(f"\nContext locked onto: {location_input}. Ask your territory questions below.")
        while True:
            user_q = input(f"Question ({location_input}) : ").strip()
            if not user_q:
                continue
            if user_q.lower() in ("exit", "quit", "q"):
                print("Exiting application...")
                return
            if user_q.lower() == "change location":
                print("\nResetting location scope...\n")
                break

            try:
                result = chain.invoke({"input": user_q})
                answer = result.get("answer", "").strip()
                if not answer:
                    answer = "Information non disponible dans le contexte."
                print(f"\nRÉPONSE :\n{answer}\n")
            except Exception as e:
                print(f"\nInference error occurred: {e}\n")


if __name__ == "__main__":
    main()
