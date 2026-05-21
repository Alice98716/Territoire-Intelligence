import os
import urllib.parse
import argparse
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_mongo_client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "")
    if not uri:
        raise ValueError("MONGO_URI is not set in .env file.")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def retrieve_territorial_context(
    user_lon: float, user_lat: float, radius_meters: int
) -> list[Document]:
    #Query MongoDB and return docs for RAG context 
    client = get_mongo_client()
    db = client.overture_maps
    rag_documents: list[Document] = []

    #Module 1: Local vacant
    print(f"Searching vacant locals near ({user_lon}, {user_lat}) within {radius_meters}m...")
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
        print(f" {len(found_items)} vacant local(s) found.")
        for local in found_items:
            text = (
                f"[IMMOBILIER VACANT - {local.get('type_local')}] "
                f"Located at {local.get('adresse')}. "
                f"This property is in a sector with {local.get('notes', 'no additional details')}. "
                f"Market price: {local.get('prix')}."
            )
            rag_documents.append(Document(page_content=text, metadata={"type": "local"}))
    except Exception as e:
        print(f"Erreur locaux (check 2dsphere index): {e}")

    #Module 2: Demographie des villes
    try:
        # Instead of: db.cities.find_one({"name": "Saint-Hyacinthe"}), plus general
        city_data = db.cities.find_one({
            "geometry": {
                "$geoIntersects": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [user_lon, user_lat]
                    }
                }
            }
        })
        if city_data:
            text = (
                f"[VILLE: {city_data.get('name', 'N/A')}] "
                f"Population 2021: {city_data.get('Population_2021', 'N/A')}"
            )
            rag_documents.append(Document(page_content=text, metadata={"type": "city"}))
            print(f"City data loaded.")
        else:
            print("No city data found.")
    except Exception as e:
        print(f"Erreur villes: {e}")

    #Module 3: NAICS
    try:
        sectors = list(db.naics.find({}).limit(10))
        for sector in sectors:
            label = sector.get("label", "Unknown sector")
            rag_documents.append(
                Document(page_content=f"[NAICS] {label}", metadata={"type": "naics"})
            )
        print(f"{len(sectors)} NAICS sector(s) loaded.")
    except Exception as e:
        print(f"Erreur NAICS: {e}")

    return rag_documents


def build_chain(docs: list[Document]):
    """Build the FAISS vector store and RAG chain from documents."""
    print("\nBuilding embeddings and vector store")
    embeddings = OllamaEmbeddings(model="nomic-embed-text") #swtiched to specialized embedding model 
    vector_store = FAISS.from_documents(docs, embeddings)

    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    #WORK ON PROMPT FOR BETTER RESULTS
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
                "k": 6
            }
        ),
        combine_docs_chain=combine_docs_chain,
    )
    return chain


def main():
    parser = argparse.ArgumentParser(description="Territoire Intelligence RAG Baseline")
    parser.add_argument("--lon", type=float, default=-72.9427, help="User Longitude") #hard coded but should be user's input 
    parser.add_argument("--lat", type=float, default=45.6255, help="User Latitude")
    parser.add_argument("--radius", type=int, default=15000, help="Search radius in meters")
    args = parser.parse_args()

    docs = retrieve_territorial_context(
        user_lon=args.lon, user_lat=args.lat, radius_meters=args.radius
    )

    if not docs:
        print("\nAucun document trouvé dans MongoDB. Vérifiez votre connexion et vos données.")
        return

    print(f"\n{len(docs)} document(s) chargés.")

    print("\nCONTEXTE ENVOYÉ AU MODÈLE")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] ({doc.metadata.get('type')}) {doc.page_content}")
    print("---------------------------------\n")

    chain = build_chain(docs)

    print("Prêt. Posez vos questions (tapez 'exit' pour quitter).\n")
    while True:
        user_q = input("Question : ").strip()
        if not user_q:
            continue
        if user_q.lower() in ("exit", "quit", "q"):
            print("Au revoir!")
            break

        try:
            result = chain.invoke({"input": user_q})
            answer = result.get("answer", "").strip()
            if not answer:
                answer = "Information non disponible dans le contexte."
            print(f"\nRÉPONSE :\n{answer}\n")
        except Exception as e:
            print(f"\nErreur lors de l'inférence : {e}\n")


if __name__ == "__main__":
    main()