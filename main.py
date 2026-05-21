import os
import urllib.parse
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def build_mongo_uri() -> str:
    raw_uri = os.getenv("MONGO_URI", "")
    if not raw_uri:
        raise ValueError("MONGO_URI is not set in .env file.")

    prefix = "mongodb+srv://"
    if not raw_uri.startswith(prefix):
        return raw_uri  

    base = raw_uri[len(prefix):]
    userinfo, host_and_options = base.split("@", 1)
    username, password = userinfo.split(":", 1)
    encoded_password = urllib.parse.quote_plus(password)
    return f"{prefix}{username}:{encoded_password}@{host_and_options}"


def retrieve_territorial_context(
    user_lon: float, user_lat: float, radius_meters: int
) -> list[Document]:
    #Query MongoDB and return docs for RAG context 
    client = MongoClient(build_mongo_uri())
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
        print(f"  → {len(found_items)} vacant local(s) found.")
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
    embeddings = OllamaEmbeddings(model="llama3.2:1b") #PEUT SWITCHER A MEILLEUR MODELE PRETENTRAINE
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
                "k": 6,
                "filter": {"type": {"$in": ["local", "city", "naics"]}} #Diverse doc retrieval (see if best)
            }
        ),
        combine_docs_chain=combine_docs_chain,
    )
    return chain


def main():
    #to load context
    docs = retrieve_territorial_context(
        user_lon=-72.9427, user_lat=45.6255, radius_meters=15000 #hard coded for testing but this would be user input 
    )

    if not docs:
        print("\n Aucun document trouvé dans MongoDB. Vérifiez votre connexion et vos données.")
        return

    print(f"\n {len(docs)} document(s) chargés.")

    #for testing phase printing the context 
    print("\n CONTEXTE ENVOYÉ AU MODÈLE ")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] ({doc.metadata.get('type')}) {doc.page_content}")
    print("---------------------------------\n")


    chain = build_chain(docs)

    #Question loop 
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