import os
from geopy.distance import geodesic
from langchain_community.vectorstores import FAISS 
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_geo_data():
    """Simulating a geospatial knowledge base with coordinates."""
    return [
        Document(
            page_content="The Old Port of Montreal features historic buildings, a Ferris wheel, and extensive walking paths along the Saint Lawrence River.",
            metadata={"name": "Old Port", "lat": 45.5048, "lon": -73.5492}
        ),
        Document(
            page_content="Mount Royal Park is a vast green space offering panoramic city views, hiking trails, and the iconic Beaver Lake.",
            metadata={"name": "Mount Royal", "lat": 45.5041, "lon": -73.5875}
        ),
        Document(
            page_content="The Parliament Buildings in Quebec City feature stunning architecture, the Fontaine de Tourny, and rich political history.",
            metadata={"name": "Parliament Buildings", "lat": 46.8083, "lon": -71.2141}
        )
    ]

def main():
    print("🧠 Initializing local embeddings engine...")
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    
    # 1. Create the vector store using our structured geo documents
    raw_docs = get_geo_data()
    vector_store = FAISS.from_documents(raw_docs, embeddings)
    print("✅ Geospatial Vector store initialized successfully!")

    # 2. Simulate user location (e.g., Downtown Montreal)
    user_location = (45.5017, -73.5673) 
    max_distance_km = 5.0 # We only want places within 5km

    # --- ADVANCED GEOSPATIAL FILTER ---
    print(f"\n🌍 Filtering database assets within {max_distance_km}km of user location...")
    
    # We pull ALL documents from FAISS to apply our geometric calculation
    all_docs = vector_store.similarity_search("", k=100)
    valid_geo_docs = []

    for doc in all_docs:
        doc_coords = (doc.metadata["lat"], doc.metadata["lon"])
        # Calculate real-world distance on the Earth's curvature
        distance = geodesic(user_location, doc_coords).kilometers
        
        if distance <= max_distance_km:
            # Inject the calculated distance directly into the text context for the LLM!
            doc.page_content += f" (Distance from user: {distance:.2f} km)"
            valid_geo_docs.append(doc)

    print(f"📌 Found {len(valid_geo_docs)} relevant geographic locations within your radius.")

    # 3. Create a temporary vector store containing ONLY local assets
    local_vector_store = FAISS.from_documents(valid_geo_docs, embeddings)

    # 4. Initialize Local LLM Chain
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a geospatial analysis assistant. Answer the question using ONLY the geographically filtered context provided below.
    
    Context: {context}
    Question: {input}
    """)

    chain = create_retrieval_chain(
        local_vector_store.as_retriever(search_kwargs={"k": 2}), 
        create_stuff_documents_chain(llm, prompt)
    )

    print("\n🚀 GEOSPATIAL RAG READY")
    while True:
        user_q = input("\nWhat would you like to analyze? (or 'exit'): ")
        if user_q.lower() == 'exit': break
        
        result = chain.invoke({"input": user_q})
        print(f"\nAI ANALYSIS: {result['answer']}")

if __name__ == "__main__":
    main()