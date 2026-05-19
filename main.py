import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS 
# Swapped to local Ollama modules
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def main():
    # --- STEP 1: LOAD ---
    print("📂 Loading documents from ./data...")
    if not os.path.exists("./data") or not os.listdir("./data"):
        print("❌ Error: No .txt files found in the /data folder.")
        return
    
    loader = DirectoryLoader("./data", glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents.")

    # --- STEP 2: CHUNK ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")

    # --- STEP 3: LOCAL EMBED ---
    print("🧠 Generating local embeddings using Llama 3.2... (No cloud APIs!)")
    # This instructs your computer to vectorize the text locally
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✅ Local vector index created successfully!")
    except Exception as e:
        print(f"❌ EMBEDDING ERROR: {e}")
        return

    # --- STEP 4: LOCAL LLM CHAIN ---
    # Using your locally downloaded model
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context below.
    Context: {context}
    Question: {input}
    """)

    chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 2}), 
        create_stuff_documents_chain(llm, prompt)
    )

    # --- STEP 5: RUN ---
    print("\n🚀 LOCAL RAG SYSTEM ONLINE (No limits!)")
    while True:
        user_q = input("\nQuestion (or 'exit'): ")
        if user_q.lower() == 'exit': break
        
        print("🔍 Searching local files and generating response...")
        result = chain.invoke({"input": user_q})
        print(f"\nAI ANSWER: {result['answer']}")

if __name__ == "__main__":
    main()