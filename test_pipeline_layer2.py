import os
import json
import time
import re
from typing import List, Dict
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.metrics import ndcg_score
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load environment to get API keys 
load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Critical error: ANTHROPIC_API_KEY is missing from the .env file")
    exit(1)
#ensure format of the API key is correct
API_KEY = API_KEY.strip()
client = Anthropic(api_key=API_KEY)
#judge model - SUBJECT TO UPDATE THROUGH THE YEARS 
JUDGE_MODEL = "claude-haiku-4-5-20251001"

#Connecting to MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "overture_maps"
RETRIEVER_K = 3  
VERBOSE_DEBUG = True  #see the QA matches 

#parameters tested 
CHUNKING_GRID = [
    {"chunk_size": 200, "chunk_overlap": 20},
    {"chunk_size": 400, "chunk_overlap": 40},   
    {"chunk_size": 400, "chunk_overlap": 100},  
    {"chunk_size": 800, "chunk_overlap": 100},  
]

#EXTRACTION OF THE DATA 
def load_qualitative_queries(filepath="test_dataset.json") -> list[str]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        qual_questions = [
            item["question"] for item in data 
            if item.get("expected_path") == "qualitative" and "question" in item
        ]
        print(f"Loaded {len(qual_questions)} Qualitative validation targets from {filepath}")
        return qual_questions
    except FileNotFoundError:
        print(f"[ERROR] {filepath} missing.")
        return []

def fetch_notes_corpus(limit=300) -> list[Document]:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[DB_NAME]
        docs = []
        for local in db.locaux_vacants.find({"notes": {"$exists": True, "$ne": ""}}).limit(limit):
            addr = local.get("adresse", "Inconnue")
            docs.append(Document(page_content=f"Local {addr}: {local['notes']}", metadata={"src": "local"}))
        for biz in db.quebec_businesses.find({"description": {"$exists": True, "$ne": ""}}).limit(limit):
            name = biz.get("nom", biz.get("name", "Inconnu"))
            docs.append(Document(page_content=f"Entreprise {name}: {biz['description']}", metadata={"src": "biz"}))
        mongo_client.close()
        print(f"Compiled text corpus of {len(docs)} foundational documents from MongoDB.")
        return docs
    except Exception as e:
        print(f"[DATABASE ERROR] Could not compile text corpus: {e}")
        return []

#CALCULATE THE METRICS 
def calculate_mrr(relevance_scores: list[int]) -> float:
    for rank, score in enumerate(relevance_scores, 1):
        if score >= 2:  #first document that is useful passes the benchmark (chosen 2 => discuss if sufficient)
            return 1.0 / rank
    return 0.0

def calculate_precision_at_k(relevance_scores: list[int], k: int) -> float:
    top_k_scores = relevance_scores[:k]
    relevant_count = sum(1 for score in top_k_scores if score >= 2)
    return relevant_count / k if k > 0 else 0.0

def calculate_ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    if not relevance_scores or sum(relevance_scores) == 0:
        return 0.0
    true_relevance = np.asarray([relevance_scores])
    ideal_order = np.asarray([[len(relevance_scores) - i for i in range(len(relevance_scores))]])
    return float(ndcg_score(true_relevance, ideal_order, k=k))

#LLM-AS-A-JUDGE + few shot anchoring
def judge_chunk_relevance(question: str, chunk: str, max_retries: int = 3) -> int:
    """
    Evaluates chunk alignment with explicit few-shot guidelines, strict defensive
    parsing, and exponential retry backoffs to isolate structural scoring drift.
    """
    prompt = f"""You are a strict quality assurance evaluator scoring a semantic information retrieval database.
    Rate how relevant the following retrieved document chunk is to answering the user's specific query.

    USER QUESTION: "{question}"
    RETRIEVED CHUNK CONTENT: "{chunk}"

    SCORING MATRIX:
    0 = Completely irrelevant. The topic or geographical context is completely detached.
    1 = Marginally relevant. Contains general matching keywords, but offers zero utility toward answering the query.
    2 = Relevant. Provides partial facts, demographic hints, or context that helps build a complete answer.
    3 = Highly relevant. Contains direct, explicit facts that answer the user's prompt directly and perfectly.

    FEW-SHOT ANCHORING EXAMPLES:
    - Q: "Quel local est adapté pour de la logistique près de l'autoroute?" 
      Chunk: "Local #4: Entrepôt industriel avec quais de chargement à 2 min de l'A-20." -> Score: 3

    CRITICAL RULES:
    - Output your final verdict in a valid JSON block containing exactly one key: "score".
    - Do not include explanations, formatting wrappers, or additional text.
    
    EXPECTED RESPONSE FORMAT:
    {{"score": 2}}
    """

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=25,
                temperature=0,  
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = response.content[0].text.strip()
            
            #uses question from question dataset
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                parsed_payload = json.loads(json_match.group(0))
                extracted_score = int(parsed_payload.get("score", 0))
                if extracted_score in [0, 1, 2, 3]:
                    return extracted_score
                    
            #fallback
            digits = re.findall(r'\d', raw_text)
            if digits:
                score = int(digits[0])
                if score in [0, 1, 2, 3]:
                    return score

        except (json.JSONDecodeError, ValueError, Exception) as error:
            wait_time = 2 ** attempt
            print(f"   [API WARNING] Attempt {attempt+1} failed ({error}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"[API CRITICAL FAIL] Falling back safely to 0 score for stability.")
    return 0 

#MAIN EXECUTION 
def execute_layer2_pipeline():
    print("=" * 70)
    print("      LAYER 2 INITIALIZATION: FAISS RETRIEVAL QUALITY GRID")
    print("=" * 70)

    questions = load_qualitative_queries("test_dataset.json")
    corpus = fetch_notes_corpus()

    if not questions or not corpus:
        print("[ABORT] Missing validation assets. Execution halted.")
        return

    embedder = OllamaEmbeddings(model="nomic-embed-text")
    
    for run in CHUNKING_GRID:
        size = run["chunk_size"]
        overlap = run["chunk_overlap"]
        print(f"\nEvaluating Hyperparameters: chunk_size={size} | chunk_overlap={overlap}")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        fragmented_chunks = splitter.split_documents(corpus)
        
        print(f"   Building volatile FAISS space with {len(fragmented_chunks)} elements...")
        db_index = FAISS.from_documents(fragmented_chunks, embedder)
        retriever = db_index.as_retriever(search_kwargs={"k": RETRIEVER_K})
        
        run_mrr, run_precision, run_ndcg = [], [], []
        
        for q in questions:
            hits = retriever.invoke(q)
            query_scores = []
            
            if VERBOSE_DEBUG:
                print(f"\n   [QA TRACKING] Query: '{q}'")
                
            for idx, doc in enumerate(hits, 1):
                score = judge_chunk_relevance(q, doc.page_content)
                query_scores.append(score)
                
                if VERBOSE_DEBUG:
                    preview = doc.page_content.replace('\n', ' ')[:65]
                    print(f"      ↳ Hit #{idx} | Score: {score} | Content Preview: {preview}...")
                    
                time.sleep(0.1)  
            
            run_mrr.append(calculate_mrr(query_scores))
            run_precision.append(calculate_precision_at_k(query_scores, k=RETRIEVER_K))
            run_ndcg.append(calculate_ndcg_at_k(query_scores, k=RETRIEVER_K))

        print(f"\n    SUMMARY METRICS FOR CONFIG [{size}/{overlap}]:")
        print(f"      ↳ Mean Reciprocal Rank (MRR)   : {np.mean(run_mrr):.3f}")
        print(f"      ↳ Precision@{RETRIEVER_K}              : {np.mean(run_precision):.3f}")
        print(f"      ↳ NDCG@{RETRIEVER_K}                   : {np.mean(run_ndcg):.3f}")
        print("-" * 70)

if __name__ == "__main__":
    execute_layer2_pipeline()