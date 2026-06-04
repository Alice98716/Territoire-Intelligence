import os
import json
import time
import re
from typing import Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not API_KEY:
    print("critical error: ANTHROPIC_API_KEY is missing.")
    exit(1)

client = Anthropic(api_key=API_KEY)
JUDGE_MODEL = "claude-haiku-4-5-20251001"

#1. AQS Weights (Layer 5.1)
WEIGHTS = {
    "faithfulness": 0.35,
    "relevance": 0.30,
    "router": 0.20,
    "retrieval": 0.15
}

#2. JUDGE FUNCTIONS
def judge_e2e_quality(question: str, context: str, answer: str) -> Dict[str, float]:
    """Uses Claude to return both Faithfulness (0.0-1.0) and Relevance (1-5)."""
    prompt = f"""Evaluate this RAG system transaction.
    Context: "{context}"
    Question: "{question}"
    Answer: "{answer}"
    
    1. Score 'relevance' from 1 to 5 (5 is perfect).
    2. Extract factual claims from the Answer. Calculate 'faithfulness' as (Supported Claims / Total Claims). 
       If no facts are claimed (e.g. standard fallback phrase), faithfulness = 1.0. If hallucinations exist, score < 1.0.
       
    Output ONLY valid JSON: {{"relevance": 4, "faithfulness": 0.5}}
    """
    try:
        res = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        json_match = re.search(r'\{.*\}', res.content[0].text.strip(), re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"[JUDGE ERROR] {e}")
        
    return {"relevance": 1, "faithfulness": 0.0}

#3. LAYER 5 BLACK-BOX EVALUATOR CLASS 
class RAGEvaluatorE2E:
    def __init__(self):
        self.latency_logs = {"spatial": [], "filter": [], "price_inquiry": [], "qualitative": []}
        self.taxonomy = {
            "ROUTER_ERROR": 0, "GEOCODE_FAIL": 0, "EMPTY_RETRIEVAL": 0,
            "HALLUCINATION": 0, "RULE_VIOLATION": 0, 
            "FALLBACK_APPROPRIATE": 0, "FALLBACK_INAPPROPRIATE": 0
        }
        self.aqs_scores = []
        
    def calculate_aqs(self, faithfulness: float, relevance: int, router_acc: int, retrieval_score: float) -> float:
        # Normalize relevance from 1-5 to 0.0-1.0
        norm_rel = (relevance - 1) / 4.0 
        aqs = (
            (WEIGHTS["faithfulness"] * faithfulness) +
            (WEIGHTS["relevance"] * norm_rel) +
            (WEIGHTS["router"] * float(router_acc)) +
            (WEIGHTS["retrieval"] * retrieval_score)
        )
        return round(aqs, 3)

    def log_failure(self, category: str):
        self.taxonomy[category] += 1

    def evaluate_test_case(self, case: dict, pipeline_wrapper: Any):
        question = case["question"]
        expected_path = case["expected_path"]
        
        # 1. Execute Black-Box Pipeline (Timing it)
        start_time = time.time()
        try:
            predicted_path, answer, context, retrieval_score = pipeline_wrapper(question)
        except Exception as e:
            if "geocode" in str(e).lower():
                self.log_failure("GEOCODE_FAIL")
            else:
                print(f"[PIPELINE ERROR] {e}")
            return
            
        exec_time = time.time() - start_time
        
        # 2. Latency Profiling
        if predicted_path in self.latency_logs:
            self.latency_logs[predicted_path].append(exec_time)
        else:
            self.latency_logs[predicted_path] = [exec_time] # Catch unexpected routes
            
        # 3. Router Accuracy
        router_ok = 1 if predicted_path == expected_path else 0
        if not router_ok:
            self.log_failure("ROUTER_ERROR")
            
        # 4. Content Evaluation (Only grade if path is qualitative/generative)
        if predicted_path == "qualitative":
            if not context:
                self.log_failure("EMPTY_RETRIEVAL")
                
            judgement = judge_e2e_quality(question, context, answer)
            faithfulness = judgement.get("faithfulness", 0.0)
            relevance = judgement.get("relevance", 1)
            
            # Taxonomy: Hallucination
            if faithfulness < 1.0:
                self.log_failure("HALLUCINATION")
                
            # Taxonomy: Fallbacks
            fallback_phrase = "Information non disponible"
            if not context and fallback_phrase in answer:
                self.log_failure("FALLBACK_APPROPRIATE")
            elif context and fallback_phrase in answer:
                self.log_failure("FALLBACK_INAPPROPRIATE")
                
            # Calculate final AQS for this query
            aqs = self.calculate_aqs(faithfulness, relevance, router_ok, retrieval_score)
            self.aqs_scores.append(aqs)
            
            print(f"[{predicted_path.upper()}] Q: {question[:40]:<40} | AQS: {aqs:.3f} | Time: {exec_time:.2f}s")
        else:
            print(f"[{predicted_path.upper()}] Q: {question[:40]:<40} | Time: {exec_time:.2f}s")

    def print_executive_report(self):
        print("\n" + "="*50)
        print(" LAYER 5: EXECUTIVE SYSTEM REPORT")
        print("="*50)
        
        # AQS
        mean_aqs = sum(self.aqs_scores) / len(self.aqs_scores) if self.aqs_scores else 0
        print(f"\n GLOBAL ANSWER QUALITY SCORE (AQS): {mean_aqs:.3f} / 1.000")
        
        # Latency Profiling
        print("\n LATENCY PROFILING:")
        for path, times in self.latency_logs.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                print(f"  - {path.capitalize():<15}: Avg {avg_time:.3f}s | Max {max_time:.3f}s")
            else:
                print(f"  - {path.capitalize():<15}: No data")
                
        # Failure Taxonomy
        print("\n FAILURE MODE TAXONOMY:")
        for category, count in self.taxonomy.items():
            if count > 0:
                print(f"  - {category:<25}: {count} occurrences")
        print("="*50)

import main as rag 

print("\n[SETUP] Initializing RAG context for automated testing...")
rag.validate_env()
db = rag.get_mongo_client()[rag.CONFIG["DB_NAME"]]

#Define a static test environment for the batch of questions
TEST_LOCATION = "Saint-Hyacinthe" 
TEST_RADIUS = 5000
city_hint = f"{TEST_LOCATION}, Quebec, Canada"

print(f"[SETUP] Geocoding test location: {TEST_LOCATION}...")
test_lon, test_lat = rag.geocode_address(TEST_LOCATION)

if test_lon is None:
    print("Critical Error: Failed to geocode test location. Exiting.")
    exit(1)

#Pre-load the DB context once to save time across all dataset questions
qualitative_str, notes_docs, raw_locals, demographics, naics_lines = rag.retrieve_territorial_context(
    db, test_lon, test_lat, TEST_RADIUS, debug=False
)
local_index = rag.build_structured_index(raw_locals)
llm_chain = rag.build_qualitative_chain(qualitative_str, notes_docs)


def real_pipeline_wrapper(question: str):
    """
    Acts as the API interface between the dataset questions and your RAG logic.
    """
    intent = rag.classify_question(question)
    answer = ""
    context_used = ""
    retrieval_score = 1.0 #Default for non-vector routes

    if intent == "spatial":
        answer = rag.answer_spatial(question, local_index, raw_locals, test_lon, test_lat, city_hint)
    elif intent == "filter":
        answer = rag.answer_filter(question, local_index)
    elif intent == "price_inquiry":
        answer = rag.answer_price_inquiry(question, local_index)
    else: 
        #For testing, we use .invoke() instead of .stream() to capture the final output cleanly
        response = llm_chain.invoke({"input": question, "chat_history": []})
        answer = response.get("answer", "")
        
        #Extract the unstructured chunks FAISS actually retrieved for Claude to grade
        retrieved_docs = response.get("context", [])
        notes_context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        #Combine structured data and retrieved notes to represent the full LLM context window
        context_used = f"--- STRUCTURED ---\n{qualitative_str}\n--- NOTES ---\n{notes_context}"

    return intent, answer, context_used, retrieval_score

#EXECUTION
if __name__ == "__main__":
    try:
        with open("test_dataset.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("[ERROR] test_dataset.json missing. Using fallback mock questions.")
        dataset = [
            {"question": "Quelle est la population ?", "expected_path": "qualitative"},
            {"question": "Montre moi les 3 locaux les plus proches", "expected_path": "spatial"},
            {"question": "Quel est le local le moins cher ?", "expected_path": "price_inquiry"}
        ]

    evaluator = RAGEvaluatorE2E()
    
    print("\nSTARTING E2E PRODUCTION BENCHMARK...\n")
    for case in dataset:
        evaluator.evaluate_test_case(case, real_pipeline_wrapper)
        time.sleep(2)
    evaluator.print_executive_report()