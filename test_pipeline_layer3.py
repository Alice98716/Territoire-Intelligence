import os
import json
import re
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from langdetect import detect, DetectorFactory


DetectorFactory.seed = 0

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("critical error: ANTHROPIC_API_KEY is missing.")
    exit(1)

client = Anthropic(api_key=API_KEY.strip())
JUDGE_MODEL = "claude-haiku-4-5-20251001"

#guardrail strings
GUARDRAIL_NO_PRICE = "Je n'ai pas accès aux prix individuels dans cette vue"
GUARDRAIL_NO_DATA = "Information non disponible dans le contexte"

#3.1 FAITHFULNESS (HALLUCINATION RATE)
def judge_faithfulness(question: str, context: str, answer: str, max_retries=3) -> dict:
    """Extracts claims and checks if they are grounded in the context."""
    prompt = f"""You are a strict hallucination-detection judge for a RAG system.
    
    Context Provided to the RAG: "{context}"
    User Question: "{question}"
    RAG Answer: "{answer}"
    
    Task:
    1. Extract every factual claim made in the RAG Answer.
    2. Label each claim as:
       - "SUPPORTED" (directly backed by the Context)
       - "CONTRADICTED" (conflicts with the Context)
       - "UNVERIFIABLE" (not mentioned in the Context at all - this is a hallucination)
       
    Output your evaluation as a valid JSON object strictly matching this schema:
    {{
      "claims": [
        {{"claim": "The local has 2500 sqft", "status": "SUPPORTED"}},
        {{"claim": "The rent is 2000$", "status": "UNVERIFIABLE"}}
      ]
    }}
    """
    
    for attempt in range(max_retries):
        try:
            res = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = res.content[0].text.strip()
            
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                claims = data.get("claims", [])
                
                if not claims:
                    return {"score": 1.0, "details": "No factual claims extracted."}
                
                supported = sum(1 for c in claims if c["status"] == "SUPPORTED")
                score = supported / len(claims)
                return {"score": score, "details": claims}
                
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return {"score": 0.0, "details": "Judge failed to parse claims."}

#3.2 ANSWER RELEVANCE ---
def judge_relevance(question: str, answer: str, max_retries=3) -> int:
    """Scores how well the generated answer addresses the question (1-5)."""
    prompt = f"""Evaluate how well the Answer addresses the Question.
    
    Question: "{question}"
    Answer: "{answer}"
    
    Score 1-5 on this rubric:
    5: Directly and completely answers the question.
    3: Partially answers, misses key elements, or includes minor irrelevant fluff.
    1: Answers a completely different question or is totally irrelevant.
    
    Output ONLY a valid JSON object: {{"score": 5}}
    """
    
    for attempt in range(max_retries):
        try:
            res = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=20,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = res.content[0].text.strip()
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return int(data.get("score", 1))
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return 1

#3.3 LANGUAGE CONSISTENCY ---
def check_language_consistency(question: str, answer: str) -> bool:
    """Verifies that the RAG responds in the same language the user asked in."""
    try:
        q_lang = detect(question)
        a_lang = detect(answer)
        if GUARDRAIL_NO_DATA in answer or GUARDRAIL_NO_PRICE in answer:
            return True
        return q_lang == a_lang
    except:
        #if the text is too short to detect
        return True

#EVALUATION EXECUTION 
def evaluate_generation(question: str, context: str, generated_answer: str, expected_rule: str = None):
    print("="*60)
    print(f"Q: {question}")
    print(f"A: {generated_answer[:80]}...")
    print("-" * 60)
    
    #3.1 Rule Checks
    rule_passed = True
    if expected_rule == "EMPTY_CONTEXT":
        rule_passed = GUARDRAIL_NO_DATA in generated_answer
        print(f"Rule 4 Check (Empty Context): {'Pass' if rule_passed else 'Fail'}")
    elif expected_rule == "PRICE_INQUIRY":
        rule_passed = GUARDRAIL_NO_PRICE in generated_answer
        print(f"Rule 2 Check (No Price):      {' Pass' if rule_passed else ' Fail'}")

    #3.3 Language Check
    lang_match = check_language_consistency(question, generated_answer)
    print(f"Lang Consistency (Rule 5):    {'Pass' if lang_match else ' Fail'}")

    #3.2 Relevance
    relevance = judge_relevance(question, generated_answer)
    print(f"Answer Relevance (1-5):       {relevance}/5")

    #3.1 Faithfulness

    if GUARDRAIL_NO_DATA in generated_answer:
        print("Faithfulness Score:           1.00 (Standard Fallback)")
    else:
        faithfulness = judge_faithfulness(question, context, generated_answer)
        print(f"Faithfulness Score:           {faithfulness['score']:.2f}/1.00")
        if faithfulness['score'] < 1.0:
            print("Hallucinations Detected:")
            for claim in faithfulness['details']:
                if claim['status'] != 'SUPPORTED':
                    print(f"     - [{claim['status']}] {claim['claim']}")
    print("="*60)


if __name__ == "__main__":
    import json
    from pymongo import MongoClient
    try:
        from main import retrieve_territorial_context, build_qualitative_chain
    except ImportError:
        print("ERROR: Could not import from main.py. Make sure evaluate_layer3.py is in the same folder.")
        exit(1)

    print("\nSTARTING LAYER 3 REAL LLM EVALUATION...\n")
    try:
        with open("test_dataset.json", 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        #Isolate questions intended for LLM (qualitatitve branch only)
        test_cases = [item for item in dataset if item.get("expected_path") == "qualitative"]
        print(f"Loaded {len(test_cases)} qualitative questions for generation testing.")
    except FileNotFoundError:
        print("ERROR: test_dataset.json not found in the current directory.")
        exit(1)

    #LIVE CONTEXT
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client_mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client_mongo["overture_maps"] 

    #User's coordinates for the session (using Saint-Hyacinthe)
    MOCK_LON = -72.950  
    MOCK_LAT = 45.630
    SEARCH_RADIUS_M = 5000 

    print("[*] Retrieving live territorial context from MongoDB...")
    
    
    qualitative_str, notes_docs, raw_locals, demographics, naics_lines = retrieve_territorial_context(
        db, MOCK_LON, MOCK_LAT, SEARCH_RADIUS_M, debug=False
    )
    

    actual_context = qualitative_str + "\n\n" + "\n".join([doc.page_content for doc in notes_docs])

    print("[*] Building the local LLM generation chain...")
    llm_chain = build_qualitative_chain(qualitative_str, notes_docs)

    #EXECUTION OF EVALUATION LOOP 
    for idx, case in enumerate(test_cases, 1):
        question = case.get("question")
        metric_target = case.get("metric")
        
        print(f"\n--- Test {idx}/{len(test_cases)} ---")
        print(f"Generating live answer for: '{question}'...")
        
        try:
            
            response_blocks = llm_chain.invoke({"input": question, "chat_history": []})
            generated_answer = response_blocks.get("answer", "")
            
          
            expected_rule = None
            if "Anti-Hallucination" in metric_target and ("Paris" in question or "Canada" in question):
                expected_rule = "EMPTY_CONTEXT"
            elif metric_target == "Faithfulness" and "prix" in question.lower():
                expected_rule = "PRICE_INQUIRY"
                
           
            evaluate_generation(
                question=question,
                context=actual_context,
                generated_answer=generated_answer,
                expected_rule=expected_rule
            )
            
        except Exception as e:
            print(f"Pipeline Execution Error: {e}")
            
        #to avoid hitting claude's limit
        time.sleep(2.5)
        
    client_mongo.close()
    print("\nLAYER 3 EVALUATION COMPLETE.")