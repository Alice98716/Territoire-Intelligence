import os
import json
import time
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("CRITICAL ERROR: ANTHROPIC_API_KEY is missing.")
    exit(1)

client = Anthropic(api_key=API_KEY.strip())
JUDGE_MODEL = "claude-haiku-4-5-20251001"

def judge_conversation_behavior(history: list, question: str, answer: str, test_type: str, max_retries=3) -> dict:
    """
    Uses Claude to evaluate different conversational memory behaviors based on the test type.
    """
    
    #format history for the prompt
    history_text = "\n".join([f"{'USER' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history])
    
    #adjust the scoring
    if test_type == "retention":
        rubric = """Task: Determine if the AI successfully retained and utilized the specific subject from the Chat History to answer the vague Current Question.
        Score 1 (Pass) if the AI correctly inferred the subject from context. Score 0 (Fail) if it got confused or asked for clarification."""
    
    elif test_type == "boundary":
        rubric = """Task: The system has a strict memory limit. The Chat History provided exceeds this limit, meaning the AI SHOULD HAVE FORGOTTEN the earliest messages.
        Score 1 (Pass) if the AI correctly states it does not know or cannot answer based on the (evicted) context. Score 0 (Fail) if it hallucinates an answer or remembers the evicted context (meaning the eviction logic failed)."""
        
    elif test_type == "pollution":
        rubric = """Task: Determine if the AI ignored the irrelevant small-talk in the Chat History and answered the Current Question factually and directly.
        Score 1 (Pass) if the answer is completely professional and focused on the current question. Score 0 (Fail) if the AI gets distracted by the off-topic chat history."""
        
    elif test_type == "crosspath":
        rubric = """Task: In the Chat History, a separate system module provided a list of specific addresses. Determine if the AI successfully analyzed those EXACT addresses in its new answer.
        Score 1 (Pass) if the AI references the specific entities from the AI's previous turn. Score 0 (Fail) if it talks in generalities or ignores the previously retrieved items."""

    prompt = f"""You are a QA judge evaluating the short-term memory logic of an AI assistant.
    
    CHAT HISTORY PRIOR TO QUESTION:
    {history_text if history_text else "[Empty History]"}
    
    CURRENT QUESTION: "{question}"
    AI'S GENERATED ANSWER: "{answer}"
    
    {rubric}
    
    Output a valid JSON object strictly matching this schema:
    {{
      "score": 1,
      "reasoning": "Brief explanation of why it passed or failed."
    }}
    """
    
    for attempt in range(max_retries):
        try:
            res = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_text = res.content[0].text.strip()
            
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            time.sleep(2 ** attempt)
            
    return {"score": 0, "reasoning": "Judge API failure."}

#RUNNING EVALUATION
def run_layer4_evaluations():
    print("=" * 70)
    print(" LAYER 4 INITIALIZATION: CONVERSATIONAL & MEMORY EVALUATION")
    print("=" * 70)

  
    try:
        from main import build_qualitative_chain, CONFIG

        dummy_context = "Le 450 rue Girouard est un local commercial de 1500 pieds carrés. Le 100 rue Wellington est un ancien restaurant."
        llm_chain = build_qualitative_chain(dummy_context, [])
        memory_limit = CONFIG.get("CHAT_MEMORY_LIMIT", 6)
    except ImportError:
        print(" ERROR: Could not import build_qualitative_chain from main.py.")
        return

    #4.1 Context retention
    print("\n TEST 4.1: Context Retention (Anaphora Resolution)")
    h_retention = [
        HumanMessage(content="Combien y a-t-il de locaux commerciaux sur la rue Girouard?"),
        AIMessage(content="Il y a un local disponible au 450 rue Girouard.")
    ]
    q_retention = "Quelle est sa superficie ?" # Uses "sa" (its), requiring memory
    
    ans_retention = llm_chain.invoke({"input": q_retention, "chat_history": h_retention}).get("answer", "")
    print(f"Q: {q_retention}\nA: {ans_retention}")
    
    eval_ret = judge_conversation_behavior(h_retention, q_retention, ans_retention, "retention")
    print(f"Result: {'PASS' if eval_ret['score'] == 1 else ' FAIL'} - {eval_ret['reasoning']}")

    #4.2 Memory boundary
    print("\nTEST 4.2: Memory Boundary (Limit = 6 messages)")
    # We push 8 messages (4 turns). The oldest pair (Turn 1) MUST be evicted by your system.
    # Note: If your slicing logic happens *before* the chain, you must apply it here: `h_boundary[-6:]`
    h_boundary = [
        HumanMessage(content="Turn 1: Le code secret de l'entreprise est ALPHA-99."),
        AIMessage(content="C'est noté, le code est ALPHA-99."),
        HumanMessage(content="Turn 2: Quelle est la météo?"),
        AIMessage(content="Il fait soleil."),
        HumanMessage(content="Turn 3: Quel est le taux de taxation?"),
        AIMessage(content="Il est de 15%."),
        HumanMessage(content="Turn 4: Merci pour les infos."),
        AIMessage(content="De rien!")
    ]
    
    sliced_history = h_boundary[-memory_limit:]
    
    q_boundary = "Quel était le code secret mentionné au tout début?"
    ans_boundary = llm_chain.invoke({"input": q_boundary, "chat_history": sliced_history}).get("answer", "")
    print(f"Q: {q_boundary}\nA: {ans_boundary}")
    
    eval_bound = judge_conversation_behavior(sliced_history, q_boundary, ans_boundary, "boundary")
    print(f"Result: {'PASS' if eval_bound['score'] == 1 else ' FAIL'} - {eval_bound['reasoning']}")

    #4.3 Context pollution
    print("\n TEST 4.3: Context Pollution (Noise Injection)")
    h_pollution = [
        HumanMessage(content="J'adore les films de science-fiction, surtout Star Wars."),
        AIMessage(content="C'est un excellent choix! Les voyages spatiaux sont fascinants."),
        HumanMessage(content="Mon chat s'appelle Moustache."),
        AIMessage(content="C'est un nom très mignon pour un chat.")
    ]
    q_pollution = "Parle-moi du local sur la rue Wellington."
    
    ans_pollution = llm_chain.invoke({"input": q_pollution, "chat_history": h_pollution}).get("answer", "")
    print(f"Q: {q_pollution}\nA: {ans_pollution}")
    
    eval_poll = judge_conversation_behavior(h_pollution, q_pollution, ans_pollution, "pollution")
    print(f"Result: {' PASS' if eval_poll['score'] == 1 else ' FAIL'} - {eval_poll['reasoning']}")

    #4.4 Cross path memory
    print("\nTEST 4.4: Cross-Path Route Memory")
    # Simulating a Turn 1 where the Spatial Path was triggered and injected this raw data into history
    h_crosspath = [
        HumanMessage(content="Quels sont les 2 locaux les plus proches?"),
        AIMessage(content="[RÉSULTAT SPATIAL] 1. 100 rue Wellington. 2. 450 rue Girouard.")
    ]
    # Simulating Turn 2 where the router selected the Qualitative Path
    q_crosspath = "Est-ce qu'un de ces deux locaux était un ancien restaurant ?"
    
    ans_crosspath = llm_chain.invoke({"input": q_crosspath, "chat_history": h_crosspath}).get("answer", "")
    print(f"Q: {q_crosspath}\nA: {ans_crosspath}")
    
    eval_cross = judge_conversation_behavior(h_crosspath, q_crosspath, ans_crosspath, "crosspath")
    print(f"Result: {' PASS' if eval_cross['score'] == 1 else ' FAIL'} - {eval_cross['reasoning']}")
    
    print("\n" + "=" * 70)
    print("  LAYER 4 EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_layer4_evaluations()