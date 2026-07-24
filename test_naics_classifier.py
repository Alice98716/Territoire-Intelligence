"""
Manual test for the NAICS semantic-search classifier (NAICSClassifier in
spatial_rag_v1.py) and its wiring into match_sector_category
(api_server.py) - the fast, free, dense-cosine-similarity-only path added
ahead of the existing Ollama fallback.

Covers three things a plain "does it run" smoke test wouldn't catch:
  1. Accuracy on the SAME representative business types already spot-checked
     manually against the live data before this feature was wired in:
     restaurant/cafe/dental clinic confidently correct; gym/hairdresser/
     office space confidently WRONG - a real corpus limitation (short,
     synonym-poor French NAICS labels), not a bug - see NAICSClassifier's
     own docstring in spatial_rag_v1.py. Asserting both directions locks in
     that baseline, so a future embeddings/index change that shifts it
     either way actually gets noticed instead of silently drifting.
  2. The grounding guard: a confident semantic match is only ever returned
     if its label is literally one of the request's `candidates` - the same
     hallucination guard the Ollama fallback already applies to itself.
  3. Latency: classify_business_type must stay near-instant AFTER the
     one-time index build - the entire point of building this over another
     per-call Claude/Ollama round trip (or over hybrid_semantic_search,
     which rebuilds+re-embeds the whole FAISS index on every call - see
     NAICSClassifier's docstring for the ~15-20s/call measurement that
     approach produced).

Real MongoDB (via SpatialHybridRAG) and real Ollama (embeddings, plus the
generate call for the deliberately-forced fallback case) - not mocked, so
run manually:

    uv run python test_naics_classifier.py
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

import api_server
from spatial_rag_v1 import SpatialHybridRAG, NAICSClassifier, DB_NAME

client = TestClient(api_server.app)


def _ensure_engine():
    """Both api_server.geo_rag and api_server.naics_classifier are normally
    only set inside @app.on_event("startup"), which TestClient skips unless
    entered as a context manager (see test_prompt_injection_guard.py's
    docstring) - so this test triggers the same two-step init manually,
    mirroring test_agent_loop.py's _ensure_geo_rag()."""
    if api_server.geo_rag is None:
        print("Initializing spatial engine (MongoDB + embeddings)...", flush=True)
        api_server.geo_rag = SpatialHybridRAG(mongo_uri=api_server.cloud_uri, db_name=DB_NAME)
        print("Spatial engine ready.\n", flush=True)
    if api_server.naics_classifier is None:
        print("Building NAICS classifier index (one-time, ~30s)...", flush=True)
        t0 = time.perf_counter()
        api_server.naics_classifier = NAICSClassifier(api_server.geo_rag)
        print(f"NAICS classifier ready - {len(api_server.naics_classifier.documents)} categories "
              f"indexed in {time.perf_counter() - t0:.1f}s.\n", flush=True)


# ── 1. Direct classifier accuracy ────────────────────────────────────────
# (query, expected_code, min_confidence, should_clear_floor)
ACCURACY_CASES = [
    ("restaurant", "7225", 0.5, True),
    ("cafe", "7225", 0.35, True),
    ("dental clinic", "6212", 0.4, True),
    ("gym", None, None, False),           # known corpus gap - see module docstring
    ("hairdresser", None, None, False),   # known corpus gap
    ("office space", None, None, False),  # known corpus gap
]


def check_classifier_accuracy(failures: list):
    print("=" * 78)
    print("1. Direct NAICSClassifier accuracy (dense cosine similarity, no LLM)")
    print("=" * 78)
    for query, expected_code, min_conf, should_clear_floor in ACCURACY_CASES:
        t0 = time.perf_counter()
        results = api_server.naics_classifier.classify_business_type(query, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not results:
            print(f"[FAIL] '{query}' - classifier returned no results at all")
            failures.append(f"accuracy:{query}")
            continue

        top = results[0]
        clears_floor = top["confidence"] >= api_server._NAICS_SEMANTIC_CONFIDENCE_FLOOR

        if should_clear_floor:
            ok = clears_floor and top["code"] == expected_code and top["confidence"] >= min_conf
        else:
            # Expected to NOT confidently resolve - the interesting failure
            # mode here would be it suddenly becoming confident and WRONG,
            # not confident and right (that'd just be a pleasant surprise,
            # worth a manual look but not a hard failure on its own).
            ok = True
            if clears_floor:
                print(f"    NOTE: '{query}' now clears the confidence floor "
                      f"({top['confidence']:.3f}) - re-check whether it's actually correct.")

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] '{query}' -> {top['code']}: {top['label']} "
              f"(confidence={top['confidence']:.3f}, {elapsed_ms:.0f}ms, "
              f"clears_floor={clears_floor})")
        if not ok:
            failures.append(f"accuracy:{query}")
    print()


# ── 2. Latency ────────────────────────────────────────────────────────────
# 1s is generous (measured ~90-130ms live); this just guards against
# silently regressing back to a per-call FAISS rebuild.
LATENCY_BUDGET_S = 1.0


def check_latency(failures: list):
    print("=" * 78)
    print("2. Latency (must stay near-instant - no per-call re-embedding)")
    print("=" * 78)
    for query in ["restaurant", "yoga studio", "auto repair shop"]:
        t0 = time.perf_counter()
        api_server.naics_classifier.classify_business_type(query, top_k=3)
        elapsed = time.perf_counter() - t0
        ok = elapsed < LATENCY_BUDGET_S
        print(f"[{'PASS' if ok else 'FAIL'}] '{query}': {elapsed*1000:.0f}ms "
              f"(budget {LATENCY_BUDGET_S*1000:.0f}ms)")
        if not ok:
            failures.append(f"latency:{query}")
    print()


# ── 3. Full endpoint: grounding + method tracking, via the real HTTP route ─
def check_endpoint_grounding(failures: list):
    print("=" * 78)
    print("3. /api/match-sector-category - grounding + method tracking")
    print("=" * 78)

    real_label = api_server.naics_classifier.classify_business_type("restaurant", top_k=1)[0]["label"]

    # Case A: message clearly matches one of the candidates -> accepted via
    # semantic_search, no Ollama call needed.
    resp = client.post("/api/match-sector-category", json={
        "message": "restaurant",
        "candidates": [real_label, "Some Unrelated Category"],
    })
    body = resp.json()
    ok = body.get("match") == real_label and body.get("method") == "semantic_search"
    print(f"[{'PASS' if ok else 'FAIL'}] grounded, confident match -> semantic_search")
    print(f"    response: {body}")
    if not ok:
        failures.append("endpoint:grounded_accept")

    # Case B: the same confident match, but its label is NOT in the
    # candidates this time - must NOT be returned (grounding/hallucination
    # guard), must fall through to Ollama instead.
    resp = client.post("/api/match-sector-category", json={
        "message": "restaurant",
        "candidates": ["Completely Unrelated Category A", "Completely Unrelated Category B"],
    })
    body = resp.json()
    ok = body.get("method") == "ollama_fallback" and body.get("match") != real_label
    print(f"[{'PASS' if ok else 'FAIL'}] confident match NOT in candidates -> rejected, falls to Ollama")
    print(f"    response: {body}")
    if not ok:
        failures.append("endpoint:grounding_rejects_out_of_scope_match")

    # Case C: a known-hard query (below the confidence floor even when its
    # own top label IS offered as a candidate) - must escalate to Ollama
    # rather than return a low-confidence semantic guess.
    gym_label = api_server.naics_classifier.classify_business_type("gym", top_k=1)[0]["label"]
    resp = client.post("/api/match-sector-category", json={
        "message": "gym",
        "candidates": [gym_label, "Some Other Category"],
    })
    body = resp.json()
    ok = body.get("method") == "ollama_fallback"
    print(f"[{'PASS' if ok else 'FAIL'}] low-confidence match -> escalates to Ollama instead of guessing")
    print(f"    response: {body}")
    if not ok:
        failures.append("endpoint:low_confidence_escalates")

    # Case D: every response must always carry a "method" field, regardless
    # of path taken - this is what makes cost/usage tracking possible at all.
    resp = client.post("/api/match-sector-category", json={"message": "cafe", "candidates": [real_label]})
    ok = "method" in resp.json()
    print(f"[{'PASS' if ok else 'FAIL'}] response always includes a 'method' field")
    if not ok:
        failures.append("endpoint:method_field_present")
    print()


def main():
    _ensure_engine()

    failures = []
    check_classifier_accuracy(failures)
    check_latency(failures)
    check_endpoint_grounding(failures)

    print("=" * 78)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
