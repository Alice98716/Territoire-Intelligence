"""
Manual integration test for the agentic tool-use loop (run_agent_loop in
api_server.py). Runs a handful of representative queries end-to-end and
prints each one's tool-call trace, so a regression in the loop's reasoning
(wrong tool picked, infinite back-and-forth, wrong final answer) is visible
without going through the FastAPI server or the frontend.

Not a pytest suite - this makes real Claude API calls and a real MongoDB
connection (via SpatialHybridRAG), so it's meant to be run manually:

    uv run python test_agent_loop.py
"""
import sys
import time

# Windows' console defaults to cp1252, which can't encode characters Claude's
# responses sometimes include (e.g. checkmarks) - reconfigure to UTF-8 so a
# print() doesn't crash the whole run partway through.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import api_server
from spatial_rag_v1 import SpatialHybridRAG, DB_NAME

# "More than 4 iterations" is the threshold the task asked to flag - not a
# hard failure, just a signal that a query needed unusually many rounds.
MAX_ITERATIONS_WARNING = 4

TEST_QUERIES = [
    ("1", "Find vacant offices within 500m of downtown Montreal"),
    (
        "2",
        "List the businesses in this dissemination area. "
        "(Test context: use the dissemination area nearest to Saint-Hyacinthe, "
        "Quebec (lat 45.6259, lon -72.9465) as \"this\" area.)",
    ),
    ("3", "Analyze the competitive landscape for bakeries near Saint-Hyacinthe"),
    ("4", "Which address is best for opening a gym near downtown?"),
    ("5", "Compare Bromont and Saint-Hyacinthe for opening a bakery"),
]


def _ensure_geo_rag():
    """run_agent_loop's tools need a live geo_rag, which api_server.py
    normally only sets inside @app.on_event("startup") when uvicorn actually
    runs the app. This script calls run_agent_loop directly (no uvicorn), so
    it has to initialize that same instance itself - same call the startup
    event makes, just triggered manually."""
    if api_server.geo_rag is None:
        print("Initializing spatial engine (MongoDB + embeddings)...", flush=True)
        api_server.geo_rag = SpatialHybridRAG(mongo_uri=api_server.cloud_uri, db_name=DB_NAME)
        print("Spatial engine ready.\n", flush=True)


def _run_one(label: str, query: str) -> dict:
    print("=" * 78)
    print(f"QUERY {label}: {query}")
    print("=" * 78)

    t0 = time.perf_counter()
    try:
        result = api_server.run_agent_loop(query)
    except Exception as e:
        print(f"CRASHED: {e}\n")
        return {
            "label": label, "query": query, "status": "crashed", "iterations": 0,
            "final_text": str(e), "tool_sequence": [], "elapsed": time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0

    tool_sequence = [r["tool"] for r in result["partial_results"]]

    print(f"\n--- Summary for query {label} ---")
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tools called in order: {tool_sequence or '(none - answered directly)'}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Final response:\n{result['final_text']}\n")

    if result["status"] == "incomplete":
        print(f"WARNING: query {label} hit the max iteration limit "
              f"({result['iterations']}) without a final answer.\n")
    elif result["iterations"] > MAX_ITERATIONS_WARNING:
        print(f"WARNING: query {label} took {result['iterations']} iterations "
              f"(> {MAX_ITERATIONS_WARNING}) to resolve.\n")

    return {**result, "label": label, "query": query, "tool_sequence": tool_sequence, "elapsed": elapsed}


def main():
    _ensure_geo_rag()

    all_results = []
    for i, (label, query) in enumerate(TEST_QUERIES):
        if i > 0:
            time.sleep(1)  # courtesy pause between queries (Nominatim's usage policy caps at 1 req/s)
        all_results.append(_run_one(label, query))

    print("=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)
    for r in all_results:
        flag = ""
        if r["status"] in ("incomplete", "crashed"):
            flag = f"  <- {r['status'].upper()}"
        elif r["iterations"] > MAX_ITERATIONS_WARNING:
            flag = f"  <- {r['iterations']} iterations (>{MAX_ITERATIONS_WARNING})"
        print(f"[{r['label']}] {r['iterations']} iter, {len(r['tool_sequence'])} tool call(s), "
              f"status={r['status']}{flag}")
        print(f"    query: {r['query'][:90]}")
        print(f"    tools: {r['tool_sequence']}")


if __name__ == "__main__":
    main()
