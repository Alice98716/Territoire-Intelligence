# agents/

Foundation for the multi-agent business viability system (Phase 0). This
package will eventually host six specialized agents — Demographer,
Competitive, Regulatory, Real Estate, Economist, and Synthesis — that
replace the current single-call analysis in `api_server.py` / `tools.py`.

## Structure

```
agents/
  base_agent.py           Abstract base class every agent inherits from
  communication/
    message_queue.py      Redis pub/sub abstraction for agent-to-agent messages
  tests/
    test_base_agent.py    Unit tests for BaseAgent
```

## BaseAgent

Every agent subclasses `BaseAgent` (`agents/base_agent.py`) and implements:

- `async validate_inputs(intent: dict) -> bool` — can this agent run against `intent`?
- `async analyze(intent: dict) -> dict` — the agent's actual work.

Callers never invoke `analyze`/`validate_inputs` directly — they call
`await agent.execute(intent)`, which validates, times, catches errors, and
returns a standardized envelope:

```python
{
    "agent": "demographer",
    "status": "success",          # or "invalid_input" / "error"
    "data": {...},                 # None unless status == "success"
    "error": None,                 # populated on invalid_input / error
    "processing_time_ms": 123.4,
    "timestamp": "2026-08-04T12:00:00+00:00",
}
```

## Message queue

`agents/communication/message_queue.py` wraps `redis.asyncio` for pub/sub
between agents (e.g. an orchestrator publishing intents, agents publishing
results back). It exposes `publish`, `subscribe` (async generator), and
`send_direct` (agent-to-agent channel `agent:<agent_id>`). All messages are
wrapped in a standard envelope with `timestamp`, `sender_agent`,
`receiver_agent`, and `payload`.

## Local dev

`docker-compose.yml` at the repo root starts Redis (used by the message
queue) and a local MongoDB (for development only — production uses MongoDB
Atlas via the `MONGO_URI` env var, see `.env.example`):

```bash
docker compose up -d
```

## Tests

```bash
uv run pytest agents/tests/
```

## Status

Phase 0 (foundation) only: `BaseAgent`, the message queue abstraction, and
package scaffolding. The six concrete agents and the orchestrator that wires
them together land in Phase 1.
