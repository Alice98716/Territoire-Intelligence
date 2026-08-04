# agents/

Foundation for the multi-agent business viability system. This package will
eventually host six specialized agents — Demographer, Competitive,
Regulatory, Real Estate, Economist, and Synthesis — that replace the
current single-call analysis in `api_server.py` / `tools.py`.

## Structure

```
agents/
  base_agent.py           Abstract base class every agent inherits from
  competitive_agent.py    AGENT 3 - Compétitif: competitive landscape / market saturation
  regulatory_agent.py     AGENT RÉGLEMENTAIRE: zonage, usages permis/conditionnels, rôle foncier
  synthesis_agent.py      AGENT SYNTHÈSE: aggregates other agents' reports into one summary/verdict
  orchestrator.py         Runs the specialized agents and feeds their reports to Synthesis
  communication/
    message_queue.py      Redis pub/sub abstraction for agent-to-agent messages
  tests/
    test_base_agent.py        Unit tests for BaseAgent
    test_competitive_agent.py Unit tests for CompetitiveAgent
    test_regulatory_agent.py  Unit tests for RegulatoryAgent
    test_synthesis_agent.py   Unit tests for SynthesisAgent
    test_orchestrator.py      Unit tests for Orchestrator
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

## CompetitiveAgent (Agent 3)

`agents/competitive_agent.py` analyzes competitive landscape and market
saturation around a location. It adds no new Mongo queries of its own —
it's built entirely on the search/ranking pipeline `tools.py` already
exposes:

- `tools.spatial_search` (`SpatialHybridRAG.hard_spatial_filter` +
  `hybrid_semantic_search` + `pareto_rank`) for the ranked, radius-bound
  candidate list and its `match_status` tagging (Exact/Broad NAICS or
  Category Match) used to separate direct from adjacent competitors.
- `NAICSClassifier.classify_business_type` (best-effort, same 0.4
  confidence floor `api_server.py` already uses) to resolve a free-text
  `business_type` to a NAICS code when the caller didn't supply one.
- `tools.find_dissemination_area` + `tools.count_businesses_in_da` for a
  secondary, boundary-aware saturation count inside the actual
  dissemination area, as a cross-check against the fixed-radius circle.

Intent shape: `{"location": str, "business_type"?: str, "naics_code"?: str,
"radius_meters"?: float}` — at least one of `business_type`/`naics_code` is
required. See the module docstring for the full output shape.

Constructed with the same `geo_rag` (`SpatialHybridRAG`) and
`naics_classifier` (`NAICSClassifier`) instances `api_server.py` builds at
startup — this agent doesn't own or connect to Mongo itself.

## RegulatoryAgent

`agents/regulatory_agent.py` analyzes the regulatory framework applicable
to a location — zoning and, where a proposed use is given, a
permitted/conditional/not-listed verdict against the zone's use lists —
plus the property-tax-roll record. Also built entirely on existing
`tools.py` wrappers, no new Mongo queries:

- `tools.find_zone_at_location` — geocodes, resolves the ville, does the
  point-in-polygon test against the zonage layer, and joins zone code
  against `usages_zones` for permitted/conditional uses. Coverage:
  Bromont, Saint-Hyacinthe, Québec for the zone code; only Bromont
  currently has the permitted/conditional use lists loaded — elsewhere
  the agent reports `verdict: "unknown"`, never a false "not_listed".
- `tools.get_property_tax_info` — the `role_foncier` record (zoning-
  adjacent fields: `cubf` land-use code, `categorie`, `imposable`,
  assessed values, parcel boundary), looked up by matricule, address, or
  coordinates. Defaults to the location's own geocoded coordinates rather
  than treating `location` as a literal civic address, since the
  underlying `"adresse"` lookup is an exact/fuzzy string match, not a
  geocoded one.

Intent shape: `{"location": str, "business_type"?: str, "identifier"?: str,
"identifier_type"?: "matricule"|"adresse"|"coordinates"}` — `identifier`
and `identifier_type` must be given together. See the module docstring for
the full output shape.

## SynthesisAgent

`agents/synthesis_agent.py` combines other agents' `execute()` envelopes
into one summary. Deliberately **rule-based, not LLM-narrated**: reusing
`tools.run_analysis`/`generate_pillar_report` (the existing narration path)
would mean force-fitting CompetitiveAgent/RegulatoryAgent's actual output
into a frontend-precomputed-metrics shape they don't produce (ISM/PAD/SPC),
and would add a paid Claude API call as a side effect of foundation work.
Narrative polish can layer on top later, once all specialized agents exist.

Intent shape: `{"agent_reports": {"competitive": {...envelope}, "regulatory":
{...envelope}, ...}}` — each value must be another agent's full `execute()`
envelope, not raw `analyze()` output. For each report:

- **Usable** (`status == "success"` and no payload-level `"error"` key) →
  contributes a one-line summary point, plus any flags (e.g. `"high"`
  saturation, a `"not_listed"`/`"unknown"` zoning verdict — the latter is
  never laundered into "favorable", matching RegulatoryAgent's own caution
  about not treating missing data as "no restrictions").
- **Unavailable** (agent-level error/invalid_input, or a soft data-level
  error) → recorded in `unavailable_agents` with a reason, excluded from
  the summary.

`overall_verdict` is `"insufficient_data"` (nothing usable),
`"proceed_with_caution"` (usable but at least one flag), or `"favorable"`
(usable, no flags). Only `"competitive"` and `"regulatory"` have a
dedicated summary extractor so far — an unrecognized agent name (the
future Demographer/Real Estate/Economist) still gets a generic summary
line rather than being dropped or crashing.

## Orchestrator

`agents/orchestrator.py` is what actually makes the multi-agent system
callable end-to-end. `Orchestrator.run(intent)` builds the right sub-intent
for each specialized agent from one shared `{"location", "business_type",
...}` input, runs `CompetitiveAgent`/`RegulatoryAgent` concurrently
(`asyncio.gather` — doesn't yet buy real parallelism since both do
blocking pymongo calls under the hood, but is the correct shape once an
async Mongo driver is in the mix), and feeds both envelopes into
`SynthesisAgent`. It's not a `BaseAgent` itself — coordinating agents isn't
a domain analysis with its own `analyze()`/`validate_inputs()` split.

Wired into `api_server.py`: a module-level `orchestrator` global is built
at startup (`agents.Orchestrator(geo_rag, naics_classifier)`, right after
`naics_classifier`) and served at `POST /api/agents/analyze`:

```json
{
  "location": "123 Rue Principale, Bromont",
  "business_type": "restaurant",
  "naics_code": "7225",
  "radius_meters": 1500,
  "identifier": "681921118600010000",
  "identifier_type": "matricule"
}
```

Response is `SynthesisAgent`'s envelope — `data.agent_reports` carries each
specialized agent's own full envelope for anyone needing per-agent detail.
Only `business_type` is currently required by anything downstream
(`naics_code`/`radius_meters` are `CompetitiveAgent`-only,
`identifier`/`identifier_type` are `RegulatoryAgent`-only — see each
agent's own section above); everything except `location` is optional.

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

Phase 0 (foundation): `BaseAgent`, the message queue abstraction, and
package scaffolding — done.

Phase 1 (concrete agents): `CompetitiveAgent`, `RegulatoryAgent`, and
`SynthesisAgent` are in. Demographer, Real Estate, and Economist, plus the
orchestrator that actually wires all six together and calls `SynthesisAgent`
with their real reports, are still to come.
