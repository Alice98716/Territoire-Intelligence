"""Coordinates the specialized agents into one request/response cycle.

Not a BaseAgent subclass - orchestration isn't a domain analysis with its
own analyze()/validate_inputs() split, it's "run the specialized agents,
hand their reports to Synthesis". Constructed once (mirrors how
api_server.py holds a single long-lived geo_rag/naics_classifier instance)
and reused across requests.

Only competitive/regulatory exist so far (see agents/README.md's Status
section) - Demographer, Real Estate, and Economist plug in here the same
way once they land: build their intent, add them to the asyncio.gather
call below, add their envelope to agent_reports.
"""

from __future__ import annotations

from typing import Any, Optional

import asyncio

from agents.competitive_agent import CompetitiveAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.synthesis_agent import SynthesisAgent
from spatial_rag_v1 import NAICSClassifier, SpatialHybridRAG


class Orchestrator:
    """Runs CompetitiveAgent + RegulatoryAgent and synthesizes their reports."""

    def __init__(self, geo_rag: SpatialHybridRAG, naics_classifier: Optional[NAICSClassifier] = None) -> None:
        """Initializes the agents this orchestrator coordinates.

        Args:
            geo_rag: Shared spatial engine instance (same one api_server.py
                builds at startup) - not owned or connected here.
            naics_classifier: Optional shared NAICS classifier, forwarded to
                CompetitiveAgent (best-effort - see its own docstring).
        """
        self._competitive = CompetitiveAgent(geo_rag, naics_classifier)
        self._regulatory = RegulatoryAgent(geo_rag)
        self._synthesis = SynthesisAgent()

    async def run(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Runs the specialized agents for `intent` and returns SynthesisAgent's envelope.

        Args:
            intent: {
                "location": str,               # required
                "business_type": str,          # optional - forwarded to both
                                                # agents (competitor category /
                                                # proposed zoning use)
                "naics_code": str,              # optional - CompetitiveAgent only
                "radius_meters": float,         # optional - CompetitiveAgent only
                "identifier": str,              # optional - RegulatoryAgent only
                "identifier_type": str,         # optional - RegulatoryAgent only
            }

        Returns:
            SynthesisAgent's `execute()` envelope - its `data.agent_reports`
            field carries each specialized agent's own full envelope, so
            callers needing per-agent detail don't need a separate return
            value for it.
        """
        location = intent.get("location")
        if not isinstance(location, str) or not location.strip():
            return {"error": "location is required"}

        business_type = intent.get("business_type")

        competitive_intent = {
            "location": location,
            "business_type": business_type,
            "naics_code": intent.get("naics_code"),
            "radius_meters": intent.get("radius_meters"),
        }
        regulatory_intent = {
            "location": location,
            "business_type": business_type,
            "identifier": intent.get("identifier"),
            "identifier_type": intent.get("identifier_type"),
        }

        # Both agents currently do blocking pymongo calls under the hood -
        # SpatialHybridRAG has no async driver yet (motor is a Phase 0
        # dependency, not yet wired in - see agents/README.md) - so gather
        # doesn't buy real concurrency today. Kept anyway: it's the correct
        # shape once that changes, and is harmless either way since these
        # two agents don't depend on each other's output.
        competitive_result, regulatory_result = await asyncio.gather(
            self._competitive.execute(competitive_intent),
            self._regulatory.execute(regulatory_intent),
        )

        return await self._synthesis.execute({
            "agent_reports": {
                "competitive": competitive_result,
                "regulatory": regulatory_result,
            }
        })
