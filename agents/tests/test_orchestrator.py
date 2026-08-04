"""Unit tests for agents.orchestrator.Orchestrator.

CompetitiveAgent/RegulatoryAgent/SynthesisAgent each have their own test
suite already - this one only tests Orchestrator's own coordination logic
(sub-intent construction, fan-out, feeding results to Synthesis), so it
stubs each sub-agent's execute() directly rather than re-faking geo_rag.
"""

import asyncio

from agents.orchestrator import Orchestrator


def run(coro):
    return asyncio.run(coro)


class DummyGeoRag:
    """Never actually called - Orchestrator.__init__ only stores it on the
    sub-agents it constructs, whose execute() is stubbed out below."""

    def geocode_landmark(self, place_name):
        raise AssertionError("DummyGeoRag should never be called in these tests")


def _stub_execute(return_value):
    calls = []

    async def _stub(intent):
        calls.append(intent)
        return return_value

    _stub.calls = calls
    return _stub


COMPETITIVE_ENVELOPE = {
    "agent": "competitive",
    "status": "success",
    "data": {"direct_competitor_count": 1, "saturation_level": "low", "competitor_density_per_km2": 0.1},
    "error": None,
    "processing_time_ms": 1.0,
    "timestamp": "2026-08-04T12:00:00+00:00",
}

REGULATORY_ENVELOPE = {
    "agent": "regulatory",
    "status": "success",
    "data": {"zoning": {"zone_code": "C03", "ville": "Bromont", "status": "ok"},
             "use_permission": {"verdict": "not_evaluated", "matched_use": None, "reason": None}},
    "error": None,
    "processing_time_ms": 1.0,
    "timestamp": "2026-08-04T12:00:00+00:00",
}


def _wired_orchestrator(competitive_result=COMPETITIVE_ENVELOPE, regulatory_result=REGULATORY_ENVELOPE):
    orchestrator = Orchestrator(geo_rag=DummyGeoRag())
    orchestrator._competitive.execute = _stub_execute(competitive_result)
    orchestrator._regulatory.execute = _stub_execute(regulatory_result)
    return orchestrator


def test_run_requires_location():
    orchestrator = _wired_orchestrator()

    result = run(orchestrator.run({}))

    assert "error" in result
    assert orchestrator._competitive.execute.calls == []
    assert orchestrator._regulatory.execute.calls == []


def test_run_forwards_shared_fields_to_both_agents():
    orchestrator = _wired_orchestrator()

    run(orchestrator.run({"location": "Bromont", "business_type": "restaurant"}))

    competitive_intent = orchestrator._competitive.execute.calls[0]
    regulatory_intent = orchestrator._regulatory.execute.calls[0]
    assert competitive_intent["location"] == "Bromont"
    assert competitive_intent["business_type"] == "restaurant"
    assert regulatory_intent["location"] == "Bromont"
    assert regulatory_intent["business_type"] == "restaurant"


def test_run_forwards_agent_specific_fields_only_to_their_owner():
    orchestrator = _wired_orchestrator()

    run(orchestrator.run({
        "location": "Bromont",
        "naics_code": "7225",
        "radius_meters": 800,
        "identifier": "681921118600010000",
        "identifier_type": "matricule",
    }))

    competitive_intent = orchestrator._competitive.execute.calls[0]
    regulatory_intent = orchestrator._regulatory.execute.calls[0]
    assert competitive_intent["naics_code"] == "7225"
    assert competitive_intent["radius_meters"] == 800
    assert "identifier" not in competitive_intent
    assert regulatory_intent["identifier"] == "681921118600010000"
    assert regulatory_intent["identifier_type"] == "matricule"
    assert "naics_code" not in regulatory_intent


def test_run_feeds_both_envelopes_into_synthesis():
    orchestrator = _wired_orchestrator()

    result = run(orchestrator.run({"location": "Bromont", "business_type": "restaurant"}))

    assert result["status"] == "success"
    assert result["data"]["agent_reports"] == {
        "competitive": COMPETITIVE_ENVELOPE,
        "regulatory": REGULATORY_ENVELOPE,
    }
    assert result["data"]["overall_verdict"] == "favorable"
