"""Unit tests for agents.synthesis_agent.SynthesisAgent.

Inputs are hand-built BaseAgent-shaped envelopes (agent/status/data/error/
processing_time_ms/timestamp) rather than real CompetitiveAgent/
RegulatoryAgent instances - this documents the exact contract
SynthesisAgent expects (another agent's execute() output, not raw
analyze() output) and keeps this suite decoupled from their internals.
"""

import asyncio

from agents.synthesis_agent import SynthesisAgent


def run(coro):
    return asyncio.run(coro)


def _envelope(status: str, data=None, error=None) -> dict:
    return {
        "agent": "test",
        "status": status,
        "data": data,
        "error": error,
        "processing_time_ms": 1.0,
        "timestamp": "2026-08-04T12:00:00+00:00",
    }


COMPETITIVE_LOW = _envelope("success", data={
    "location": "Bromont",
    "direct_competitor_count": 2,
    "saturation_level": "low",
    "competitor_density_per_km2": 0.28,
})

COMPETITIVE_HIGH = _envelope("success", data={
    "location": "Bromont",
    "direct_competitor_count": 9,
    "saturation_level": "high",
    "competitor_density_per_km2": 1.27,
})

REGULATORY_PERMITTED = _envelope("success", data={
    "location": "Bromont",
    "zoning": {"zone_code": "C03", "ville": "Bromont", "status": "ok"},
    "use_permission": {"verdict": "permitted", "matched_use": "Restaurants", "reason": None},
})

REGULATORY_NOT_LISTED = _envelope("success", data={
    "location": "Bromont",
    "zoning": {"zone_code": "C03", "ville": "Bromont", "status": "ok"},
    "use_permission": {"verdict": "not_listed", "matched_use": None, "reason": "not found"},
})

REGULATORY_UNKNOWN = _envelope("success", data={
    "location": "Québec",
    "zoning": {"zone_code": "ID42", "ville": "Québec", "status": "usage_data_unavailable"},
    "use_permission": {"verdict": "unknown", "matched_use": None, "reason": "no usage data"},
})


# ── validate_inputs ─────────────────────────────────────────────────────

def test_validate_rejects_missing_agent_reports():
    agent = SynthesisAgent()
    assert run(agent.validate_inputs({})) is False


def test_validate_rejects_empty_agent_reports():
    agent = SynthesisAgent()
    assert run(agent.validate_inputs({"agent_reports": {}})) is False


def test_validate_rejects_malformed_envelope():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": {"data": {}}}}  # no "status" key
    assert run(agent.validate_inputs(intent)) is False


def test_validate_accepts_well_formed_envelopes():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_LOW}}
    assert run(agent.validate_inputs(intent)) is True


# ── analyze: verdict aggregation ────────────────────────────────────────

def test_favorable_when_no_flags_raised():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_LOW, "regulatory": REGULATORY_PERMITTED}}

    result = run(agent.execute(intent))

    assert result["status"] == "success"
    data = result["data"]
    assert data["overall_verdict"] == "favorable"
    assert data["flags"] == []
    assert len(data["summary_points"]) == 2
    assert data["unavailable_agents"] == {}


def test_high_saturation_triggers_caution_flag():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_HIGH, "regulatory": REGULATORY_PERMITTED}}

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "proceed_with_caution"
    assert any("saturation" in f.lower() for f in data["flags"])


def test_not_listed_use_triggers_caution_flag():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_LOW, "regulatory": REGULATORY_NOT_LISTED}}

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "proceed_with_caution"
    assert any("n'apparaît pas" in f for f in data["flags"])


def test_unknown_zoning_status_never_reads_as_favorable():
    """RegulatoryAgent's own docstring insists silence must never read as
    'no restrictions' - synthesis must not launder that into 'favorable'."""
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_LOW, "regulatory": REGULATORY_UNKNOWN}}

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "proceed_with_caution"
    assert any("vérifier manuellement" in f for f in data["flags"])


def test_insufficient_data_when_every_report_is_unusable():
    agent = SynthesisAgent()
    intent = {
        "agent_reports": {
            "competitive": _envelope("error", error="boom"),
            "regulatory": _envelope("invalid_input", error="Input validation failed"),
        }
    }

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "insufficient_data"
    assert data["summary_points"] == []
    assert set(data["unavailable_agents"]) == {"competitive", "regulatory"}


def test_soft_data_level_error_counts_as_unavailable_not_usable():
    """status == 'success' with a payload-level {"error": ...} (e.g. a
    geocode failure inside CompetitiveAgent/RegulatoryAgent) must not be
    treated as real data just because the envelope status is 'success'."""
    agent = SynthesisAgent()
    intent = {
        "agent_reports": {
            "competitive": _envelope("success", data={"location": "Nowhere", "error": "Could not geocode 'Nowhere'"}),
        }
    }

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "insufficient_data"
    assert data["unavailable_agents"]["competitive"] == "Could not geocode 'Nowhere'"


def test_unrecognized_agent_name_gets_generic_summary_without_crashing():
    agent = SynthesisAgent()
    intent = {
        "agent_reports": {
            "demographer": _envelope("success", data={"population": 12000, "median_income": 54000}),
        }
    }

    result = run(agent.execute(intent))

    data = result["data"]
    assert data["overall_verdict"] == "favorable"
    assert "Demographer" in data["summary_points"][0]


def test_agent_reports_passed_through_for_traceability():
    agent = SynthesisAgent()
    intent = {"agent_reports": {"competitive": COMPETITIVE_LOW}}

    result = run(agent.execute(intent))

    assert result["data"]["agent_reports"] == {"competitive": COMPETITIVE_LOW}
