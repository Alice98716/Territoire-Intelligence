"""AGENT SYNTHÈSE — combines specialized agents' reports into one summary.

Deliberately rule-based, not LLM-narrated, for this phase: tools.run_analysis
/ api_server.generate_pillar_report already narrate a report via Claude, but
that function is built around the FRONTEND's own precomputed "4 pillars"
metrics (ISM, PAD, SPC, ...) - a different shape than what CompetitiveAgent/
RegulatoryAgent actually produce. Force-fitting their output into those
fields would mean inventing numbers that don't exist, and would add a paid
LLM call as a hidden side effect of what's still foundation work. This
agent instead deterministically aggregates whatever specialized-agent
envelopes it's given - narrative polish can be layered on top later, once
all specialized agents exist and there's a real, complete input shape to
narrate from.

Extensibility: only "competitive" and "regulatory" have a dedicated entry
in _SUMMARY_EXTRACTORS below, since those are the only two specialized
agents that exist so far. An unrecognized agent name (the future
Demographer/RealEstate/Economist) still gets a generic summary line via
_summarize_generic rather than being silently dropped or crashing - add a
dedicated extractor for each as it lands.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agents.base_agent import BaseAgent

# (agent_name, data) -> (summary_point | None, flags: list[str])
SummaryExtractor = Callable[[str, dict[str, Any]], tuple[Optional[str], list[str]]]

# Only "high" is treated as a red flag - "moderate" is informational only
# (see CompetitiveAgent's saturation_level thresholds), so a merely busy
# area doesn't get treated the same as a genuinely saturated one.
_SATURATED = "high"

# Verdicts that must surface as a flag - "not_listed" is a real gap (the
# proposed use isn't in the zone's list under this wording), "unknown"
# means the zoning lookup itself couldn't determine an answer. Neither is
# swept into "favorable" - see RegulatoryAgent's own docstring on why
# silence must never read as "no restrictions".
_REGULATORY_FLAG_VERDICTS = {"not_listed", "unknown"}


def _summarize_competitive(_name: str, data: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    count = data.get("direct_competitor_count")
    saturation = data.get("saturation_level")
    density = data.get("competitor_density_per_km2")

    point = None
    if count is not None:
        point = (
            f"Compétitif : {count} concurrent(s) direct(s) dans la zone "
            f"(densité {density}/km², saturation {saturation})."
        )

    flags = []
    if saturation == _SATURATED:
        flags.append(f"Saturation concurrentielle élevée ({count} concurrents directs dans la zone).")
    return point, flags


def _summarize_regulatory(_name: str, data: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    zoning = data.get("zoning") or {}
    use_permission = data.get("use_permission") or {}
    verdict = use_permission.get("verdict")
    zone_code = zoning.get("zone_code")
    ville = zoning.get("ville", "?")

    point = None
    if verdict == "not_evaluated":
        if zone_code:
            point = f"Réglementaire : zone {zone_code} identifiée ({ville})."
    elif verdict:
        point = f"Réglementaire : usage proposé — verdict « {verdict} » pour la zone {zone_code} ({ville})."

    flags = []
    if verdict == "not_listed":
        flags.append(
            f"L'usage proposé n'apparaît pas explicitement dans les usages permis/conditionnels "
            f"de la zone {zone_code} ({ville})."
        )
    elif verdict == "unknown":
        flags.append(
            f"Le statut réglementaire de l'usage proposé n'a pas pu être déterminé pour {ville} "
            f"— à vérifier manuellement avant de procéder."
        )
    return point, flags


def _summarize_generic(name: str, data: dict[str, Any]) -> tuple[Optional[str], list[str]]:
    """Fallback for any agent without a dedicated extractor yet (Demographer,
    Real Estate, Economist) - keeps synthesis from silently dropping or
    crashing on a report it doesn't have domain-specific handling for."""
    return f"{name.capitalize()} : données disponibles ({len(data)} champs).", []


_SUMMARY_EXTRACTORS: dict[str, SummaryExtractor] = {
    "competitive": _summarize_competitive,
    "regulatory": _summarize_regulatory,
}


class SynthesisAgent(BaseAgent):
    """Combines other agents' `execute()` envelopes into one summary.

    Expects `intent` to look like:
        {
            "agent_reports": {
                "competitive": {<CompetitiveAgent.execute() envelope>},
                "regulatory": {<RegulatoryAgent.execute() envelope>},
                ...
            }
        }
    Each value must be a standardized BaseAgent envelope (agent/status/data/
    error/processing_time_ms/timestamp) - i.e. exactly what another agent's
    `execute()` returns, not raw `analyze()` output.
    """

    def __init__(self) -> None:
        super().__init__(
            name="synthesis",
            description="Combine les rapports des agents spécialisés en une synthèse et une recommandation globale.",
        )

    async def validate_inputs(self, intent: dict[str, Any]) -> bool:
        """Requires a non-empty agent_reports dict of well-formed envelopes."""
        reports = intent.get("agent_reports")
        if not isinstance(reports, dict) or not reports:
            return False
        return all(isinstance(report, dict) and "status" in report for report in reports.values())

    async def analyze(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Splits reports into usable/unavailable, then aggregates a verdict."""
        reports = intent["agent_reports"]

        usable: dict[str, dict[str, Any]] = {}
        unavailable: dict[str, str] = {}
        for name, envelope in reports.items():
            data = envelope.get("data")
            # status == "success" at the envelope level doesn't guarantee
            # usable data - CompetitiveAgent/RegulatoryAgent both still
            # return status "success" with a soft {"error": ...} payload
            # when e.g. geocoding fails (see their own docstrings), rather
            # than raising. Both cases mean "nothing to synthesize here".
            if envelope.get("status") == "success" and isinstance(data, dict) and "error" not in data:
                usable[name] = data
            else:
                unavailable[name] = self._unavailable_reason(envelope)

        summary_points: list[str] = []
        flags: list[str] = []
        for name, data in usable.items():
            extractor = _SUMMARY_EXTRACTORS.get(name, _summarize_generic)
            point, agent_flags = extractor(name, data)
            if point:
                summary_points.append(point)
            flags.extend(agent_flags)

        if not usable:
            overall_verdict = "insufficient_data"
        elif flags:
            overall_verdict = "proceed_with_caution"
        else:
            overall_verdict = "favorable"

        return {
            "overall_verdict": overall_verdict,
            "summary_points": summary_points,
            "flags": flags,
            "unavailable_agents": unavailable,
            "agent_reports": reports,
        }

    @staticmethod
    def _unavailable_reason(envelope: dict[str, Any]) -> str:
        status = envelope.get("status")
        if status == "success":
            data = envelope.get("data") or {}
            return data.get("error", "Analyse incomplète (données manquantes).")
        return envelope.get("error") or f"Échec de l'agent (status={status})."
