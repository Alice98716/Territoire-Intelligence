"""AGENT RÉGLEMENTAIRE — L'Analyste Réglementation.

Analyzes the regulatory framework applicable to a target location: zoning
(zone code, permitted/conditional uses) and the property-tax-roll record
(land-use code, assessment category, taxable status). Built entirely on
tools.py's existing find_zone_at_location and get_property_tax_info
wrappers - this agent adds no new Mongo queries or geometry logic of its
own, only the permitted-use verdict on top of what those tools already
return.

Coverage is whatever the underlying data covers, not universal: zoning
currently resolves for Bromont, Saint-Hyacinthe, and Québec only, and
permitted/conditional use LISTS (not just the zone code) currently only
exist for Bromont - see find_zone_at_location's own docstring in tools.py.
Elsewhere this agent reports status "usage_data_unavailable" /
verdict "unknown" rather than silently implying "no restrictions" -
exactly the caution find_zone_at_location's docstring already calls for.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.base_agent import BaseAgent
from spatial_rag_v1 import SpatialHybridRAG, _fold_accents
from tools import find_zone_at_location, get_property_tax_info

# Same three modes get_property_tax_info itself accepts - validated here too
# so a bad identifier_type is rejected as "invalid_input" up front rather
# than surfacing as a generic {"error": ...} buried inside "property".
_VALID_IDENTIFIER_TYPES = {"matricule", "adresse", "coordinates"}


class RegulatoryAgent(BaseAgent):
    """Zoning + permitted-use + property-tax-roll analysis for a location.

    Expects `intent` to look like:
        {
            "location": str,             # required, geocodable place name/address
            "business_type": str,        # optional, proposed use to check against
                                          # the zone's permitted/conditional lists
                                          # ("proposed_use" also accepted)
            "identifier": str,           # optional override for the property-tax
                                          # lookup (matricule, exact civic address,
                                          # or "lon,lat") - defaults to location's
                                          # own geocoded coordinates
            "identifier_type": str,      # required alongside "identifier":
                                          # "matricule" | "adresse" | "coordinates"
        }
    """

    def __init__(self, geo_rag: SpatialHybridRAG) -> None:
        """Initializes the agent.

        Args:
            geo_rag: Shared spatial engine instance (same one api_server.py
                builds at startup) - not owned or connected here.
        """
        super().__init__(
            name="regulatory",
            description="Analyse le cadre réglementaire (zonage, usages permis/conditionnels, rôle foncier) applicable à la zone cible.",
        )
        self._geo_rag = geo_rag

    async def validate_inputs(self, intent: dict[str, Any]) -> bool:
        """Requires a non-empty location; identifier/identifier_type must be paired and valid."""
        location = intent.get("location")
        if not isinstance(location, str) or not location.strip():
            return False

        identifier = intent.get("identifier")
        identifier_type = intent.get("identifier_type")
        if identifier is not None and identifier_type not in _VALID_IDENTIFIER_TYPES:
            return False
        if identifier_type is not None and identifier_type not in _VALID_IDENTIFIER_TYPES:
            return False

        return True

    async def analyze(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Resolves zoning + property-tax-roll data, then a permitted-use verdict."""
        location = intent["location"].strip()
        proposed_use = (intent.get("business_type") or intent.get("proposed_use") or "").strip() or None

        lat, lon = self._geo_rag.geocode_landmark(location)
        if lat is None or lon is None:
            return {"location": location, "error": f"Could not geocode '{location}'"}

        zoning = find_zone_at_location(self._geo_rag, location)

        identifier = intent.get("identifier")
        identifier_type = intent.get("identifier_type")
        if identifier is None:
            # No explicit identifier - fall back to the coordinates already
            # geocoded above rather than treating `location` as a literal
            # civic address, since get_property_tax_info's "adresse" mode is
            # an exact/fuzzy string match against role_foncier (not a
            # geocoded lookup - see SpatialHybridRAG.find_property_tax_record's
            # docstring) and `location` may be a landmark name instead.
            identifier = f"{lon},{lat}"
            identifier_type = "coordinates"

        property_info = get_property_tax_info(self._geo_rag, identifier, identifier_type)

        return {
            "location": location,
            "proposed_use": proposed_use,
            "zoning": zoning,
            "property": property_info,
            "use_permission": self._use_permission(zoning, proposed_use),
        }

    @staticmethod
    def _use_permission(zoning: dict[str, Any], proposed_use: Optional[str]) -> dict[str, Any]:
        """Checks proposed_use against the zone's permitted/conditional lists.

        Matching is the same accent-folded substring containment pareto_rank
        and count_businesses_in_da already use for category matching
        elsewhere in this codebase - not a controlled-vocabulary lookup, so
        a miss means "not found under this wording", not "prohibited".
        Verdict is "unknown" (never a false "not_listed") whenever the zone
        lookup itself failed or this ville has no usage-list data loaded -
        find_zone_at_location's docstring is explicit that silence must not
        be reported as "no restrictions".
        """
        if proposed_use is None:
            return {
                "verdict": "not_evaluated",
                "matched_use": None,
                "reason": "No proposed use given to check against the zone's permitted/conditional use lists.",
            }
        if "error" in zoning:
            return {"verdict": "unknown", "matched_use": None, "reason": zoning["error"]}
        if zoning.get("status") != "ok":
            return {
                "verdict": "unknown",
                "matched_use": None,
                "reason": zoning.get("message", "Usage data unavailable for this zone."),
            }

        folded_target = _fold_accents(proposed_use)
        for use in zoning.get("usages_permis") or []:
            if folded_target in _fold_accents(str(use)):
                return {"verdict": "permitted", "matched_use": use, "reason": None}
        for use in zoning.get("usages_conditionnels") or []:
            if folded_target in _fold_accents(str(use)):
                return {"verdict": "conditional", "matched_use": use, "reason": None}

        return {
            "verdict": "not_listed",
            "matched_use": None,
            "reason": (
                f"'{proposed_use}' n'apparaît pas explicitement dans les usages "
                f"permis ou conditionnels de la zone {zoning.get('zone_code')}."
            ),
        }
