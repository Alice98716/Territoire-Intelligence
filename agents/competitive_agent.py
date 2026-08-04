"""AGENT 3 : COMPÉTITIF — L'Analyste Concurrence.

Analyzes the competitive landscape and market saturation around a target
location. Built entirely on the spatial search/ranking pipeline that
already exists (tools.spatial_search -> SpatialHybridRAG.hard_spatial_filter
+ hybrid_semantic_search + pareto_rank, plus the dissemination-area lookup
and count_businesses_in_da) - this agent adds no new Mongo queries of its
own, only the competitive-landscape interpretation on top of results those
tools already return.
"""

from __future__ import annotations

from typing import Any, Optional

import logging
import math

from agents.base_agent import BaseAgent
from spatial_rag_v1 import NAICSClassifier, SpatialHybridRAG
from tools import count_businesses_in_da, find_dissemination_area, spatial_search

logger = logging.getLogger(__name__)

# Same first-pass floor, same caveat, as api_server.py's
# _NAICS_SEMANTIC_CONFIDENCE_FLOOR: NAICSClassifier.classify_business_type's
# own docstring measures ~0.6 for a good match, ~0.44 for plausible-but-
# wrong, ~0.30-0.35 for outright bad - there is no cleanly-separating
# threshold yet. Below this floor, we keep searching on the free-text
# business_type instead of trusting the resolved code.
_NAICS_CONFIDENCE_FLOOR = 0.4

# A walkable retail/service catchment, not a regional one - callers analyzing
# a different kind of business should pass radius_meters explicitly.
_DEFAULT_RADIUS_M = 1500.0

# Direct-competitor-count buckets are a judgment call, not a measured
# statistic - tuned for the default walkable radius above.
_SATURATION_THRESHOLDS = (
    (2, "low"),
    (6, "moderate"),
)
_SATURATION_FALLBACK = "high"

# pareto_rank (spatial_rag_v1.py) tags quebec_businesses results with one of
# these depending on which of naics_code/business_type spatial_search was
# given (target_naics takes priority over target_category when both are
# set - see pareto_rank's docstring/branching): "Exact NAICS"/"Broad NAICS"/
# "No Match" for a NAICS-code search, "Category Match"/"Semantic Rank Only"
# for a free-text business_type search. Direct/adjacent below picks the tier
# that means "same category", from whichever pair actually applied.
_DIRECT_MATCH_STATUSES = {"Exact NAICS", "Category Match"}
_ADJACENT_MATCH_STATUSES = {"Broad NAICS"}


class CompetitiveAgent(BaseAgent):
    """Competitive-landscape and market-saturation analysis for a location.

    Expects `intent` to look like:
        {
            "location": str,               # required, geocodable place name
            "business_type": str,          # free-text category (e.g. "restaurant")
            "naics_code": str,             # explicit NAICS code/prefix
            "radius_meters": float,        # optional, defaults to 1500m
        }
    At least one of business_type/naics_code is required - "competitors"
    only means something relative to a category.
    """

    def __init__(self, geo_rag: SpatialHybridRAG, naics_classifier: Optional[NAICSClassifier] = None) -> None:
        """Initializes the agent.

        Args:
            geo_rag: Shared spatial engine instance (same one api_server.py
                builds at startup) - not owned or connected here.
            naics_classifier: Optional shared NAICS classifier, used to
                resolve a free-text business_type to a NAICS code when the
                caller didn't supply one directly. Best-effort: if absent,
                or if it can't classify confidently, the agent falls back to
                spatial_search's own free-text category matching.
        """
        super().__init__(
            name="competitive",
            description="Analyse le paysage concurrentiel et la saturation du marché dans la zone cible.",
        )
        self._geo_rag = geo_rag
        self._naics_classifier = naics_classifier

    async def validate_inputs(self, intent: dict[str, Any]) -> bool:
        """Requires a non-empty location and at least one of business_type/naics_code."""
        location = intent.get("location")
        if not isinstance(location, str) or not location.strip():
            return False

        business_type = intent.get("business_type")
        naics_code = intent.get("naics_code")
        has_category = (isinstance(business_type, str) and business_type.strip()) or (
            isinstance(naics_code, str) and naics_code.strip()
        )
        if not has_category:
            return False

        radius_meters = intent.get("radius_meters")
        if radius_meters is not None and not (isinstance(radius_meters, (int, float)) and radius_meters > 0):
            return False

        return True

    async def analyze(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Runs the competitor search, then interprets it as a saturation read."""
        location = intent["location"].strip()
        radius_meters = float(intent.get("radius_meters") or _DEFAULT_RADIUS_M)
        business_type = (intent.get("business_type") or "").strip() or None
        naics_code = (intent.get("naics_code") or "").strip() or None

        naics_resolution = None
        if not naics_code and business_type:
            naics_resolution = self._resolve_naics(business_type)
            if naics_resolution and naics_resolution["confidence"] >= _NAICS_CONFIDENCE_FLOOR:
                naics_code = naics_resolution["code"]

        search = spatial_search(
            self._geo_rag,
            location,
            radius_meters=radius_meters,
            business_type=business_type,
            naics_code=naics_code,
        )
        if "error" in search:
            return {"location": location, "error": search["error"]}

        results = search["results"]
        businesses = [r for r in results if r.get("source_collection") == "quebec_businesses"]
        direct = sorted(
            (r for r in businesses if r.get("match_status") in _DIRECT_MATCH_STATUSES),
            key=lambda r: r.get("distance_m", radius_meters),
        )
        adjacent = [r for r in businesses if r.get("match_status") in _ADJACENT_MATCH_STATUSES]

        # naics_code searches restrict spatial_search to quebec_businesses
        # only (locaux_vacants has no NAICS field - see spatial_search's own
        # docstring), so vacant-space counts from that path would silently
        # read 0 rather than "not searched". Report that honestly instead of
        # guessing.
        naics_path_used = naics_code is not None
        vacant_space_count = None
        if not naics_path_used:
            vacant_space_count = sum(1 for r in results if r.get("source_collection") == "locaux_vacants")

        radius_km2 = math.pi * (radius_meters / 1000) ** 2
        density_per_km2 = round(len(direct) / radius_km2, 2) if radius_km2 else 0.0

        return {
            "location": location,
            "radius_meters": radius_meters,
            "naics_code": naics_code,
            "naics_resolution": naics_resolution,
            "direct_competitor_count": len(direct),
            "adjacent_competitor_count": len(adjacent),
            "competitor_density_per_km2": density_per_km2,
            "saturation_level": self._saturation_level(len(direct)),
            "nearest_competitor": direct[0] if direct else None,
            "top_competitors": direct[:10],
            "vacant_space_count": vacant_space_count,
            "vacant_space_search_skipped": naics_path_used,
            "dissemination_area_saturation": self._da_saturation(location, business_type, naics_resolution),
        }

    def _resolve_naics(self, business_type: str) -> Optional[dict[str, Any]]:
        """Best-effort free-text -> NAICS resolution. Never raises."""
        if self._naics_classifier is None:
            return None
        try:
            matches = self._naics_classifier.classify_business_type(business_type, top_k=1)
        except Exception:
            logger.exception("competitive: NAICS resolution failed for %r", business_type)
            return None
        if not matches:
            return None
        top = matches[0]
        return {"code": top["code"], "label": top.get("label"), "confidence": top.get("confidence", 0.0)}

    def _da_saturation(
        self, location: str, business_type: Optional[str], naics_resolution: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Boundary-aware cross-check: a category count within the actual
        dissemination area containing `location`, instead of the fixed-radius
        circle used above. Best-effort: returns None on any lookup failure
        rather than failing the whole analysis over a secondary metric.
        """
        try:
            area_lookup = find_dissemination_area(self._geo_rag, location, limit=1)
            areas = area_lookup.get("areas") or []
            if not areas:
                return None
            area = areas[0]
            geometry = area.get("geometry")
            if not geometry:
                return None

            category_phrase = business_type or (naics_resolution or {}).get("label")
            count_result = count_businesses_in_da(self._geo_rag, geometry, business_category=category_phrase)
            if "error" in count_result:
                return None

            return {
                "da_code": area.get("da_code"),
                "business_count": count_result.get("count"),
                "category_phrase": category_phrase,
            }
        except Exception:
            logger.exception("competitive: dissemination-area saturation lookup failed for %r", location)
            return None

    @staticmethod
    def _saturation_level(direct_competitor_count: int) -> str:
        for threshold, label in _SATURATION_THRESHOLDS:
            if direct_competitor_count <= threshold:
                return label
        return _SATURATION_FALLBACK
