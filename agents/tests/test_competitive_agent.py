"""Unit tests for agents.competitive_agent.CompetitiveAgent.

Uses a fake geo_rag double (no MongoDB, no embeddings) so these run fast and
offline. The one piece of "already in place" logic worth exercising for
real rather than faking is SpatialHybridRAG.pareto_rank's match_status
tagging (spatial_rag_v1.py) - it does no I/O, so it's reused unmodified
here; only geocoding/candidate-fetching (real Mongo calls) are faked.

Full end-to-end verification against live MongoDB/NAICS embeddings follows
this repo's existing convention (see test_naics_classifier.py's docstring)
of a separate, manually-run integration script - not part of this file.
"""

import asyncio

import pytest
from langchain_core.documents import Document

from agents.competitive_agent import CompetitiveAgent
from spatial_rag_v1 import SpatialHybridRAG


def run(coro):
    return asyncio.run(coro)


def _business(naics_code: str, distance_m: float, label: str = "Business") -> Document:
    return Document(
        page_content=label,
        metadata={
            "id": label,
            "distance_m": distance_m,
            "lon": -73.6,
            "lat": 45.5,
            "source_collection": "quebec_businesses",
            "naics_code": naics_code,
        },
    )


def _vacant(distance_m: float, type_local: str = "commercial") -> Document:
    return Document(
        page_content="Local vacant",
        metadata={
            "id": "vacant",
            "distance_m": distance_m,
            "lon": -73.6,
            "lat": 45.5,
            "source_collection": "locaux_vacants",
            "type_local": type_local,
        },
    )


class FakeGeoRag:
    """Duck-typed stand-in for SpatialHybridRAG - no Mongo, no embeddings."""

    def __init__(self, businesses=None, vacant=None, das=None, geocode_result=(45.5, -73.6)):
        self.businesses = businesses or []
        self.vacant = vacant or []
        self.das = das if das is not None else []
        self.geocode_result = geocode_result
        self.geocode_calls = []

    def geocode_landmark(self, place_name):
        self.geocode_calls.append(place_name)
        if self.geocode_result is None:
            return (None, None)
        return self.geocode_result

    def hard_spatial_filter(self, collection_name, lon, lat, max_distance_meters, naics_prefix=None):
        if collection_name == "quebec_businesses":
            return list(self.businesses)
        if collection_name == "locaux_vacants":
            return list(self.vacant)
        return []

    def hybrid_semantic_search(self, query, spatial_docs, top_k=15):
        for doc in spatial_docs:
            doc.metadata.setdefault("semantic_score", 0.5)
        return spatial_docs[:top_k]

    # Real pareto_rank: pure computation over doc.metadata, no self.db access.
    pareto_rank = SpatialHybridRAG.pareto_rank

    def find_candidate_das(self, lon, lat, radius, limit=60):
        return list(self.das)

    def count_businesses_in_da(self, da_geometry, category_phrase=None, collection="quebec_businesses"):
        return sum(1 for b in self.businesses if category_phrase is None or category_phrase.lower() in b.page_content.lower())


DA_DOC = {
    "Geographie": 24540318,
    "Population_2021": 4200,
    "menages_total": 1800,
    "revenu_median": 52000,
    "age_moyen": 41.2,
    "geometry": {"type": "Polygon", "coordinates": [[[-73.61, 45.49], [-73.59, 45.49], [-73.59, 45.51], [-73.61, 45.51], [-73.61, 45.49]]]},
}


def _agent(**kwargs):
    return CompetitiveAgent(geo_rag=FakeGeoRag(**kwargs))


# ── validate_inputs ─────────────────────────────────────────────────────

def test_validate_rejects_missing_location():
    agent = _agent()
    assert run(agent.validate_inputs({"business_type": "restaurant"})) is False


def test_validate_rejects_missing_category():
    agent = _agent()
    assert run(agent.validate_inputs({"location": "Montreal"})) is False


def test_validate_rejects_non_positive_radius():
    agent = _agent()
    intent = {"location": "Montreal", "business_type": "restaurant", "radius_meters": -100}
    assert run(agent.validate_inputs(intent)) is False


def test_validate_accepts_naics_code_without_business_type():
    agent = _agent()
    intent = {"location": "Montreal", "naics_code": "7225"}
    assert run(agent.validate_inputs(intent)) is True


# ── analyze via naics_code path ─────────────────────────────────────────

def test_naics_path_classifies_direct_and_adjacent_competitors():
    businesses = [
        _business("722511", 300, "Exact close"),   # startswith target -> Exact NAICS
        _business("7225", 900, "Exact broad-code"),  # == target -> still Exact NAICS
        _business("72", 500, "Sector only"),         # target startswith doc -> Broad NAICS
        _business("454110", 100, "Unrelated"),       # no relation -> No Match
    ]
    geo_rag = FakeGeoRag(businesses=businesses)
    agent = CompetitiveAgent(geo_rag=geo_rag)

    result = run(agent.execute({"location": "Montreal", "naics_code": "7225", "radius_meters": 1500}))

    assert result["status"] == "success"
    data = result["data"]
    assert data["direct_competitor_count"] == 2
    assert data["adjacent_competitor_count"] == 1
    assert data["saturation_level"] == "low"
    assert data["nearest_competitor"]["id"] == "Exact close"
    assert [c["id"] for c in data["top_competitors"]] == ["Exact close", "Exact broad-code"]
    # locaux_vacants isn't searched on the naics_code path (see spatial_search's
    # own docstring) - this must read as "not searched", not a false zero.
    assert data["vacant_space_search_skipped"] is True
    assert data["vacant_space_count"] is None


def test_saturation_level_scales_with_direct_competitor_count():
    businesses = [_business("7225", 100 * i, f"b{i}") for i in range(8)]
    geo_rag = FakeGeoRag(businesses=businesses)
    agent = CompetitiveAgent(geo_rag=geo_rag)

    result = run(agent.execute({"location": "Montreal", "naics_code": "7225"}))

    assert result["data"]["direct_competitor_count"] == 8
    assert result["data"]["saturation_level"] == "high"


# ── analyze via business_type path (no naics_code, no/failed classifier) ──

def test_business_type_path_counts_vacant_spaces():
    businesses = [_business("7225", 200, "Cafe A")]
    vacant = [_vacant(150), _vacant(400)]
    geo_rag = FakeGeoRag(businesses=businesses, vacant=vacant)
    agent = CompetitiveAgent(geo_rag=geo_rag)  # no naics_classifier

    result = run(agent.execute({"location": "Montreal", "business_type": "cafe"}))

    assert result["status"] == "success"
    assert result["data"]["vacant_space_search_skipped"] is False
    assert result["data"]["vacant_space_count"] == 2


class FakeNaicsClassifier:
    def __init__(self, confidence: float, code: str = "7225", label: str = "Restaurants"):
        self._confidence = confidence
        self._code = code
        self._label = label

    def classify_business_type(self, query, top_k=1):
        return [{"code": self._code, "label": self._label, "confidence": self._confidence}]


def test_confident_naics_resolution_switches_to_naics_path():
    businesses = [_business("722511", 200, "Cafe A")]
    vacant = [_vacant(150)]
    geo_rag = FakeGeoRag(businesses=businesses, vacant=vacant)
    agent = CompetitiveAgent(geo_rag=geo_rag, naics_classifier=FakeNaicsClassifier(confidence=0.65))

    result = run(agent.execute({"location": "Montreal", "business_type": "cafe"}))

    data = result["data"]
    assert data["naics_code"] == "7225"
    assert data["naics_resolution"]["confidence"] == 0.65
    # naics_code now set -> vacant search skipped, same as the direct-naics test above.
    assert data["vacant_space_search_skipped"] is True


def test_low_confidence_naics_resolution_falls_back_to_business_type_search():
    businesses = [_business("722511", 200, "Cafe A")]
    geo_rag = FakeGeoRag(businesses=businesses)
    agent = CompetitiveAgent(geo_rag=geo_rag, naics_classifier=FakeNaicsClassifier(confidence=0.2))

    result = run(agent.execute({"location": "Montreal", "business_type": "cafe"}))

    data = result["data"]
    assert data["naics_code"] is None
    assert data["naics_resolution"]["confidence"] == 0.2
    assert data["vacant_space_search_skipped"] is False


# ── dissemination-area cross-check ──────────────────────────────────────

def test_da_saturation_included_when_area_found():
    businesses = [_business("7225", 200, "Cafe A"), _business("7225", 300, "Cafe B")]
    geo_rag = FakeGeoRag(businesses=businesses, das=[DA_DOC])
    agent = CompetitiveAgent(geo_rag=geo_rag)

    result = run(agent.execute({"location": "Montreal", "business_type": "cafe"}))

    da = result["data"]["dissemination_area_saturation"]
    assert da is not None
    assert da["da_code"] == 24540318
    assert da["business_count"] == 2


def test_da_saturation_is_none_when_no_area_found():
    businesses = [_business("7225", 200, "Cafe A")]
    geo_rag = FakeGeoRag(businesses=businesses, das=[])
    agent = CompetitiveAgent(geo_rag=geo_rag)

    result = run(agent.execute({"location": "Montreal", "business_type": "cafe"}))

    assert result["data"]["dissemination_area_saturation"] is None


# ── geocode failure ──────────────────────────────────────────────────────

def test_ungeocodable_location_returns_success_status_with_error_payload():
    """spatial_search reports a bad location as {"error": ...}, not an
    exception - execute() still sees a normal analyze() return, so this is
    "success" at the envelope level with an error payload inside "data",
    not "status": "error"."""
    geo_rag = FakeGeoRag(businesses=[], geocode_result=(None, None))
    agent = CompetitiveAgent(geo_rag=geo_rag)

    result = run(agent.execute({"location": "Nowhereville", "business_type": "cafe"}))

    assert result["status"] == "success"
    assert "error" in result["data"]
