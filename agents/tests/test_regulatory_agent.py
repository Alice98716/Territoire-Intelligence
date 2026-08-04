"""Unit tests for agents.regulatory_agent.RegulatoryAgent.

find_zone_at_location and get_property_tax_info (tools.py) each join across
several Mongo collections plus point-in-polygon geometry - not worth
re-deriving in a fake. Instead this patches those two functions at the
module boundary (monkeypatch) and only fakes geo_rag.geocode_landmark,
which RegulatoryAgent calls directly. Real, unmocked coverage of
find_zone_at_location/get_property_tax_info themselves belongs in a
manually-run integration script against live MongoDB, same convention as
test_naics_classifier.py.
"""

import asyncio

import agents.regulatory_agent as regulatory_agent
from agents.regulatory_agent import RegulatoryAgent


def run(coro):
    return asyncio.run(coro)


class FakeGeoRag:
    def __init__(self, geocode_result=(45.5, -73.6)):
        self.geocode_result = geocode_result

    def geocode_landmark(self, place_name):
        return self.geocode_result


ZONING_OK_BROMONT = {
    "zone_code": "C03",
    "ville": "Bromont",
    "usages_permis": ["Restaurants", "Commerce de détail"],
    "usages_conditionnels": ["Débit de boissons"],
    "status": "ok",
}

ZONING_USAGE_UNAVAILABLE = {
    "zone_code": "4052",
    "ville": "Saint-Hyacinthe",
    "usages_permis": None,
    "usages_conditionnels": None,
    "status": "usage_data_unavailable",
    "message": "Le zonage a été identifié (code 4052), mais les usages permis/conditionnels ne sont pas encore disponibles.",
}

ZONING_ERROR = {"error": "Aucune zone trouvée à cette adresse."}

PROPERTY_INFO = {
    "mat18": "681921118600010000",
    "adresse": "50 Rue JADE",
    "ville": "Bromont",
    "categorie": "Commercial",
    "cubf": "2000",
    "valeur_terrain": 120000,
    "valeur_totale": 480000,
    "valeur_batiment": 360000,
    "superficie_m2": 850,
    "imposable": True,
    "code_mun": "45015",
    "polygon_ring": None,
}


def _patch(monkeypatch, zoning=None, property_info=None):
    monkeypatch.setattr(regulatory_agent, "find_zone_at_location", lambda geo_rag, location: zoning)
    monkeypatch.setattr(
        regulatory_agent, "get_property_tax_info", lambda geo_rag, identifier, identifier_type: property_info
    )


# ── validate_inputs ─────────────────────────────────────────────────────

def test_validate_rejects_missing_location():
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())
    assert run(agent.validate_inputs({})) is False


def test_validate_accepts_location_alone():
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())
    assert run(agent.validate_inputs({"location": "50 Rue JADE, Bromont"})) is True


def test_validate_rejects_identifier_without_identifier_type():
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())
    intent = {"location": "Bromont", "identifier": "681921118600010000"}
    assert run(agent.validate_inputs(intent)) is False


def test_validate_rejects_unknown_identifier_type():
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())
    intent = {"location": "Bromont", "identifier": "x", "identifier_type": "postal_code"}
    assert run(agent.validate_inputs(intent)) is False


def test_validate_accepts_valid_identifier_pair():
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())
    intent = {"location": "Bromont", "identifier": "681921118600010000", "identifier_type": "matricule"}
    assert run(agent.validate_inputs(intent)) is True


# ── analyze: geocoding and identifier defaulting ────────────────────────

def test_ungeocodable_location_short_circuits_before_zoning_or_property_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr(regulatory_agent, "find_zone_at_location", lambda *a: calls.append("zoning"))
    monkeypatch.setattr(regulatory_agent, "get_property_tax_info", lambda *a: calls.append("property"))
    agent = RegulatoryAgent(geo_rag=FakeGeoRag(geocode_result=(None, None)))

    result = run(agent.execute({"location": "Nowhereville"}))

    assert result["status"] == "success"
    assert "error" in result["data"]
    assert calls == []


def test_default_identifier_uses_geocoded_coordinates(monkeypatch):
    captured = {}

    def fake_get_property_tax_info(geo_rag, identifier, identifier_type):
        captured["identifier"] = identifier
        captured["identifier_type"] = identifier_type
        return PROPERTY_INFO

    monkeypatch.setattr(regulatory_agent, "find_zone_at_location", lambda geo_rag, location: ZONING_OK_BROMONT)
    monkeypatch.setattr(regulatory_agent, "get_property_tax_info", fake_get_property_tax_info)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag(geocode_result=(45.5, -73.6)))

    run(agent.execute({"location": "downtown Bromont"}))

    assert captured["identifier_type"] == "coordinates"
    assert captured["identifier"] == "-73.6,45.5"


def test_explicit_identifier_overrides_default(monkeypatch):
    captured = {}

    def fake_get_property_tax_info(geo_rag, identifier, identifier_type):
        captured["identifier"] = identifier
        captured["identifier_type"] = identifier_type
        return PROPERTY_INFO

    monkeypatch.setattr(regulatory_agent, "find_zone_at_location", lambda geo_rag, location: ZONING_OK_BROMONT)
    monkeypatch.setattr(regulatory_agent, "get_property_tax_info", fake_get_property_tax_info)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    run(agent.execute({
        "location": "Bromont",
        "identifier": "681921118600010000",
        "identifier_type": "matricule",
    }))

    assert captured == {"identifier": "681921118600010000", "identifier_type": "matricule"}


def test_analyze_bundles_zoning_and_property_info(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_OK_BROMONT, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Bromont"}))

    assert result["status"] == "success"
    assert result["data"]["zoning"] == ZONING_OK_BROMONT
    assert result["data"]["property"] == PROPERTY_INFO


# ── use_permission verdicts ──────────────────────────────────────────────

def test_use_permission_not_evaluated_without_proposed_use(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_OK_BROMONT, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Bromont"}))

    assert result["data"]["use_permission"]["verdict"] == "not_evaluated"


def test_use_permission_permitted_matches_accent_and_case_insensitively(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_OK_BROMONT, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Bromont", "business_type": "restaurant"}))

    verdict = result["data"]["use_permission"]
    assert verdict["verdict"] == "permitted"
    assert verdict["matched_use"] == "Restaurants"


def test_use_permission_conditional_when_only_in_conditional_list(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_OK_BROMONT, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Bromont", "business_type": "débit de boissons"}))

    assert result["data"]["use_permission"]["verdict"] == "conditional"


def test_use_permission_not_listed_when_absent_from_both_lists(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_OK_BROMONT, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Bromont", "business_type": "usine chimique"}))

    assert result["data"]["use_permission"]["verdict"] == "not_listed"


def test_use_permission_unknown_when_usage_data_unavailable(monkeypatch):
    """Must never read as 'no restrictions' - find_zone_at_location's own
    docstring calls this out explicitly."""
    _patch(monkeypatch, zoning=ZONING_USAGE_UNAVAILABLE, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Saint-Hyacinthe", "business_type": "restaurant"}))

    assert result["data"]["use_permission"]["verdict"] == "unknown"


def test_use_permission_unknown_when_zoning_lookup_errors(monkeypatch):
    _patch(monkeypatch, zoning=ZONING_ERROR, property_info=PROPERTY_INFO)
    agent = RegulatoryAgent(geo_rag=FakeGeoRag())

    result = run(agent.execute({"location": "Unknown Address", "business_type": "restaurant"}))

    assert result["data"]["use_permission"]["verdict"] == "unknown"
