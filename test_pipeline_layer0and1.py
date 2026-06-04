import pytest
import math
from main import (
    haversine_m,
    geocode_address,
    classify_question,
    _parse_top_n,
    _parse_type_filter,
    _parse_price,
    _apply_type_filter
)

# ==========================================
# LAYER 0: Component-Level (Pre-RAG)
# ==========================================

#0.1 Geocoder Tests
@pytest.mark.parametrize("address, expected_lat, expected_lon, max_error_m", [
    #random places with their ground truth coordinates 
    ("Palais des congrès, Montréal", 45.504, -73.559, 200),
    ("Marché Jean-Talon, Montréal", 45.536, -73.615, 200),
    ("Saint-Jean-sur-Richelieu", 45.305, -73.253, 5000), #rural area 
])
def test_geocoder_accuracy(address, expected_lat, expected_lon, max_error_m):
    lon, lat = geocode_address(address, context="Quebec, Canada")
    
    assert lon is not None and lat is not None, f"Geocoder failed gracefully, but null returned for {address}"
    
    distance = haversine_m(expected_lon, expected_lat, lon, lat)
    assert distance < max_error_m, f"Geocoding {address} missed by {distance:.2f}m (Target: <{max_error_m}m)"

def test_geocoder_graceful_failure():
    lon, lat = geocode_address("A_Completely_Fake_Address_That_Does_Not_Exist_12345")
    assert lon is None
    assert lat is None

#0.2 Router / Intent Classifier
@pytest.mark.parametrize("question, expected_intent", [
    # Spatial
    ("Quels sont les locaux les plus proches du Palais des congrès?", "spatial"),
    ("Trouve des bureaux à moins de 2km", "spatial"),
    # Filter
    ("Liste tous les locaux vacants", "filter"),
    ("Montre les locaux commerciaux", "filter"),
    # Price Inquiry
    ("Quels sont les bureaux les moins chers?", "price_inquiry"),
    ("Quel est le loyer moyen?", "price_inquiry"),
    # Qualitative
    ("Quel local est le plus adapté près du marché Atwater?", "qualitative"), # Override test
    ("Pourquoi ce secteur est-il recommandé?", "qualitative"),
    ("Donne-moi une analyse de la zone", "qualitative")
])
def test_intent_classifier(question, expected_intent):
    intent = classify_question(question)
    assert intent == expected_intent, f"Failed on: '{question}'"

# ==========================================
# LAYER 1: Structural Paths (Deterministic)
# ==========================================

#1.1 Path 1 - Spatial Parsing
@pytest.mark.parametrize("question, expected_n", [
    ("top 3 locaux", 3),
    ("les 5 plus proches", 5),
    ("give me 7", 7),
    ("les deux premiers", 2), 
    ("liste les locaux", 5), #Default fallback
])
def test_parse_top_n(question, expected_n):
    assert _parse_top_n(question, default=5) == expected_n

#1.2 Path 2 - Type Filters
@pytest.mark.parametrize("question, expected_filters", [
    ("bureaux à louer", ["bureau"]),
    ("espace professionnel et commercial", ["bureau", "commercial"]),
    ("entrepôt industriel", ["industriel"]),
    ("un bel appartement", ["résidentiel"]),
    ("un local sans type précis", []),
])
def test_parse_type_filter(question, expected_filters):
    filters = _parse_type_filter(question)
    assert set(filters) == set(expected_filters)

def test_apply_type_filter():
    mock_scored = [
        (100, {"Type": "bureau"}),
        (200, {"Type": "Commercial"}),
        (300, {"Type": "industriel, bureau"})
    ]
    
    #Test single filter
    res = _apply_type_filter(mock_scored, ["bureau"])
    assert len(res) == 2
    
    #Test empty filter 
    res = _apply_type_filter(mock_scored, [])
    assert len(res) == 3

#1.3 Path 2b - Price Parsing Robustness
@pytest.mark.parametrize("price_str, expected_float", [
    ("1 200$/mois", 1200.0),
    ("1,200", 1200.0), 
    ("1200.00", 1200.0),
    ("$1200", 1200.0),
    ("1 200,50", 1200.5), #Testing European decimal format
    ("sur demande", float("inf")),
    (None, float("inf")),
])
def test_parse_price(price_str, expected_float):
    assert _parse_price(price_str) == expected_float
