"""
Tool definitions for the agent, in the {name, description, input_schema} shape
Anthropic's tool-use API expects (pass TOOL_DEFINITIONS straight into the
`tools=` argument of client.messages.create).

Every wrapper below calls an EXISTING method on SpatialHybridRAG (spatial_rag_v1.py)
or an existing helper extracted from api_server.py - no new spatial/analysis logic
is implemented here. Spatial/demographic wrappers take `geo_rag` (a SpatialHybridRAG
instance) as their first argument rather than constructing their own, since
api_server.py already owns the single long-lived instance (one Mongo connection,
one embeddings model) built at startup.

One deviation from a literal wrap, flagged here because it changes the tool's
shape: run_analysis. There is no existing code path that computes competitive/
retention/siteScore/implementation metrics FROM a location - those metrics
(ISM, PAD, SPC, etc.) are computed entirely on the frontend and were only ever
narrated by generate_pillar_report() (extracted from chat_turn_2). So
run_analysis takes the metrics dicts as input, not a location - see its
docstring below.

parse_uploaded_document reads a file already saved to disk by api_server.py's
POST /api/upload-document endpoint and extracts {name, lat, lon, properties}
points from it - it does NOT turn those points into a map layer. There is no
"add this as a new map layer" tool here: the frontend's LayerKey type is a
closed union where every existing layer fetches its own data from its own
endpoint, with no slot for arbitrary uploaded data - extending that is a
separate frontend task, out of scope for this file.

find_zone_at_location is the other deviation: there is no existing
SpatialHybridRAG method for zoning, so the point-in-polygon test, the
per-ville property-key mapping, and the usages_zones join all live here as
private helpers (_point_in_polygon_ring, _ville_for_point,
_usages_zones_for_bromont_normalized) - the same pattern _flatten_polygon_ring
and _geojson_centroid already use elsewhere in this file for geometry logic
that has no other home. It reads geo_rag.db directly (geo_layers, donnees,
cities, usages_zones) rather than adding one-off single-purpose methods to
SpatialHybridRAG for a tool that's the only caller. Confirmed against the
live database before writing this: geo_layers has exactly one zonage
document per ville (Bromont/Saint-Hyacinthe/Québec), every feature in all
three is a Polygon with a single ring (no holes, no MultiPolygon), and
usages_zones currently only has data for Bromont - Saint-Hyacinthe and
Québec resolve a zone_code but always fall into the usage_data_unavailable
branch.
"""

import json
import math
from typing import Optional

from spatial_rag_v1 import SpatialHybridRAG, _fold_accents

# Fields pulled from get_smart_demographics() when a tool caller doesn't need
# anything more specific - same default set api_server.py's compare block uses.
DEFAULT_DEMOGRAPHIC_FIELDS = ["Population_2021", "menages_total", "revenu_median", "age_moyen"]

# Collections searched by spatial_search when the caller names no specific one -
# matches query_pipeline's "both" behavior in spatial_rag_v1.py.
SPATIAL_SEARCH_COLLECTIONS = ["locaux_vacants", "quebec_businesses"]

# Defaults matching the cross-query pipeline in api_server.py (CROSS_QUERY_SEARCH_RADIUS_M,
# CROSS_QUERY_DA_CANDIDATE_LIMIT), since find_candidate_das requires both explicitly.
DEFAULT_DA_SEARCH_RADIUS_M = 8000
DEFAULT_DA_CANDIDATE_LIMIT = 60

# Column-name candidates parse_uploaded_document looks for in a CSV/Excel
# file, matched case- and accent-insensitively (via _fold_accents, same fold
# pareto_rank already uses for category matching).
_LAT_COLUMN_CANDIDATES = ["lat", "latitude", "y"]
_LON_COLUMN_CANDIDATES = ["lon", "lng", "long", "longitude", "x"]
_NAME_COLUMN_CANDIDATES = ["name", "nom", "label", "client", "title"]
_ADDRESS_COLUMN_CANDIDATES = ["address", "adresse", "location", "emplacement"]

# find_zone_at_location: which geojson.features[i].properties key holds the
# zone code, per ville - confirmed against the live geo_layers data, where
# each of the 3 supported villes uses a different source field. Listed as a
# tuple so a ville can name fallback keys, in priority order; Saint-Hyacinthe
# is the only one that currently needs one (NUM_ZONE is a bare code like
# "4052", ETIQUETTE is "4052  C03" - checked across all 1091 of its features
# and both are always populated, but NUM_ZONE is kept primary since it's the
# cleaner join key and ETIQUETTE isn't guaranteed to stay that way).
_ZONAGE_PROPERTY_KEYS_BY_VILLE = {
    "Bromont": ("ZONAGE",),
    "Saint-Hyacinthe": ("NUM_ZONE", "ETIQUETTE"),
    "Québec": ("ID",),
}


# ═══════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — Anthropic tool-use format
# ═══════════════════════════════════════════════════════════════════════════
TOOL_DEFINITIONS = [
    {
        "name": "geocode",
        "description": (
            "Resolve a place name, address, or landmark (e.g. 'Saint-Hyacinthe train "
            "station', 'downtown Bromont') into latitude/longitude coordinates. Use this "
            "FIRST whenever another tool needs coordinates and you only have text - "
            "spatial_search, get_demographics, and find_dissemination_area all geocode "
            "internally, but call this directly when you need the coordinates themselves "
            "(e.g. to then draw a polygon). Do not use this to re-resolve a location the "
            "user already disambiguated in a prior turn - reuse those coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "place_name": {
                    "type": "string",
                    "description": "Landmark, address, or place name to resolve, e.g. "
                                   "'Saint-Hyacinthe city center' or '123 Rue Principale, Bromont'.",
                }
            },
            "required": ["place_name"],
        },
    },
    {
        "name": "spatial_search",
        "description": (
            "Find the closest matching businesses and/or vacant commercial spaces around "
            "a location, ranked by a blend of distance and semantic/category relevance. "
            "Use this when the user asks for 'the closest X near Y', 'vacant offices "
            "within Z meters of Y', or names a specific NAICS code or business type to "
            "search for around a point. Do NOT use this when the search area is an "
            "arbitrary drawn polygon (a trade-area/isochrone) rather than a point+radius "
            "circle - use polygon_filter for that instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Place name or address to search around. Geocoded automatically.",
                },
                "radius_meters": {
                    "type": "number",
                    "default": 5000,
                    "description": "Search radius in meters. Defaults to 5000 if omitted.",
                },
                "business_type": {
                    "type": "string",
                    "description": "Free-text category to boost matches on (semantic ranking, "
                                   "not a hard filter), e.g. 'yoga studio', 'vacant office'. Omit "
                                   "if not filtering by type. Prefer naics_code instead whenever "
                                   "you know it - it's an exact filter, this is a soft ranking hint.",
                },
                "naics_code": {
                    "type": "string",
                    "description": "Explicit NAICS code to filter businesses to, e.g. '722' "
                                   "(restaurants) or '6212' (dentists) - matches any business "
                                   "whose own NAICS code falls under this one, at any level of "
                                   "the hierarchy. Use this (not business_type) for 'show me the "
                                   "businesses in category X' - it's an actual database filter, "
                                   "not a ranking boost, so it returns every matching business in "
                                   "range rather than just the closest/best-scoring few. Omit if "
                                   "the user didn't name or imply a specific category.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "polygon_filter",
        "description": (
            "Find businesses (or vacant spaces) whose location falls strictly inside an "
            "arbitrary polygon - typically the trade-area/isochrone (zone de chalandise) "
            "already drawn on the map dashboard. Use this instead of spatial_search "
            "whenever the search area is a real polygon rather than a point+radius circle, "
            "e.g. 'what's inside the zone I just drew', or any competitive/catchment-area "
            "analysis anchored to that drawn shape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "polygon_ring": {
                    "type": "array",
                    "description": "Closed GeoJSON ring: a list of [lon, lat] point pairs, "
                                   "first point repeated as the last, at least 4 entries.",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "business_type": {
                    "type": "string",
                    "description": "Optional free-text category to filter results to, e.g. "
                                   "'restaurant'. Matches accent-insensitively against each "
                                   "result's label.",
                },
                "collection": {
                    "type": "string",
                    "enum": ["quebec_businesses", "locaux_vacants"],
                    "default": "quebec_businesses",
                    "description": "Which collection to search inside the polygon.",
                },
            },
            "required": ["polygon_ring"],
        },
    },
    {
        "name": "compare_locations",
        "description": (
            "Run the same spatial_search look-up independently for two or more named "
            "locations, so their results can be placed side by side. Use this when the "
            "user explicitly wants to compare 2+ places for the same business type or "
            "NAICS code (e.g. 'compare Bromont and Saint-Hyacinthe for a bakery', 'which "
            "is better, X or Y'). For a single-location query, use spatial_search directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "2 or more place names to compare, e.g. ['Bromont', 'Saint-Hyacinthe'].",
                },
                "radius_meters": {
                    "type": "number",
                    "default": 5000,
                    "description": "Search radius in meters applied to each location.",
                },
                "business_type": {
                    "type": "string",
                    "description": "Free-text category to boost on for every location (semantic "
                                   "ranking, not a hard filter), e.g. 'bakery'.",
                },
                "naics_code": {
                    "type": "string",
                    "description": "Explicit NAICS code to filter every location's results to - "
                                   "see spatial_search's naics_code for the containment/matching "
                                   "rule. Prefer this over business_type when the category is a "
                                   "known NAICS code.",
                },
            },
            "required": ["location_names"],
        },
    },
    {
        "name": "run_analysis",
        "description": (
            "Write a territorial-intelligence report (competitive, retention, site-score, "
            "implementation, or all four pillars) narrating metrics the map dashboard has "
            "ALREADY computed. IMPORTANT: this tool does not compute ISM/PAD/SPC/etc. from "
            "a location - no backend code does that today, those metrics are computed by "
            "the frontend. Only call this once the requested panel(s) have finished "
            "calculating on screen and their metrics can be passed in below. Use "
            "panel='all' for a full 4-pillar report, or a single panel name when the user "
            "asked for just one (e.g. 'analyze the competition at X' -> panel='competitive', "
            "with advanced_metrics filled in from what's on screen for X)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "panel": {
                    "type": "string",
                    "enum": ["competitive", "retention", "siteScore", "implementation", "all"],
                    "description": "Which pillar to report on, or 'all' for the full 4-pillar report.",
                },
                "advanced_metrics": {
                    "type": "object",
                    "description": "Competitive-pillar metrics already computed by the frontend "
                                   "(e.g. saturation_marche_ism, distance_concurrent_proche_dcp). "
                                   "Needed if panel is 'competitive' or 'all'.",
                },
                "site_score_metrics": {
                    "type": "object",
                    "description": "Site-score-pillar metrics already computed by the frontend "
                                   "(e.g. score_global, taux_vacance). Needed if panel is "
                                   "'siteScore' or 'all'.",
                },
                "retention_metrics": {
                    "type": "object",
                    "description": "Retention-pillar metrics already computed by the frontend "
                                   "(e.g. pouvoir_achat_disponible_pad, diversite_shannon). "
                                   "Needed if panel is 'retention' or 'all'.",
                },
                "implementation_metrics": {
                    "type": "object",
                    "description": "Implementation-pillar metrics already computed by the "
                                   "frontend (e.g. score_composite_spc). Needed if panel is "
                                   "'implementation' or 'all'.",
                },
                "population": {"description": "Population of the trade area."},
                "households": {"description": "Household count of the trade area."},
                "median_income": {"description": "Median household income of the trade area."},
                "avg_age": {"description": "Average age in the trade area."},
                "businesses": {
                    "type": "array",
                    "description": "Nearby business dicts (each with a 'sector' or 'category' "
                                   "key) used for the competitive sector breakdown.",
                },
            },
            "required": ["panel"],
        },
    },
    {
        "name": "get_demographics",
        "description": (
            "Fetch aggregated census demographics (population, households, median income, "
            "average age) for the area around a location - a whole city when the location "
            "matches a known city name, otherwise a radius around the geocoded point. Use "
            "this when the user asks about population, income, household count, or age for "
            "a place, without needing a full competitive/site analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or place/address to fetch demographics for.",
                },
                "radius_meters": {
                    "type": "number",
                    "default": 5000,
                    "description": "Radius in meters around the geocoded point, used only "
                                   "when 'location' isn't a recognized city name.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "find_dissemination_area",
        "description": (
            "Find StatCan dissemination areas (small census geography polygons) near a "
            "location, nearest first, each carrying its own demographic fields. Use this "
            "as the FIRST step of a cross-collection query like 'areas near X with high "
            "income but few restaurants' - it gives you candidate polygons to then rank by "
            "demographics and, via count_businesses_in_da, by business density. Use "
            "get_dissemination_area_by_code instead when the user names a SPECIFIC DA by "
            "its exact code rather than describing a place/point to search near."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Place name or address to search around.",
                },
                "radius_meters": {
                    "type": "number",
                    "default": DEFAULT_DA_SEARCH_RADIUS_M,
                    "description": "Search radius in meters.",
                },
                "limit": {
                    "type": "integer",
                    "default": DEFAULT_DA_CANDIDATE_LIMIT,
                    "description": "Max number of candidate dissemination areas to return, nearest first.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_dissemination_area_by_code",
        "description": (
            "Look up a single dissemination area by its EXACT code (e.g. '24540318'), "
            "returning its boundary as a flat polygon_ring plus a few key demographic "
            "fields - not a proximity search, and not the raw database record. Use this "
            "whenever the user names a specific DA/zone by its code rather than describing "
            "a point to search near (e.g. 'how many businesses are in DA 24540318', 'list "
            "the businesses in dissemination area 24540173', 'count properties in zone "
            "24010024'). Pass the returned 'polygon_ring' straight into polygon_filter or "
            "count_businesses_in_da - both accept this exact shape. Use find_dissemination_area "
            "instead when the user describes a place/point and wants the nearest DA(s), not "
            "an already-known code. Returns the main geographic boundary. If the DA consists "
            "of multiple disconnected parts (rare - about 1 in 15,000), only the first/primary "
            "part is returned and a non-null 'warning' field explains what was excluded - "
            "relay that warning to the user rather than reporting a count/list as complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "da_code": {
                    "type": "string",
                    "description": "The dissemination area's exact code, e.g. '24540318'.",
                },
            },
            "required": ["da_code"],
        },
    },
    {
        "name": "count_businesses_in_da",
        "description": (
            "Count how many businesses (or vacant locals, via the collection param) fall "
            "inside a given dissemination-area polygon, optionally restricted to a free-text "
            "category (e.g. 'restaurant', 'café', or 'bureau'/'commercial' for vacant locals). "
            "Use this right after find_dissemination_area or get_dissemination_area_by_code, "
            "once you have a polygon to count things inside - not for open-ended radius "
            "searches, which spatial_search already covers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "da_geometry": {
                    "description": "Either a full GeoJSON Polygon geometry object (from "
                                   "find_dissemination_area's 'geometry' field) or a flat "
                                   "polygon_ring array of [lon, lat] pairs (from "
                                   "get_dissemination_area_by_code's 'polygon_ring' field). "
                                   "Both shapes are accepted.",
                },
                "business_category": {
                    "type": "string",
                    "description": "Optional free-text category to restrict the count to. "
                                   "Omit to count all matching documents in the polygon.",
                },
                "collection": {
                    "type": "string",
                    "enum": ["quebec_businesses", "locaux_vacants"],
                    "default": "quebec_businesses",
                    "description": "Which collection to count inside the polygon - businesses "
                                   "or vacant locals.",
                },
            },
            "required": ["da_geometry"],
        },
    },
    {
        "name": "get_property_tax_info",
        "description": (
            "Look up a single property's tax-roll record (assessed land/building/total "
            "value, use code, lot size, municipality) by its exact matricule number, its "
            "address, or its coordinates. Use this when the user asks about a SPECIFIC "
            "property's assessed value, use classification, or lot size (e.g. 'what's the "
            "assessed value of matricule 681921118600010000', 'what's the property tax info "
            "for 50 Rue JADE'). Also returns polygon_ring: the property's boundary shape as "
            "a flat [[lon, lat], ...] ring, when available (currently Saint-Hyacinthe and "
            "Bromont only) - null otherwise. For Bromont, the 3 assessed-value fields come "
            "from a cross-checked, confirmed-authoritative source rather than role_foncier's "
            "own (occasionally stale) copy. This does NOT return owner/proprietor information "
            "- that data does not exist in this system (confirmed: Quebec's public rôle "
            "d'évaluation data excludes ownership for privacy)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "The value to look up: an 18-digit matricule (e.g. "
                                   "'681921118600010000'), a street address (e.g. '50 Rue "
                                   "JADE'), or 'lon,lat' coordinates (e.g. '-72.691,45.309') "
                                   "- matching whichever identifier_type is given.",
                },
                "identifier_type": {
                    "type": "string",
                    "enum": ["matricule", "adresse", "coordinates"],
                    "description": "Which kind of identifier is being passed. 'matricule' is "
                                   "the most reliable - it's always populated and unique.",
                },
            },
            "required": ["identifier", "identifier_type"],
        },
    },
    {
        "name": "find_zone_at_location",
        "description": (
            "Look up the zoning (zonage) at a specific address or point: the zone code and, "
            "where available, its permitted (usages_permis) and conditional (usages_conditionnels) "
            "uses. Use this when the user asks what's allowed to be built/operated at a specific "
            "address (e.g. 'what can I build at 123 Rue Principale, Bromont', 'is this address "
            "zoned commercial'). Currently only covers Bromont, Saint-Hyacinthe, and Québec - "
            "other municipalities return an error. Even within those three, permitted/conditional "
            "use data currently only exists for Bromont; Saint-Hyacinthe and Québec will still "
            "return the zone code but with a status flagging that usage data isn't loaded yet - "
            "relay that status to the user rather than reporting silence as 'no restrictions'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Address or place name to look up the zoning for.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "parse_uploaded_document",
        "description": (
            "Extract location data (name, coordinates, and any other columns/properties) "
            "from a file the user has uploaded - CSV, GeoJSON, or Excel. Use this as the "
            "FIRST step whenever the user references an uploaded document/file (e.g. "
            "'update the map with the document I provided', 'plot the addresses in this "
            "spreadsheet'). Rows/features that only have an address (no lat/lon) are "
            "geocoded automatically. This tool only extracts and returns the points - it "
            "does not add anything to the map itself, since there is currently no map "
            "layer for arbitrary uploaded data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Server-side path of the previously uploaded file, as "
                                   "returned by POST /api/upload-document.",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["csv", "geojson", "excel"],
                    "description": "Format of the uploaded file.",
                },
            },
            "required": ["file_path", "file_type"],
        },
    },
    {
        "name": "toggle_layer",
        "description": (
            "Show or hide one of the map's data layers in the React dashboard. Use this "
            "when the user asks to see/hide a layer with no actual search or analysis "
            "attached, e.g. 'show me the vacant locals layer'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_name": {
                    "type": "string",
                    "enum": ["commerces", "locauxVacants", "statcanDA", "zonage"],
                    "description": "Which map layer to toggle.",
                }
            },
            "required": ["layer_name"],
        },
    },
    {
        "name": "set_filter",
        "description": (
            "Apply a sector or NAICS-code filter to what's currently shown on the map, "
            "without running a new search. Use this for a follow-up like 'now just show "
            "me restaurants' or 'filter to NAICS 722' on results already on screen - for a "
            "fresh search anchored to a location, use spatial_search instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Free-text sector/category to filter the map to, e.g. 'restaurants'.",
                },
                "naics_code": {
                    "type": "string",
                    "description": "Explicit NAICS code to filter the map to, e.g. '722'.",
                },
            },
        },
    },
    {
        "name": "set_basemap",
        "description": (
            "Switch the map's visual basemap style (e.g. satellite vs. street view). Use "
            "this only for a pure display-preference request ('switch to satellite view') "
            "with no data or location implication."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "basemap_type": {
                    "type": "string",
                    "description": "Basemap style to switch to (e.g. 'streets', 'satellite', "
                                   "'light', 'dark') - the accepted values depend on what the "
                                   "frontend's map component supports; this backend does not "
                                   "validate against a fixed list.",
                }
            },
            "required": ["basemap_type"],
        },
    },
    {
        "name": "draw_isochrone",
        "description": (
            "Draw a travel-time isochrone (a 'zone de chalandise') of N minutes by a given "
            "travel mode around a location, without necessarily running a search yet. Use "
            "this when the user specifies an explicit time/distance zone ('within 10 "
            "minutes of X', 'a 15-minute walking zone around Y') and the immediate need is "
            "just to draw that shape - pair it with spatial_search or polygon_filter "
            "afterward if they also want results inside it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Place name or address to center the isochrone on.",
                },
                "minutes": {
                    "type": "integer",
                    "default": 10,
                    "description": "Travel time in minutes.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["driving", "walking", "cycling"],
                    "default": "driving",
                    "description": "Travel mode for the isochrone.",
                },
            },
            "required": ["location"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# WRAPPER FUNCTIONS — each calls an existing method, no new logic
# ═══════════════════════════════════════════════════════════════════════════

def geocode(geo_rag: SpatialHybridRAG, place_name: str) -> dict:
    """Wraps SpatialHybridRAG.geocode_landmark."""
    lat, lon = geo_rag.geocode_landmark(place_name)
    return {"found": lat is not None and lon is not None, "lat": lat, "lon": lon}


# ALLOWLIST: id/distance_m/lon/lat/label/source_collection/match_status only,
# picked explicitly off d.metadata rather than passed through - see the fuller
# rationale above polygon_filter below (same _doc_from_mongo-curated metadata,
# same two-layer reasoning), which returns a different field subset
# (nom/adresse/secteur/code_naics) of that same underlying metadata dict.
def spatial_search(
    geo_rag: SpatialHybridRAG,
    location: str,
    radius_meters: float = 5000,
    business_type: Optional[str] = None,
    naics_code: Optional[str] = None,
) -> dict:
    """Wraps hard_spatial_filter + hybrid_semantic_search + pareto_rank - the
    same combo query_pipeline runs - searching both collections by default
    (query_pipeline's "both" behavior when nothing narrows the target).

    When naics_code is given, this is a "show me businesses in category X"
    request, not an open-ended relevance search - hard_spatial_filter pushes
    the NAICS match into the Mongo query itself (see its docstring) and the
    expensive hybrid_semantic_search re-embedding pass is skipped entirely,
    since every surviving candidate already matches the requested category;
    ranking is by distance + the same NAICS-containment boost pareto_rank
    always applies. locaux_vacants has no NAICS field at all, so it's
    skipped rather than fetched and then discarded."""
    lat, lon = geo_rag.geocode_landmark(location)
    if lat is None or lon is None:
        return {"error": f"Could not geocode '{location}'", "results": []}

    collections_to_search = ["quebec_businesses"] if naics_code else SPATIAL_SEARCH_COLLECTIONS
    candidates = []
    for collection in collections_to_search:
        candidates.extend(geo_rag.hard_spatial_filter(collection, lon, lat, radius_meters, naics_prefix=naics_code))

    if not candidates:
        return {"results": []}

    if naics_code:
        ranked = candidates
    else:
        query_text = business_type or location
        ranked = geo_rag.hybrid_semantic_search(query=query_text, spatial_docs=candidates, top_k=len(candidates))
    ranked = geo_rag.pareto_rank(ranked, radius_meters, target_naics=naics_code, target_category=business_type)

    return {
        "results": [
            {
                "id": d.metadata.get("id"),
                "distance_m": round(d.metadata.get("distance_m", 0)),
                "lon": d.metadata.get("lon"),
                "lat": d.metadata.get("lat"),
                "label": d.page_content,
                "source_collection": d.metadata.get("source_collection"),
                "match_status": d.metadata.get("match_status", "Standard Rank"),
            }
            for d in ranked
        ]
    }


# ALLOWLIST (polygon_filter, spatial_search): id/lon/lat/label/nom/adresse/
# secteur/code_naics only - enforced twice over. _doc_from_mongo
# (spatial_rag_v1.py) already builds a curated `metadata` dict per document
# instead of handing back the raw quebec_businesses/locaux_vacants doc, so
# there's no _id, no raw Overture Maps `names`/`categories`/`addresses`
# sub-objects, and no other collection-specific field ever reaches `.metadata`
# in the first place. The explicit `.get(...)` picklist below is the second
# layer: even if _doc_from_mongo's metadata dict grows a new key later, it
# doesn't automatically appear in a tool_result without a deliberate line
# added here.
def polygon_filter(
    geo_rag: SpatialHybridRAG,
    polygon_ring: list,
    business_type: Optional[str] = None,
    collection: str = "quebec_businesses",
) -> dict:
    """Wraps hard_polygon_filter. business_type applies the same accent-folded
    substring match pareto_rank uses for category boosting elsewhere, since
    hard_polygon_filter itself has no ranking/filtering of its own.

    Same field shape as api_server.py's spatial_search results_to_send
    (nom/adresse/secteur/code_naics alongside id/lon/lat/label) - both read
    off the same _doc_from_mongo metadata, so a caller consuming either
    doesn't need two different shapes for what is otherwise the same data."""
    docs = geo_rag.hard_polygon_filter(collection, polygon_ring)
    if business_type:
        folded_target = _fold_accents(business_type)
        docs = [d for d in docs if folded_target in _fold_accents(d.page_content)]
    return {
        "results": [
            {
                "id": d.metadata.get("id"),
                "lon": d.metadata.get("lon"),
                "lat": d.metadata.get("lat"),
                "label": d.page_content,
                "nom": d.metadata.get("nom"),
                "adresse": d.metadata.get("adresse"),
                "secteur": d.metadata.get("secteur"),
                "code_naics": d.metadata.get("naics_code") or None,
            }
            for d in docs
        ]
    }


def compare_locations(
    geo_rag: SpatialHybridRAG,
    location_names: list,
    radius_meters: float = 5000,
    business_type: Optional[str] = None,
    naics_code: Optional[str] = None,
) -> dict:
    """Wraps calling spatial_search once per location."""
    return {
        "locations": [
            {"name": name, **spatial_search(geo_rag, name, radius_meters, business_type, naics_code)}
            for name in location_names
        ]
    }


def run_analysis(
    panel: str,
    advanced_metrics: Optional[dict] = None,
    site_score_metrics: Optional[dict] = None,
    retention_metrics: Optional[dict] = None,
    implementation_metrics: Optional[dict] = None,
    population=0,
    households="Non disponible",
    median_income="Non disponible",
    avg_age="Non disponible",
    businesses: Optional[list] = None,
) -> dict:
    """Wraps generate_pillar_report (api_server.py), extracted from the block-
    building code that used to live inline in chat_turn_2. Deferred import:
    api_server.py imports TOOL_DEFINITIONS from this module at startup, so
    importing api_server at module load time here would be circular - by the
    time this function actually runs, api_server has already finished
    importing, so this resolves fine."""
    from api_server import generate_pillar_report

    analysis_scope = "full" if panel == "all" else "single"
    final_text = generate_pillar_report(
        advanced_metrics=advanced_metrics or {},
        site_score=site_score_metrics or {},
        retention=retention_metrics or {},
        implementation=implementation_metrics or {},
        population=population,
        households=households,
        median_income=median_income,
        avg_age=avg_age,
        businesses=businesses or [],
        analysis_scope=analysis_scope,
        focus_panel=None if panel == "all" else panel,
    )
    return {"final_text": final_text}


def get_demographics(geo_rag: SpatialHybridRAG, location: str, radius_meters: float = 5000) -> dict:
    """Wraps get_smart_demographics."""
    lat, lon = geo_rag.geocode_landmark(location)
    if lat is None or lon is None:
        return {"error": f"Could not geocode '{location}'"}
    docs = geo_rag.get_smart_demographics(location, lat, lon, radius_meters, DEFAULT_DEMOGRAPHIC_FIELDS)
    if not docs:
        return {"metrics": None}
    return {"metrics": docs[0].metadata.get("metrics"), "summary": docs[0].page_content}


# find_candidate_das (spatial_rag_v1.py) returns raw "donnees" Mongo docs -
# each one carries a non-JSON-serializable _id, an sdrCode used only
# internally to join against the cities collection (see _ville_for_point
# above), and whatever other StatCan census columns that collection happens
# to have beyond the four this app actually uses anywhere (DEFAULT_DEMOGRAPHIC_
# FIELDS). None of that belongs in a tool_result handed to Claude: _id/sdrCode
# are internal plumbing with no meaning to the model, and the wider census
# field set is excluded for the same reason get_dissemination_area_by_code
# below limits itself to four named fields rather than the whole document -
# one consistent, deliberately narrow demographic vocabulary across both DA
# tools, not whatever columns the collection happens to contain. "geometry"
# is the one raw field kept as-is (renamed fields aside) because
# count_businesses_in_da/_as_polygon_geometry and api_server.py's
# _last_resolved_polygon fallback both chain directly off of it.
_DA_CANDIDATE_FIELD_MAP = {
    "Geographie": "da_code",
    "Population_2021": "population",
    "menages_total": "num_households",
    "revenu_median": "median_income",
    "age_moyen": "avg_age",
}


def _safe_da_candidate(doc: dict) -> dict:
    safe = {out_key: doc.get(raw_key) for raw_key, out_key in _DA_CANDIDATE_FIELD_MAP.items()}
    safe["geometry"] = doc.get("geometry")
    return safe


def find_dissemination_area(
    geo_rag: SpatialHybridRAG,
    location: str,
    radius_meters: float = DEFAULT_DA_SEARCH_RADIUS_M,
    limit: int = DEFAULT_DA_CANDIDATE_LIMIT,
) -> dict:
    """Wraps find_candidate_das. Each area is filtered through
    _DA_CANDIDATE_FIELD_MAP (da_code/population/num_households/median_income/
    avg_age) plus geometry - NOT the raw Mongo doc find_candidate_das itself
    returns (see that allowlist's comment for why)."""
    lat, lon = geo_rag.geocode_landmark(location)
    if lat is None or lon is None:
        return {"error": f"Could not geocode '{location}'", "areas": []}
    raw_areas = geo_rag.find_candidate_das(lon, lat, radius_meters, limit)
    return {"areas": [_safe_da_candidate(a) for a in raw_areas]}


def _flatten_polygon_ring(geometry: Optional[dict]) -> Optional[list]:
    """Extract a GeoJSON Polygon/MultiPolygon's outer ring as a flat list of
    [lon, lat] pairs. Polygon nests as coordinates[ring][point], so
    coordinates[0] IS the outer ring; MultiPolygon nests one level deeper
    (coordinates[polygon][ring][point]), so it needs coordinates[0][0]
    instead. Verified against the live donnees collection rather than
    assumed: 13805 docs are Polygon, 1 is MultiPolygon, 1382 have no
    geometry at all - the last case is why this returns None instead of
    raising when geometry/coordinates is missing or malformed."""
    if not geometry or "coordinates" not in geometry:
        return None
    coords = geometry["coordinates"]
    try:
        ring = coords[0][0] if geometry.get("type") == "MultiPolygon" else coords[0]
        return [[pt[0], pt[1]] for pt in ring]
    except (KeyError, IndexError, TypeError):
        return None


# ALLOWLIST: da_code/geometry_type/warning/polygon_ring/population/
# num_households/median_income/avg_age only. get_da_by_code returns the raw
# "donnees" doc - same _id/sdrCode/wider-census-column concerns as
# find_candidate_das above - so this hand-picks the same four demographic
# fields (population/num_households/median_income/avg_age) rather than the
# full document, and the docstring below covers why raw geometry itself is
# excluded in favor of polygon_ring.
def get_dissemination_area_by_code(geo_rag: SpatialHybridRAG, da_code: str) -> dict:
    """Wraps get_da_by_code. Does NOT return the raw MongoDB document - only
    a flat polygon_ring (ready to pass straight into polygon_filter or
    count_businesses_in_da, see _as_polygon_geometry below) plus a handful
    of named demographic fields for context. Deliberately omits the raw
    geometry/coordinates blob alongside polygon_ring so there's no ambiguity
    about which field to use.

    A rare (1 of 15,188 DAs, confirmed against the live collection) but real
    case: some DAs are stored as MultiPolygon with genuinely disconnected
    parts - e.g. DA 24980071 is a mainland area (394-point ring) plus a
    separate 5-point offshore islet, not one shape arbitrarily wrapped in
    MultiPolygon format. _flatten_polygon_ring only ever keeps the FIRST
    part (coordinates[0][0]) - any additional disconnected parts are
    excluded from polygon_ring and therefore from anything chained off of it
    (counts, business lists). Rather than drop that silently, this includes
    a "warning" field whenever geometry_type is "MultiPolygon", so Claude
    sees it in the tool_result and can relay it to the user instead of
    reporting an undercount with no caveat."""
    area = geo_rag.get_da_by_code(da_code)
    if area is None:
        return {"error": f"No dissemination area found with code '{da_code}'"}

    geometry = area.get("geometry")
    polygon_ring = _flatten_polygon_ring(geometry)
    if polygon_ring is None:
        return {"error": f"Dissemination area '{da_code}' has no usable boundary geometry"}

    geometry_type = geometry.get("type") if geometry else None
    warning = None
    if geometry_type == "MultiPolygon":
        num_parts = len(geometry.get("coordinates") or [])
        excluded = num_parts - 1
        if excluded > 0:
            warning = (
                f"⚠️ Dissemination area {da_code} contains multiple disconnected "
                f"geographic parts (mainland + islets/detached areas). Results "
                f"below show only the main part; {excluded} additional disconnected "
                f"part{'s' if excluded != 1 else ''} {'are' if excluded != 1 else 'is'} "
                f"excluded from this count."
            )

    return {
        "da_code": area.get("Geographie"),
        "geometry_type": geometry_type,
        "warning": warning,
        "polygon_ring": polygon_ring,
        "population": area.get("Population_2021"),
        "num_households": area.get("menages_total"),
        "median_income": area.get("revenu_median"),
        "avg_age": area.get("age_moyen"),
    }


def _as_polygon_geometry(da_geometry):
    """Normalizes either shape a caller might pass into count_businesses_in_da
    into the {"type": "Polygon", "coordinates": [[...]]} shape its $geoWithin
    query needs: a full GeoJSON geometry dict (e.g. find_dissemination_area's
    'geometry' field) is passed through as-is; a flat polygon_ring list of
    [lon, lat] pairs (e.g. get_dissemination_area_by_code's 'polygon_ring',
    now that it no longer returns a raw geometry dict) gets wrapped. Returns
    None for anything else, rather than guessing."""
    if isinstance(da_geometry, dict) and "coordinates" in da_geometry:
        return da_geometry
    if isinstance(da_geometry, list):
        return {"type": "Polygon", "coordinates": [da_geometry]}
    return None


# No allowlist needed: this returns only an integer count (see
# spatial_rag_v1.SpatialHybridRAG.count_businesses_in_da), never the matching
# documents themselves - there's no per-business field to leak because no
# business record ever leaves the database in the first place.
def count_businesses_in_da(
    geo_rag: SpatialHybridRAG,
    da_geometry,
    business_category: Optional[str] = None,
    collection: str = "quebec_businesses",
) -> dict:
    """Wraps count_businesses_in_da. Accepts either a full GeoJSON Polygon
    geometry (from find_dissemination_area's 'geometry' field) or a flat
    polygon_ring list of [lon, lat] pairs (from
    get_dissemination_area_by_code's 'polygon_ring' field) - see
    _as_polygon_geometry - so callers don't have to guess which shape to send.
    collection defaults to "quebec_businesses"; pass "locaux_vacants" to
    count vacant spaces inside the polygon instead."""
    geometry = _as_polygon_geometry(da_geometry)
    if geometry is None:
        return {"error": "da_geometry must be a GeoJSON Polygon geometry object or a flat [[lon, lat], ...] polygon_ring list"}
    return {"count": geo_rag.count_businesses_in_da(geometry, business_category, collection)}


# ─── role_foncier (property tax roll) lookup ───────────────────────────────
# PRIVACY: role_foncier was confirmed via a full-collection field audit
# (212,678 docs, every distinct top-level field name enumerated server-side
# via aggregation, not just a sample) to contain NO owner/proprietor field
# of any kind - standard for Quebec's public rôle d'évaluation foncière
# data, where ownership is excluded for privacy. Confirmed field-free as of
# 2026-07-17.
#
# The ALLOWLIST below (rather than a blocklist) is a defensive measure
# anyway: if this collection is ever joined with another dataset that DOES
# carry owner info, an allowlist means that field simply never reaches the
# output without a deliberate code change to add it - a blocklist would
# silently start leaking it the moment the join happened. _assert_no_owner_fields
# is a second, independent layer on top of that: it inspects the RAW fetched
# document (before allowlist filtering), so a future schema change is caught
# immediately and loudly, instead of the allowlist just quietly continuing
# to filter it out with nobody noticing the underlying data changed.
_PROPERTY_TAX_ALLOWLIST = [
    "mat18", "adresse", "ville", "categorie", "cubf",
    "valeur_terrain", "valeur_totale", "valeur_batiment",
    "superficie_m2", "imposable", "code_mun",
]

# Substring markers checked against fold-accented, lowercased field names -
# "nom_proprietaire" is caught by "proprietaire" alone, no separate entry needed.
_OWNER_FIELD_MARKERS = ("proprietaire", "owner", "titulaire")


def _assert_no_owner_fields(doc: dict, source: str) -> None:
    for key in doc.keys():
        folded = _fold_accents(str(key))
        if any(marker in folded for marker in _OWNER_FIELD_MARKERS):
            raise RuntimeError(
                f"Privacy check failed: field '{key}' found in {source} looks like an "
                f"ownership field. get_property_tax_info's allowlist was written assuming "
                f"role_foncier has none - refusing to proceed until this is reviewed, "
                f"rather than silently deciding whether it's safe to expose."
            )


# ─── unitesEvaluationMatriceGraphique (property boundary + authoritative values) join
# geo_layers also carries a "matrice graphique" cadastral layer - one Polygon
# per unité d'évaluation - for exactly 2 of the 4 villes role_foncier covers
# (Saint-Hyacinthe, Bromont; confirmed via geo_layers.distinct("type") per
# ville - Québec and Les Îles-de-la-Madeleine don't have this layer at all).
# role_foncier itself only stores a Point (see "location" on the raw doc),
# never a boundary shape, so the polygon is always genuinely new information
# to attach here.
#
# The two villes' layer documents come from unrelated source systems with
# completely different schemas - confirmed against real features in both:
#
#  - Saint-Hyacinthe (1 doc, 17150 features): properties are just
#    {ID, ADM01A, SI0317C, SI0318A, SI0528C, DATE_CREATION, DATE_MODIFICATION,
#    USER_MODIFICATION, X_FIC_JOINT, ID_OLD} - no owner or value fields at
#    all, so role_foncier's own valeur_* fields are the only source here (and
#    confirmed internally consistent: valeur_terrain + valeur_batiment ==
#    valeur_totale for all 18495 of its records). Join key: mat18[:10] ==
#    SI0317C and mat18[10:14] == SI0528C - verified against 8 real records by
#    cross-referencing X_FIC_JOINT's embedded plan filename (e.g. ".../Plan
#    localisation_4451-34-6065_..._(6480).pdf") back to role_foncier's own
#    mat18 for the same civic/street. Doesn't resolve for the ~8% of features
#    with a non-"0000" SI0528C (subdivided/condo units) - role_foncier
#    appears to only carry a value at the base-parcel level for those, so
#    this deliberately returns no match rather than guessing which base
#    parcel a subdivided unit rolls up to.
#
#  - Bromont (4 chunked docs, ~6428 features each): a raw municipal GIS
#    export that embeds MAT18 verbatim (exact string match against
#    role_foncier's own mat18 field, confirmed across 500 checked records),
#    AND a real, independently-sourced copy of the assessed values -
#    B61T/B61B/B61I ("bloc 61" of the standard rôle d'évaluation record:
#    Terrain/Bâtiment/Immeuble-i.e.-total; confirmed internally consistent,
#    B61T + B61B == B61I for all 6428 records in every chunk). THIS is the
#    "other" source: confirmed authoritative over role_foncier's own
#    valeur_terrain/valeur_batiment/valeur_totale for Bromont (which,
#    despite disagreeing with these on 460 of 500 checked records, is
#    ALSO internally self-consistent - the two sources are each coherent on
#    their own, just diverged from each other, most likely because
#    role_foncier's copy is a staler extraction). get_property_tax_info
#    below overrides role_foncier's 3 valeur_* fields with these whenever a
#    match is found.
#
#    This layer ALSO embeds the registered owner's name, mailing address,
#    and postal code directly (B75NOM/B75PRENOM/B75N1/B75R/B75M/B75CP/B75C/
#    PROVINCE/PAYS/Sexe/PropPrin/PctPoss) - _UNITE_EVAL_VALUE_FIELDS_BY_VILLE
#    below is an explicit allowlist (not "pass the matched properties dict
#    through") specifically so those never reach the output, the same
#    defensive posture _PROPERTY_TAX_ALLOWLIST already takes for
#    role_foncier itself.
#
# NOTE for whoever picks this up next: Québec's role_foncier data has its
# own, unrelated data-quality problem - valeur_terrain + valeur_batiment !=
# valeur_totale on 88.3% of its 174475 valued records (confirmed), and
# valeur_batiment is null on all but 193 of its 175011 records. Québec has
# no unitesEvaluationMatriceGraphique (or any other value-bearing) geo_layers
# entry to cross-check or override from, so this bug is NOT fixed by
# anything here - it needs its own investigation into Québec's role_foncier
# ETL/source data.
_UNITE_EVAL_LAYER_TYPE = "unitesEvaluationMatriceGraphique"

# Which of the matched feature's OWN properties are safe, confirmed-accurate
# value fields to surface, per ville, mapped to the role_foncier output key
# they override. Saint-Hyacinthe deliberately has no entry - its schema
# carries no value fields at all (see the module comment above).
_UNITE_EVAL_VALUE_FIELDS_BY_VILLE = {
    "Bromont": {"B61T": "valeur_terrain", "B61B": "valeur_batiment", "B61I": "valeur_totale"},
}


def _unite_evaluation_join_key(ville: str, mat18: str) -> Optional[dict]:
    """The {property_key: value} filter that identifies mat18 within ville's
    unitesEvaluationMatriceGraphique schema - see the module comment above
    for why each ville's key is different. None if this ville doesn't carry
    the layer, or mat18 is too short to derive a Saint-Hyacinthe-style key
    from (a defensive check only - role_foncier's mat18 is always 18 chars
    in the live data, confirmed for all 4 villes)."""
    if ville == "Bromont":
        return {"MAT18": mat18} if mat18 else None
    if ville == "Saint-Hyacinthe":
        if not mat18 or len(mat18) < 14:
            return None
        return {"SI0317C": mat18[:10], "SI0528C": mat18[10:14]}
    return None


def _find_unite_evaluation_feature(geo_rag: SpatialHybridRAG, ville: str, mat18: str) -> Optional[dict]:
    """Finds this property's feature in geo_layers' unitesEvaluationMatriceGraphique
    layer and returns {"geometry": ..., "values": {...}} - "values" uses the
    role_foncier-side key names already (via _UNITE_EVAL_VALUE_FIELDS_BY_VILLE),
    empty for villes with no confirmed-safe value fields (Saint-Hyacinthe).
    Never returns anything else from the matched feature's properties - see
    the module comment above for why (Bromont's raw properties carry owner
    PII). Filters server-side via aggregation rather than pulling every
    feature into Python, since a full village layer runs into the tens of
    thousands of features; $ifNull covers both document shapes the live data
    actually uses - a single geojson.features doc (Saint-Hyacinthe) or
    several chunked docs storing features directly (Bromont, split across 4
    documents because of MongoDB's 16MB document limit)."""
    key_filter = _unite_evaluation_join_key(ville, mat18)
    if key_filter is None:
        return None

    value_fields = _UNITE_EVAL_VALUE_FIELDS_BY_VILLE.get(ville, {})
    match_cond = {"$and": [{"$eq": [f"$$f.properties.{k}", v]} for k, v in key_filter.items()]}
    matched_shape = {"geometry": "$$m.geometry"}
    for source_key in value_fields:
        matched_shape[source_key] = f"$$m.properties.{source_key}"

    pipeline = [
        {"$match": {"ville": ville, "type": _UNITE_EVAL_LAYER_TYPE}},
        {"$project": {
            "matched": {
                "$arrayElemAt": [
                    {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": {"$ifNull": ["$features", "$geojson.features"]},
                                    "as": "f",
                                    "cond": match_cond,
                                }
                            },
                            "as": "m",
                            "in": matched_shape,
                        }
                    },
                    0,
                ]
            }
        }},
        {"$match": {"matched": {"$ne": None}}},
        {"$limit": 1},
    ]
    result = list(geo_rag.db["geo_layers"].aggregate(pipeline))
    if not result:
        return None

    matched = result[0]["matched"]
    values = {out_key: matched.get(src_key) for src_key, out_key in value_fields.items()}
    return {"geometry": matched.get("geometry"), "values": values}


def get_property_tax_info(geo_rag: SpatialHybridRAG, identifier: str, identifier_type: str) -> dict:
    """Looks up a single role_foncier (property tax roll) record by exact
    matricule (mat18), address, or coordinates, returning the fields in
    _PROPERTY_TAX_ALLOWLIST above (never the raw document - see the privacy
    note there) plus polygon_ring: the property's boundary shape as a flat
    [[lon, lat], ...] ring, joined in from geo_layers'
    unitesEvaluationMatriceGraphique layer by matricule (see
    _find_unite_evaluation_feature above) - null when this ville has no such
    layer, or no feature matches.

    For villes where that layer carries its own confirmed-accurate value
    fields (currently Bromont only - see _UNITE_EVAL_VALUE_FIELDS_BY_VILLE),
    valeur_terrain/valeur_batiment/valeur_totale are OVERRIDDEN with the
    layer's values instead of role_foncier's own - the module comment above
    _UNITE_EVAL_LAYER_TYPE documents why that source is authoritative there.
    Elsewhere (Saint-Hyacinthe, Québec, Les Îles-de-la-Madeleine) these 3
    fields still come from role_foncier directly, since no alternative
    exists. Sparse fields (valeur_batiment, superficie_m2, polygon_ring)
    come back as null rather than raising when absent - .get() handles that
    uniformly for every allowlisted field, sparse or not.

    identifier_type == "coordinates" expects identifier as "lon,lat" (e.g.
    "-72.691,45.309") since identifier is a single string across all three
    modes, not a separate lat/lon pair."""
    if identifier_type not in ("matricule", "adresse", "coordinates"):
        return {"error": f"Unknown identifier_type '{identifier_type}' - expected 'matricule', 'adresse', or 'coordinates'."}

    parsed_identifier = identifier
    if identifier_type == "coordinates":
        try:
            lon_str, lat_str = identifier.split(",")
            parsed_identifier = (float(lon_str.strip()), float(lat_str.strip()))
        except (ValueError, AttributeError):
            return {"error": f"Could not parse coordinates from '{identifier}' - expected 'lon,lat'."}

    doc = geo_rag.find_property_tax_record(identifier_type, parsed_identifier)
    if doc is None:
        return {"error": "Aucune information trouvée pour cet identifiant."}

    _assert_no_owner_fields(doc, "role_foncier document")

    result = {field: doc.get(field) for field in _PROPERTY_TAX_ALLOWLIST}
    matched = _find_unite_evaluation_feature(geo_rag, doc.get("ville"), doc.get("mat18"))
    if matched:
        result["polygon_ring"] = _flatten_polygon_ring(matched.get("geometry"))
        for key, value in matched.get("values", {}).items():
            if value is not None:
                result[key] = value
    else:
        result["polygon_ring"] = None
    return result


# ─── find_zone_at_location ──────────────────────────────────────────────────

def _point_in_polygon_ring(lon: float, lat: float, ring: list) -> bool:
    """Standard PNPOLY ray-casting test: True if (lon, lat) is inside the
    closed ring. Used instead of shapely - present in this venv but absent
    from uv.lock, i.e. not an actual pinned project dependency - so this
    doesn't silently rely on a package nothing here declares. Only needs the
    outer ring: confirmed every feature across all 3 supported villes'
    zonage layers is a Polygon with a single ring (no holes, no MultiPolygon)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            x_intersect = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _ville_for_point(geo_rag: SpatialHybridRAG, lon: float, lat: float) -> Optional[str]:
    """Resolves a geocoded point to its municipality name by intersecting it
    against the real census dissemination-area polygons in donnees - the
    same $geoIntersects pattern get_smart_demographics already uses for 'the
    exact census block containing X' - then reading that block's sdrCode
    back against the cities collection. Deliberately not using cities'
    own 'bbox' field: that's a coarse rectangle, not the municipality's
    actual boundary, and would misattribute points near a border (verified:
    a naive centroid-of-vertices point inside a real, concave Bromont zone
    polygon resolved to the wrong neighboring municipality via bbox-style
    reasoning, while this donnees-based lookup got it right)."""
    block = geo_rag.db["donnees"].find_one(
        {"geometry": {"$geoIntersects": {"$geometry": {"type": "Point", "coordinates": [lon, lat]}}}},
        {"sdrCode": 1},
    )
    if not block or not block.get("sdrCode"):
        return None
    city_doc = geo_rag.db["cities"].find_one({"sdrCode": block["sdrCode"]}, {"name": 1})
    return city_doc.get("name") if city_doc else None


def _usages_zones_for_bromont_normalized(geo_rag: SpatialHybridRAG, zone_code: str) -> Optional[dict]:
    """Bromont-only fallback for find_zone_at_location: geo_layers' ZONAGE
    and usages_zones' zone_id disagree on hyphen placement for exactly one
    zone (of 199, confirmed against the live data) - "PDA10-08" vs.
    "PDA-10-08". Strips all hyphens from both sides before comparing rather
    than special-casing that one code, in case the same inconsistency exists
    elsewhere and just hasn't been hit yet."""
    normalized_target = zone_code.replace("-", "")
    for candidate in geo_rag.db["usages_zones"].find({"ville": "Bromont"}):
        if candidate.get("zone_id", "").replace("-", "") == normalized_target:
            return candidate
    return None


# ALLOWLIST: zone_code/ville/usages_permis/usages_conditionnels (plus
# status/message on the no-data branch) only - never the matched geo_layers
# feature's raw `properties` dict (which holds whatever per-ville GIS
# authoring fields _ZONAGE_PROPERTY_KEYS_BY_VILLE picks a code out of, not
# meant for an end user) or the zonage polygon's own coordinates (irrelevant
# to a permitted-use question, and unbounded in size for a complex ville-wide
# layer). usages_zones' matched document is also hand-picked rather than
# passed through, for the same reason.
def find_zone_at_location(geo_rag: SpatialHybridRAG, location: str) -> dict:
    """Geocodes location, resolves it to a ville, finds which zonage polygon
    (geo_layers) contains the point, and joins the resulting zone code
    against usages_zones for permitted/conditional uses. See the module
    docstring's find_zone_at_location paragraph for why this implements new
    logic directly here instead of wrapping an existing SpatialHybridRAG
    method, and confirmed-facts about the underlying data."""
    lat, lon = geo_rag.geocode_landmark(location)
    if lat is None or lon is None:
        return {"error": f"Could not geocode '{location}'"}

    ville = _ville_for_point(geo_rag, lon, lat)
    if ville not in _ZONAGE_PROPERTY_KEYS_BY_VILLE:
        return {"error": "Données de zonage non disponibles pour cette ville."}

    layer_doc = geo_rag.db["geo_layers"].find_one({"ville": ville, "type": "zonage"})
    if not layer_doc:
        return {"error": "Données de zonage non disponibles pour cette ville."}

    property_keys = _ZONAGE_PROPERTY_KEYS_BY_VILLE[ville]
    matched_properties = None
    for feature in layer_doc.get("geojson", {}).get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Polygon" or not coordinates:
            continue
        if _point_in_polygon_ring(lon, lat, coordinates[0]):
            matched_properties = feature.get("properties") or {}
            break

    if matched_properties is None:
        return {"error": "Aucune zone trouvée à cette adresse."}

    zone_code = next((matched_properties.get(key) for key in property_keys if matched_properties.get(key)), None)
    if zone_code is None:
        return {"error": "Aucune zone trouvée à cette adresse."}
    zone_code = str(zone_code)

    usage_doc = geo_rag.db["usages_zones"].find_one({"ville": ville, "zone_id": zone_code})
    if usage_doc is None and ville == "Bromont":
        usage_doc = _usages_zones_for_bromont_normalized(geo_rag, zone_code)

    if usage_doc is not None:
        return {
            "zone_code": zone_code,
            "ville": ville,
            "usages_permis": usage_doc.get("usages_permis", []),
            "usages_conditionnels": usage_doc.get("usages_conditionnels", []),
            "status": "ok",
        }

    return {
        "zone_code": zone_code,
        "ville": ville,
        "usages_permis": None,
        "usages_conditionnels": None,
        "status": "usage_data_unavailable",
        "message": (
            f"Le zonage a été identifié (code {zone_code}), mais les usages permis/conditionnels "
            f"ne sont pas encore disponibles pour {ville} dans cette base de données."
        ),
    }


def _find_column(columns, candidates: list) -> Optional[str]:
    """Case/accent-insensitive lookup of the first matching real column name."""
    folded = {_fold_accents(str(c)): c for c in columns}
    for candidate in candidates:
        if candidate in folded:
            return folded[candidate]
    return None


def _row_properties(row) -> dict:
    """A pandas row as a plain JSON-safe dict - NaN -> None, everything else
    kept as-is if already JSON-native, else str()'d (e.g. pandas Timestamps)."""
    out = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _parse_tabular(geo_rag: SpatialHybridRAG, df) -> dict:
    """Shared row-extraction logic for both CSV and Excel, since pandas hands
    back the same DataFrame shape either way."""
    lat_col = _find_column(df.columns, _LAT_COLUMN_CANDIDATES)
    lon_col = _find_column(df.columns, _LON_COLUMN_CANDIDATES)
    name_col = _find_column(df.columns, _NAME_COLUMN_CANDIDATES)
    address_col = _find_column(df.columns, _ADDRESS_COLUMN_CANDIDATES)

    if not (lat_col and lon_col) and not address_col:
        return {
            "error": "No lat/lon columns or an address column found in this file.",
            "points": [], "skipped": 0, "total_rows": len(df),
        }

    points = []
    skipped = 0
    for i, row in df.iterrows():
        lat = lon = None
        if lat_col and lon_col:
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                if math.isnan(lat) or math.isnan(lon):
                    lat = lon = None
            except (TypeError, ValueError):
                lat = lon = None

        if lat is None and address_col and row.get(address_col):
            lat, lon = geo_rag.geocode_landmark(str(row[address_col]))

        if lat is None or lon is None:
            skipped += 1
            continue

        if name_col and row.get(name_col):
            name = str(row[name_col])
        elif address_col and row.get(address_col):
            name = str(row[address_col])
        else:
            name = f"Row {i + 1}"

        points.append({"name": name, "lat": lat, "lon": lon, "properties": _row_properties(row)})

    return {"points": points, "skipped": skipped, "total_rows": len(df)}


def _geojson_centroid(geometry: dict) -> Optional[list]:
    """Approximate [lon, lat] centroid of a Polygon/MultiPolygon - a plain
    average of the outer ring's points, the same approach api_server.py's
    _polygon_centroid uses for chalandise zones. Duplicated here in miniature
    rather than imported, since api_server.py imports this module at load
    time and importing back would be circular."""
    try:
        ring = geometry["coordinates"][0]
        if geometry["type"] == "MultiPolygon":
            ring = ring[0]
        if not ring:
            return None
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        return [sum(lons) / len(lons), sum(lats) / len(lats)]
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return None


def _parse_geojson(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        return {
            "error": f"Unsupported GeoJSON root type '{data.get('type')}' - expected Feature or FeatureCollection.",
            "points": [], "skipped": 0, "total_rows": 0,
        }

    points = []
    skipped = 0
    for i, feature in enumerate(features):
        geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        coords = None

        if geometry.get("type") == "Point":
            coords = geometry.get("coordinates")
        elif geometry.get("type") in ("Polygon", "MultiPolygon"):
            coords = _geojson_centroid(geometry)

        if not coords or len(coords) < 2:
            skipped += 1
            continue

        lon, lat = coords[0], coords[1]
        name = str(props.get("name") or props.get("nom") or f"Feature {i + 1}")
        points.append({"name": name, "lat": lat, "lon": lon, "properties": props})

    return {"points": points, "skipped": skipped, "total_rows": len(features)}


def parse_uploaded_document(geo_rag: SpatialHybridRAG, file_path: str, file_type: str) -> dict:
    """Extracts {name, lat, lon, properties} points from an uploaded CSV,
    GeoJSON, or Excel file. Rows/features with only an address (no lat/lon)
    are geocoded via geo_rag.geocode_landmark; rows that end up with no
    usable coordinates are counted as skipped rather than silently dropped."""
    file_type = (file_type or "").lower()
    try:
        if file_type == "csv":
            import pandas as pd
            return _parse_tabular(geo_rag, pd.read_csv(file_path))
        elif file_type == "excel":
            import pandas as pd
            return _parse_tabular(geo_rag, pd.read_excel(file_path))
        elif file_type == "geojson":
            return _parse_geojson(file_path)
        else:
            return {
                "error": f"Unsupported file_type '{file_type}' - expected 'csv', 'geojson', or 'excel'.",
                "points": [], "skipped": 0, "total_rows": 0,
            }
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}", "points": [], "skipped": 0, "total_rows": 0}
    except Exception as e:
        return {"error": f"Failed to parse {file_type} file: {e}", "points": [], "skipped": 0, "total_rows": 0}


# ── Map/UI tools: no computation, just tell React what to do ───────────────
# Same {"action_type": ..., ...} shape every map_command response in
# api_server.py already uses, so the frontend dispatch code doesn't need a
# second convention.

def toggle_layer(layer_name: str) -> dict:
    return {"action_type": "toggle_layer", "layer_name": layer_name}


def set_filter(sector: Optional[str] = None, naics_code: Optional[str] = None) -> dict:
    return {"action_type": "set_filter", "sector": sector, "naics_code": naics_code}


def set_basemap(basemap_type: str) -> dict:
    return {"action_type": "set_basemap", "basemap_type": basemap_type}


def draw_isochrone(location: str, minutes: int = 10, mode: str = "driving") -> dict:
    return {"action_type": "draw_isochrone", "location": location, "minutes": minutes, "mode": mode}


# Name -> wrapper function, for a future dispatch loop.
TOOL_FUNCTIONS = {
    "geocode": geocode,
    "spatial_search": spatial_search,
    "polygon_filter": polygon_filter,
    "compare_locations": compare_locations,
    "run_analysis": run_analysis,
    "get_demographics": get_demographics,
    "find_dissemination_area": find_dissemination_area,
    "get_dissemination_area_by_code": get_dissemination_area_by_code,
    "count_businesses_in_da": count_businesses_in_da,
    "get_property_tax_info": get_property_tax_info,
    "find_zone_at_location": find_zone_at_location,
    "parse_uploaded_document": parse_uploaded_document,
    "toggle_layer": toggle_layer,
    "set_filter": set_filter,
    "set_basemap": set_basemap,
    "draw_isochrone": draw_isochrone,
}


if __name__ == "__main__":
    print(json.dumps(TOOL_DEFINITIONS, indent=2, ensure_ascii=False))
    print(f"\n{len(TOOL_DEFINITIONS)} tools defined.")
