import sys
print("[Import] Loading standard libraries...", flush=True)
import re
import json
import pymongo
import os
import math
import unicodedata
from collections import OrderedDict
from typing import List, Optional

print("[Import] Loading Pydantic...", flush=True)
from pydantic import BaseModel, Field

print("[Import] Loading dotenv...", flush=True)
from dotenv import load_dotenv

print("[Import] Loading Geopy...", flush=True)
from geopy.geocoders import Nominatim

print("[Import] Loading LangChain Ollama...", flush=True)
from langchain_ollama import ChatOllama, OllamaEmbeddings

print("[Import] Loading LangChain BM25...", flush=True)
from langchain_community.retrievers import BM25Retriever

print("[Import] Loading LangChain Prompts...", flush=True)
from langchain_core.prompts import ChatPromptTemplate

print("[Import] Loading FAISS...", flush=True)
from langchain_community.vectorstores import FAISS

print("[Import] Loading LangChain Core...", flush=True)
from langchain_core.documents import Document

print("[Import] Successfully loaded all libraries!", flush=True)

# ... (the rest of your GLOBAL CONFIGURATION stays the same) ...

# GLOBAL CONFIGURATION
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_RADIUS_METERS = 5000
NOMINATIM_USER_AGENT = "quebec_territorial_intelligence_rag"
DB_NAME = "ti_city"
NAICS_COLLECTION_NAME = "naics"


# MODIFICATION 2: Define strict Pydantic schema for structured output
class SpatialQuerySchema(BaseModel):
    location: str = Field(
        description="The landmark/city/place name. Always extract the full anchor name (e.g., 'train station of Saint-Hyacinthe')."
    )
    distance_meters: Optional[int] = Field(
        default=None,
        description="Integer representing distance in meters. Return null if no explicit distance is mentioned."
    )
    collection_intent: str = Field(
        description="Must be exactly one of: 'demographics', 'locaux_vacants', 'quebec_businesses', or 'both'."
    )
    required_fields: List[str] = Field(
        default_factory=list,
        description="An array of strings (max 6) matching keys exactly from the available columns list."
    )
    business_category: Optional[str] = Field(
        default=None,
        description="Type of business, service, or industry requested (e.g., 'restaurants', 'gyms'). Return null if not applicable."
    )


class QueryExtractor:
    """
    Handles query understanding by combining low-latency Regular Expressions
    with a local LLM fallback utilizing structural Pydantic guarantees.
    """
    def __init__(self, model_name: str, db: pymongo.database.Database, embeddings):
        # HARDCODE THE LOCALHOST PORT
        self.llm = ChatOllama(
            model=model_name, 
            temperature=0,
            base_url="http://127.0.0.1:11434"
        )
        self.structured_llm = self.llm.with_structured_output(SpatialQuerySchema)
        self.db = db
        self.embeddings = embeddings

        self.naics_retriever = None #later replace with self._init_naics_index()
        self.distance_pattern = r'(\d+(?:\.\d+)?)\s*(m|km|mi|meters?|kilometers?|miles?|mètres?|kilomètres?)\b'
        self.location_pattern = r'(?:near|close to|around|près de|à côté de|autour de|within\s+(?:\d+(?:\.\d+)?\s*[a-zA-Z]+)?\s+of)\s+(?:the|le|la|l\')?\s*([^,.\n?]+)'
        self.naics_pattern = r'\bnaics\s*[:#-]?\s*(\d{2,6})\b'

    def _init_naics_index(self):
        print("[System] Initializing NAICS semantic lookup index...")
        cursor = self.db[NAICS_COLLECTION_NAME].find({})
        docs = []
        for doc in cursor:
            naics_code = doc.get("naics")
            label = doc.get("label", "")
            subsectors = doc.get("subsectors", [])
            subsector_text = " ".join([str(s) for s in subsectors]) if isinstance(subsectors, list) else str(subsectors)
            search_content = f"{label}. {subsector_text}"

            if naics_code and label:
                docs.append(Document(page_content=search_content, metadata={"naics": naics_code}))

        if docs:
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"[System] NAICS index built successfully with {len(docs)} categories.")
            return vectorstore.as_retriever(search_kwargs={"k": 1})

        print("[System Warning] NAICS index is empty.")
        return None

    def _convert_to_meters(self, value: float, unit: str) -> float:
        unit = unit.lower()
        if unit in ['km', 'kilometer', 'kilometers', 'kilomètre', 'kilomètres']:
            return value * 1000
        elif unit in ['mi', 'mile', 'miles']:
            return value * 1609.34
        return value

    def extract(self, query: str) -> dict:
        query_clean = query.lower()
        dist_match = re.search(self.distance_pattern, query_clean)
        loc_match = re.search(self.location_pattern, query_clean)
        intent = "both"

        demo_keywords = ["demographics", "demographic", "démographie", "population", "habitants"]
        vacant_keywords = ["vacant", "vacants", "empty", "available", "libre", "louer", "location"]
        business_keywords = ["business", "businesses", "cafe", "commerce", "restaurant", "boutique", "office", "bureau"]

        if any(word in query_clean for word in demo_keywords):
            intent = "demographics"
        elif any(word in query_clean for word in vacant_keywords):
            intent = "locaux_vacants"
        elif any(word in query_clean for word in business_keywords):
            intent = "quebec_businesses"

        if dist_match and loc_match and intent != "both":
            raw_val = float(dist_match.group(1))
            unit_str = dist_match.group(2)
            dist_meters = self._convert_to_meters(raw_val, unit_str)

            extracted_loc = loc_match.group(1).strip()
            extracted_loc = re.sub(r'\b(?:within|inside|in)\s+.*$', '', extracted_loc, flags=re.IGNORECASE).strip()

            naics_match = re.search(self.naics_pattern, query_clean)
            extracted_naics = naics_match.group(1) if naics_match else None

            if intent == "quebec_businesses" and not extracted_naics:
                return self._fallback_llm_fields(query, extracted_loc, dist_meters, intent)
            if intent == "demographics":
                return self._fallback_llm_fields(query, extracted_loc, dist_meters, intent)

            return {
                "location": extracted_loc,
                "distance_meters": dist_meters,
                "collection_intent": intent,
                "required_fields": [],
                "naics_code": extracted_naics
            }

        return self._fallback_llm_fields(query, None, None, intent)

    def _fallback_llm_fields(self, query: str, pre_loc: str, pre_dist: float, intent: str) -> dict:
        print("[Query Understanding] Running LLM Schema Agent with Native Pydantic Guardrails...")
        sample_doc = self.db["donnees"].find_one()
        available_columns = list(sample_doc.keys()) if sample_doc else []

        prompt = ChatPromptTemplate.from_template(
            "You are a precise geospatial and database routing agent.\n"
            "Analyze the following user query and extract the spatial constraints.\n"
            "Here is the list of available columns in our demographic database: {columns}\n\n"
            "Query: '{query}'\n"
        )

        # MODIFICATION 2: Invoke the structured output chain directly
        chain = prompt | self.structured_llm
        try:
            structured_response = chain.invoke({
                "query": query,
                "columns": available_columns
            })

            # Convert Pydantic object back into standard dictionary for the execution pipeline
            parsed_json = structured_response.model_dump()

            # Semantic Lookup for NAICS
            extracted_category = parsed_json.get("business_category")
            parsed_json["naics_code"] = None

            if extracted_category and self.naics_retriever:
                naics_docs = self.naics_retriever.invoke(extracted_category)
                if naics_docs:
                    matched_code = naics_docs[0].metadata["naics"]
                    parsed_json["naics_code"] = matched_code
                    print(f"[Query Understanding] Grounded intent '{extracted_category}' to NAICS {matched_code}")

            # Route through REGEX overrides
            if pre_loc: parsed_json["location"] = pre_loc
            if pre_dist: parsed_json["distance_meters"] = pre_dist
            if intent != "both": parsed_json["collection_intent"] = intent

            return parsed_json

        except Exception as e:
            print(f"[Query Understanding Error] Failed validation: {e}")
            return {"location": pre_loc, "distance_meters": pre_dist, "collection_intent": intent, "required_fields": [], "naics_code": None}


# locaux_vacants.prix is free text with inconsistent units across listings
# ("950 $/mois +TPS/TVQ" vs "$0.01/pi²/an") - only extract a figure when it's
# unambiguously per-month. Silently comparing a per-sqft-per-year figure to a
# monthly budget would misrank listings rather than just skip the ones we
# can't confidently parse.
_MONTHLY_PRICE_PATTERN = re.compile(r'([\d\s]+(?:[.,]\d+)?)\s*\$?\s*/\s*mois', re.IGNORECASE)


def _fold_accents(text: str) -> str:
    """Lowercases and strips accents so 'cafe' matches 'Café' - French Quebec
    business/category labels are consistently accented, but users type either
    way, and a literal substring check would otherwise miss real matches."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def _parse_monthly_price(prix_str) -> Optional[float]:
    if not prix_str or not isinstance(prix_str, str):
        return None
    match = _MONTHLY_PRICE_PATTERN.search(prix_str)
    if not match:
        return None
    raw = match.group(1).replace(" ", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


class SpatialHybridRAG:
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.extractor = QueryExtractor(LLM_MODEL, self.db, self.embeddings)
        # doc id -> (page_content, embedding vector), populated lazily by
        # hybrid_semantic_search. Candidate sets overlap heavily across calls
        # (nearby search radii, compare_locations hitting the same city,
        # repeated queries) so this avoids re-embedding the same business via
        # Ollama every time - see hybrid_semantic_search for why that re-embed
        # cost otherwise dominates its latency.
        self._embedding_cache: dict = {}

    # ------------------------------------------------------------------
    # Shared document construction (was duplicated in hard_spatial_filter)
    # ------------------------------------------------------------------
    def _doc_from_mongo(self, doc: dict, collection_name: str, distance_m: Optional[float] = None) -> Document:
        doc_lon, doc_lat = None, None
        if "geometry" in doc and "coordinates" in doc["geometry"]:
            doc_lon, doc_lat = doc["geometry"]["coordinates"]

        if collection_name == "quebec_businesses":
            # Mongo stores several of these sub-objects as explicit null (not just
            # absent keys) on a real fraction of documents - doc.get(key, {}) only
            # substitutes the default when the key is missing, so an explicit null
            # still comes back as None and crashes the next .get() call. `or {}`
            # catches both cases.
            names_obj = doc.get("names") or {}
            business_name = names_obj.get("primary", "Unknown Business")
            cats_obj = doc.get("categories") or {}
            doc_type = cats_obj.get("primary", "Unknown Type")
            sector_obj = cats_obj.get("sector") or {}
            sector = sector_obj.get("label", "Unknown Sector")
            extracted_naics = str(sector_obj.get("naics", ""))
            # addresses is a list (Overture Maps schema), not a flat string like
            # locaux_vacants' "adresse" - freeform is the closest street-address
            # equivalent. "or []" guards the same explicit-null case as above.
            addresses_list = doc.get("addresses") or []
            business_address = (addresses_list[0].get("freeform") if addresses_list else None) or "Adresse inconnue"

            rich_text_content = f"Business Name: {business_name}. Type: {doc_type}. Sector: {sector}. NAICS: {extracted_naics}."
            metadata = {
                "id": str(doc["_id"]),
                "dauid": doc.get("DAUID", "Unknown"),
                "source_collection": collection_name,
                "lon": doc_lon,
                "lat": doc_lat,
                "naics_code": extracted_naics,
                # Same values already baked into rich_text_content above, also
                # exposed as their own metadata keys so callers (e.g.
                # api_server.py's spatial_search results_to_send) can surface
                # them as separate fields instead of parsing the label string.
                "nom": business_name,
                "secteur": sector,
                "adresse": business_address,
            }
        else:
            type_local = doc.get("type_local", "Espace vacant")
            address = doc.get("adresse", "Adresse inconnue")

            rich_text_content = f"Vacant Space Available. Property Type: {type_local}. Address: {address}."
            metadata = {
                "id": str(doc["_id"]),
                "source_collection": collection_name,
                "lon": doc_lon,
                "lat": doc_lat,
                "type_local": type_local,
                "prix": doc.get("prix"),
                # nom/secteur/naics_code don't apply to vacant locals - left
                # absent from metadata here (not set to a placeholder string)
                # so results_to_send's md.get(...) naturally returns None for
                # them, same as businesses never getting a type_local/prix key.
                "adresse": address,
            }

        if distance_m is not None:
            metadata["distance_m"] = distance_m

        return Document(page_content=rich_text_content, metadata=metadata)

    # MODIFICATION 3: Offload spatial filtering and distance calculation to MongoDB Aggregation
    def hard_spatial_filter(self, collection_name: str, lon: float, lat: float, max_distance_meters: float) -> list:
        collection = self.db[collection_name]

        pipeline = [
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": [lon, lat]},
                    "distanceField": "distance_m",
                    "maxDistance": max_distance_meters,
                    "spherical": True
                }
            },
            {"$limit": 150}
        ]

        cursor = collection.aggregate(pipeline)
        return [self._doc_from_mongo(doc, collection_name, distance_m=doc.get("distance_m", 0.0)) for doc in cursor]

    # NEW: query a collection for everything inside a closed polygon ring, instead
    # of a point + radius. This is what lets turn 2 of the agent loop actually use
    # the chalandise-zone polygon React sends back.
    def hard_polygon_filter(self, collection_name: str, polygon_ring: list) -> list:
        collection = self.db[collection_name]
        geo_query = {
            "geometry": {
                "$geoWithin": {
                    "$geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_ring]
                    }
                }
            }
        }
        cursor = collection.find(geo_query).limit(150)
        return [self._doc_from_mongo(doc, collection_name) for doc in cursor]

    def geocode_landmark(self, landmark_name: str) -> tuple:
        try:
            clean_name = landmark_name.lower()
            suffixes_to_remove = ["city center", "city centre", "center", "centre", "downtown", "centre-ville"]
            for suffix in suffixes_to_remove:
                clean_name = re.sub(rf'\b{suffix}\b\s*$', '', clean_name).strip()

            clean_name = clean_name.title() if clean_name else landmark_name
            search_context = f"{clean_name}, Quebec, Canada"

            location = self.geolocator.geocode(search_context, timeout=5)
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            print(f"[Geocoder Error] Nominatim query failed: {e}")
        return None, None

    def hybrid_semantic_search(self, query: str, spatial_docs: list, top_k: int = 15) -> list:
        if not spatial_docs:
            return []

        nomic_query = f"search_query: {query}"

        # FAISS.from_documents() would call embed_documents() on every doc in
        # spatial_docs on every call - the dominant cost of this function
        # (NAICSClassifier's docstring above measured ~15-20s/call for just
        # 318 STATIC docs via Ollama; spatial_docs here run up to ~300 and
        # differ per query, so a per-call rebuild is normally unavoidable).
        # Candidate sets do overlap across calls though (overlapping search
        # radii, compare_locations hitting the same city, repeated queries
        # against the same area), so cache each document's embedding by id
        # and only pay the Ollama round-trip for docs we haven't seen before.
        texts_to_embed, docs_to_embed, cached_pairs = [], [], []
        for doc in spatial_docs:
            doc_id = doc.metadata["id"]
            cached = self._embedding_cache.get(doc_id)
            if cached is not None and cached[0] == doc.page_content:
                cached_pairs.append((doc, cached[1]))
            else:
                docs_to_embed.append(doc)
                texts_to_embed.append(doc.page_content)

        new_embeddings = self.embeddings.embed_documents(texts_to_embed) if texts_to_embed else []
        for doc, embedding in zip(docs_to_embed, new_embeddings):
            self._embedding_cache[doc.metadata["id"]] = (doc.page_content, embedding)

        all_pairs = cached_pairs + list(zip(docs_to_embed, new_embeddings))
        text_embeddings = [(doc.page_content, embedding) for doc, embedding in all_pairs]
        metadatas = [doc.metadata for doc, _ in all_pairs]
        ids = [doc.metadata["id"] for doc, _ in all_pairs]

        vectorstore = FAISS.from_embeddings(text_embeddings, self.embeddings, metadatas=metadatas, ids=ids)
        dense_results = vectorstore.similarity_search(nomic_query, k=len(spatial_docs))

        bm25_retriever = BM25Retriever.from_documents(spatial_docs)
        bm25_retriever.k = len(spatial_docs)
        sparse_results = bm25_retriever.invoke(query)

        rrf_scores = {}
        doc_map = {doc.metadata["id"]: doc for doc in spatial_docs}
        k_rrf = 60

        for rank, doc in enumerate(dense_results):
            doc_id = doc.metadata["id"]
            if doc_id not in rrf_scores: rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + (rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc.metadata["id"]
            if doc_id not in rrf_scores: rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + (rank + 1))

        fused_docs = []
        for doc_id, rrf_score in rrf_scores.items():
            original_doc = doc_map[doc_id]
            normalized_semantic_score = rrf_score * 30.5
            original_doc.metadata["semantic_score"] = min(1.0, max(0.0, normalized_semantic_score))
            fused_docs.append(original_doc)

        fused_docs.sort(key=lambda x: x.metadata["semantic_score"], reverse=True)
        return fused_docs[:top_k]

    # MODIFICATION 3: Removed manual Haversine math loop entirely
    def pareto_rank(self, hybrid_results: list, max_radius: float, target_naics: str = None,
                     target_category: str = None, max_budget: float = None,
                     w_spatial=0.6, w_semantic=0.4) -> list:
        ranked_results = []
        for doc in hybrid_results:
            distance_meters = doc.metadata.get("distance_m", max_radius)

            spatial_score = max(0.0, min(1.0, 1.0 - (distance_meters / max_radius)))
            semantic_score = doc.metadata.get("semantic_score", 0.0)

            final_score = (w_spatial * spatial_score) + (w_semantic * semantic_score)
            collection = doc.metadata.get("source_collection")

            if collection == "quebec_businesses" and target_naics:
                doc_naics = doc.metadata.get("naics_code", "")
                if doc_naics and doc_naics.startswith(target_naics):
                    final_score += 10.0
                    doc.metadata["match_status"] = "Exact NAICS"
                elif doc_naics and target_naics.startswith(doc_naics):
                    final_score += 5.0
                    doc.metadata["match_status"] = "Broad NAICS"
                else:
                    doc.metadata["match_status"] = "No Match"

            elif collection == "quebec_businesses" and target_category:
                # A free-text category name (e.g. "yoga studio") the user typed
                # directly rather than a numeric NAICS code - hard-boost a
                # literal match on top of the semantic ranking already applied,
                # the same way an exact NAICS match does above.
                if _fold_accents(target_category) in _fold_accents(doc.page_content):
                    final_score += 8.0
                    doc.metadata["match_status"] = "Category Match"
                else:
                    doc.metadata["match_status"] = "Semantic Rank Only"

            elif collection == "locaux_vacants" and target_category:
                doc_type = _fold_accents(str(doc.metadata.get("type_local", "")))
                req_type = _fold_accents(target_category)
                service_keywords = ["service", "services", "office", "bureau", "bureaux"]
                retail_keywords = ["retail", "store", "commerce", "boutique", "commercial"]

                if (any(k in req_type for k in service_keywords) and "bureau" in doc_type) or \
                   (any(k in req_type for k in retail_keywords) and "commercial" in doc_type) or \
                   (req_type in doc_type):
                    final_score += 10.0
                    doc.metadata["match_status"] = "Type Local Boosted"
                else:
                    doc.metadata["match_status"] = "Semantic Rank Only"
            else:
                doc.metadata["match_status"] = "Standard Rank"

            # Budget is an independent axis from the category/NAICS matching
            # above - a vacant local can be both the right type AND within
            # budget, so this boost stacks rather than replacing match_status.
            if collection == "locaux_vacants" and max_budget is not None:
                monthly_price = _parse_monthly_price(doc.metadata.get("prix"))
                doc.metadata["monthly_price"] = monthly_price
                if monthly_price is not None:
                    if monthly_price <= max_budget:
                        final_score += 8.0
                        doc.metadata["price_match"] = "within_budget"
                    else:
                        doc.metadata["price_match"] = "over_budget"
                else:
                    doc.metadata["price_match"] = "unknown"

            doc.metadata["pareto_score"] = final_score
            ranked_results.append(doc)

        ranked_results.sort(key=lambda x: x.metadata.get("pareto_score", 0.0), reverse=True)
        return ranked_results

    def get_smart_demographics(self, location_name: str, lat: float, lon: float, radius: float, requested_fields: list) -> list:
        def aggregate_data(cursor, location_desc):
            aggregated_totals = {field.lower(): 0.0 for field in requested_fields}
            display_keys = {field.lower(): field for field in requested_fields}
            valid_doc_count = 0

            for doc in cursor:
                valid_doc_count += 1
                doc_lower = {k.lower(): v for k, v in doc.items()}

                for target_field in requested_fields:
                    tf_lower = target_field.lower()
                    val = doc_lower.get(tf_lower)
                    if val is not None:
                        try:
                            aggregated_totals[tf_lower] += float(val)
                        except (ValueError, TypeError):
                            pass

            if valid_doc_count == 0: return []

            output_metrics = {}
            for tf_lower, total_value in aggregated_totals.items():
                original_key = display_keys[tf_lower]
                if any(kw in tf_lower for kw in ["moyen", "average", "median", "taux", "rate"]):
                    output_metrics[original_key] = round(total_value / valid_doc_count, 2)
                else:
                    output_metrics[original_key] = int(total_value)

            data_str = ", ".join([f"{k}: {v}" for k, v in output_metrics.items()])
            context = f"Aggregated Demographics for {location_desc} (calculated from {valid_doc_count} geographic census blocks): {data_str}"

            # `metrics` exposes the same numbers structured (not just embedded
            # in page_content text) so callers that need real numbers - like
            # the compare verdict - don't have to re-parse a sentence.
            return [Document(page_content=context, metadata={
                "source": "aggregated_donnees", "blocks_merged": valid_doc_count, "metrics": output_metrics,
            })]

        city_doc = self.db["cities"].find_one({"name": {"$regex": f"^{location_name}$", "$options": "i"}})
        if city_doc:
            sdr_code = city_doc.get("sdrCode")
            cursor = self.db["donnees"].find({"sdrCode": sdr_code})
            return aggregate_data(cursor, f"the City of {location_name}")

        if radius:
            spatial_query = {"geometry": {"$near": {"$geometry": {"type": "Point", "coordinates": [lon, lat]}, "$maxDistance": radius}}}
            loc_desc = f"a {radius}m radius around {location_name}"
        else:
            spatial_query = {"geometry": {"$geoIntersects": {"$geometry": {"type": "Point", "coordinates": [lon, lat]}}}}
            loc_desc = f"the exact census block containing {location_name}"

        cursor = self.db["donnees"].find(spatial_query)
        return aggregate_data(cursor, loc_desc)

    # ------------------------------------------------------------------
    # Cross-collection reasoning: rank dissemination areas by a demographic
    # field, then check business density per area - two independent
    # collections (donnees + quebec_businesses) joined by geography rather
    # than a shared key, since neither collection references the other.
    # ------------------------------------------------------------------
    def find_candidate_das(self, lon: float, lat: float, radius: float, limit: int = 60) -> list:
        """Dissemination areas (donnees) near a point, nearest first. Returns
        raw Mongo docs (not Documents) - callers need the full demographic
        field set and the raw polygon, not a page_content summary."""
        cursor = self.db["donnees"].find(
            {"geometry": {"$near": {"$geometry": {"type": "Point", "coordinates": [lon, lat]}, "$maxDistance": radius}}}
        ).limit(limit)
        return list(cursor)

    def get_da_by_code(self, da_code) -> Optional[dict]:
        """Look up a single dissemination area by its EXACT code, rather than
        by proximity to a point - find_candidate_das only supports the
        latter. The donnees collection's code field is "Geographie", stored
        as an int (e.g. 24540318, not "24540318") - confirmed directly
        against the live collection rather than assumed. Returns the same
        raw-doc shape find_candidate_das's list items have (including a
        "geometry" field), or None if the code doesn't match anything."""
        try:
            code = int(da_code)
        except (TypeError, ValueError):
            return None
        return self.db["donnees"].find_one({"Geographie": code})

    def count_businesses_in_da(self, da_geometry: dict, category_phrase: Optional[str] = None,
                                collection: str = "quebec_businesses") -> int:
        """Documents (businesses, or vacant locals when collection=
        "locaux_vacants") whose point falls inside a DA's polygon. With no
        category, a plain indexed count. With one, fetches (capped) and
        applies the same accent-folded substring match pareto_rank uses for
        category boosting, since Mongo can't do that fold server-side. The
        two collections have different schemas (quebec_businesses has
        names/categories.sector; locaux_vacants has a single type_local
        field - see _doc_from_mongo), so the category match looks at
        different fields depending on which collection is queried."""
        geo_filter = {"geometry": {"$geoWithin": {"$geometry": da_geometry}}}
        if not category_phrase:
            return self.db[collection].count_documents(geo_filter)

        folded_target = _fold_accents(category_phrase)

        if collection == "locaux_vacants":
            cursor = self.db[collection].find(geo_filter, {"type_local": 1}).limit(500)
            return sum(
                1 for doc in cursor
                if folded_target in _fold_accents(str(doc.get("type_local", "")))
            )

        cursor = self.db[collection].find(geo_filter, {"names": 1, "categories": 1}).limit(500)
        count = 0
        for doc in cursor:
            names_obj = doc.get("names") or {}
            cats_obj = doc.get("categories") or {}
            sector_obj = cats_obj.get("sector") or {}
            text = f"{names_obj.get('primary','')} {cats_obj.get('primary','')} {sector_obj.get('label','')}"
            if folded_target in _fold_accents(text):
                count += 1
        return count

    # ------------------------------------------------------------------
    # role_foncier (property tax roll) lookups
    # ------------------------------------------------------------------
    def find_property_tax_record(self, identifier_type: str, identifier, radius_meters: float = 30) -> Optional[dict]:
        """Looks up a single role_foncier document by exact matricule
        (mat18), address, or nearest point within a small radius of [lon, lat]
        coordinates. Returns the raw Mongo doc, or None if nothing matches -
        filtering to a safe field allowlist is the caller's job (see
        tools.get_property_tax_info), not this method's."""
        collection = self.db["role_foncier"]

        if identifier_type == "matricule":
            return collection.find_one({"mat18": str(identifier).strip()})

        if identifier_type == "adresse":
            address = str(identifier).strip()
            # Exact match first (case-insensitive - casing is inconsistent
            # across municipalities in this dataset).
            doc = collection.find_one({"adresse": {"$regex": f"^{re.escape(address)}$", "$options": "i"}})
            if doc:
                return doc

            # Fuzzy fallback: split a leading civic number off and match it
            # exactly against no_civique, with the remaining street text
            # matched against nom_rue as a substring (adresse embeds words
            # like "Rue"/"Chemin" that nom_rue alone doesn't carry - e.g.
            # adresse "50 Rue JADE" but nom_rue "JADE"). Deliberately NOT
            # code_postal - confirmed only 13.6% populated, with malformed
            # values observed (e.g. "4"), so it's not a safe required field.
            m = re.match(r'^\s*(\d+)\s*,?\s*(.*)$', address)
            if m:
                civic, street_text = m.group(1), m.group(2).strip()
                if street_text:
                    return collection.find_one({
                        "no_civique": civic,
                        "nom_rue": {"$regex": re.escape(street_text), "$options": "i"},
                    })
            return None

        if identifier_type == "coordinates":
            lon, lat = identifier
            return collection.find_one({
                "location": {
                    "$near": {"$geometry": {"type": "Point", "coordinates": [lon, lat]}, "$maxDistance": radius_meters}
                }
            })

        return None

    def query_pipeline(self, user_query: str, fallback_lat: float, fallback_lon: float) -> list:
        extracted_constraints = self.extractor.extract(user_query)
        landmark = extracted_constraints.get("location")
        raw_radius = extracted_constraints.get("distance_meters")
        target_db = extracted_constraints.get("collection_intent", "both")
        dynamic_fields = extracted_constraints.get("required_fields", [])
        naics_code = extracted_constraints.get("naics_code")
        raw_category = extracted_constraints.get("business_category", "")

        radius = float(raw_radius) if raw_radius is not None else None
        lat, lon = fallback_lat, fallback_lon
        if landmark:
            g_lat, g_lon = self.geocode_landmark(landmark)
            if g_lat is not None and g_lon is not None:
                lat, lon = g_lat, g_lon

        if target_db == "demographics":
            if not landmark or lat is None or lon is None: return []
            return self.get_smart_demographics(landmark, lat, lon, radius, dynamic_fields)

        search_radius = radius if radius is not None else DEFAULT_RADIUS_METERS
        search_radius = max(100.0, min(float(search_radius), 10000.0))
        all_spatial_candidates = []

        if target_db in ["locaux_vacants", "both"]:
            all_spatial_candidates.extend(self.hard_spatial_filter("locaux_vacants", lon, lat, search_radius))

        if target_db in ["quebec_businesses", "both"]:
            all_spatial_candidates.extend(self.hard_spatial_filter("quebec_businesses", lon, lat, search_radius))

        if not all_spatial_candidates: return []

        hybrid_matches = self.hybrid_semantic_search(
            query=user_query,
            spatial_docs=all_spatial_candidates,
            top_k=len(all_spatial_candidates)
        )

        return self.pareto_rank(hybrid_matches, search_radius, target_naics=naics_code, target_category=raw_category)


# ═══════════════════════════════════════════════════════════════════════════
# NAICSClassifier — semantic NAICS classification against db.naics.
#
# Originally built to call SpatialHybridRAG.hybrid_semantic_search()
# (BM25 + FAISS, RRF-fused) directly, matching the pattern used everywhere
# else in this file. Measured against the live data before wiring it into
# anything, and that direct reuse turned out to be wrong on both axes it was
# meant to win on:
#
#  1. LATENCY: hybrid_semantic_search rebuilds a fresh FAISS vectorstore (and
#     re-embeds every document) on every call - correct for its actual
#     callers (spatial_search etc.), whose candidate set is a different,
#     freshly-geo-filtered list each time. db.naics's 318 nodes never change
#     at runtime, so rebuilding+re-embedding them per call is pure waste:
#     measured at ~15-20s/call (10s of which is just embed_documents() on the
#     318 labels via Ollama), vs. ~0.06s to embed a single query. Fixed below
#     by embedding the 318 nodes ONCE in __init__ and reusing that index.
#
#  2. RELEVANCE: BM25Retriever.invoke("gym") and .invoke("cafe") against this
#     corpus returned the IDENTICAL top-5 results for both queries (verified
#     directly) - meaningless noise, not signal. This corpus is short (3-8
#     word) French official-taxonomy labels; English business-idea queries
#     share close to zero literal tokens with them, so BM25's lexical
#     overlap scoring has nothing to lexically match on and effectively
#     returns corpus-frequency noise. RRF-fusing that 50/50 with the (good)
#     dense ranking actively dragged real answers down - fused "gym" put
#     NAICS 513 "Édition" (publishing) first at a fused "confidence" of
#     0.869, while dense-only search alone correctly favored sports/fitness-
#     adjacent categories. Fixed by dropping BM25 for this specific
#     classifier and ranking on cosine similarity only.
#
# Even after both fixes, read classify_business_type's docstring below before
# using this for an auto-accept threshold anywhere - the scores are still not
# reliably separable between "actually right" and "plausible-sounding but
# wrong" at the label granularity db.naics currently has.
# ═══════════════════════════════════════════════════════════════════════════
class NAICSClassifier:
    def __init__(self, geo_rag: "SpatialHybridRAG"):
        self.geo_rag = geo_rag
        self.documents = self._load_naics_documents()
        # Built once and reused for every classify_business_type call - see
        # the LATENCY point above for why rebuilding this per call (the
        # pattern every other hybrid_semantic_search caller uses) doesn't fit
        # a static reference taxonomy like this one.
        self.vectorstore = (
            FAISS.from_documents(self.documents, self.geo_rag.embeddings, normalize_L2=True)
            if self.documents else None
        )

    def _load_naics_documents(self) -> list:
        """One Document per sector/subsector/group node in db.naics. businesses
        arrays (UUID lists, sometimes tens of thousands of entries per group)
        are excluded at the query level ({"businesses": 0}) - they're not
        descriptive text and would otherwise bloat every embedding call for
        no ranking benefit."""
        documents = []
        for sector in self.geo_rag.db[NAICS_COLLECTION_NAME].find({}, {"businesses": 0}):
            documents.append(self._make_doc(sector.get("naics"), sector.get("label")))
            for subsector in sector.get("subsectors", []):
                documents.append(self._make_doc(subsector.get("naics"), subsector.get("label")))
                for group in subsector.get("groups", []):
                    documents.append(self._make_doc(group.get("naics"), group.get("label")))
        return [d for d in documents if d is not None]

    @staticmethod
    def _make_doc(code, label) -> Optional[Document]:
        if not code or not label:
            return None
        return Document(page_content=label, metadata={"naics": code, "label": label})

    def classify_business_type(self, query: str, top_k: int = 5) -> list:
        """Ranks every db.naics node against `query` by cosine similarity
        (normalized FAISS index built once in __init__ - only the query
        itself gets embedded per call) and returns up to top_k
        {"code", "label", "confidence"} dicts, best match first.

        `confidence` is LangChain's relevance_score (roughly cosine
        similarity in [0, 1] for this embedding model) - NOT a calibrated
        "P(this is correct)". Measured against real queries at this label
        granularity: a near-exact vocabulary match ("restaurant" -> NAICS
        7225) scores ~0.6; good-but-inexact matches ("cafe" -> 7225) score
        ~0.44; and outright wrong matches (an English word with no related
        category at all) can still land at ~0.30-0.35 - close enough to
        "gym"'s best real score that no single fixed threshold cleanly
        separates right from wrong yet. Improving that needs richer per-
        category text in db.naics (synonyms/example business names), not
        another code change here - don't wire an auto-accept cutoff off this
        score without validating it against a larger labeled query set first."""
        if not self.vectorstore or not query.strip():
            return []
        results = self.vectorstore.similarity_search_with_relevance_scores(
            f"search_query: {query}", k=top_k
        )
        # FAISS returns score as numpy.float32, not a native Python float -
        # jsonable_encoder (FastAPI) can't serialize that (confirmed: it
        # crashes the whole request with "'numpy.float32' object is not
        # iterable" the moment a caller returns this dict straight from an
        # endpoint), so cast here once rather than relying on every caller
        # to remember to.
        return [
            {"code": doc.metadata["naics"], "label": doc.metadata["label"], "confidence": float(score)}
            for doc, score in results
        ]


if __name__ == "__main__":
    load_dotenv()
    cloud_uri = os.getenv("MONGO_URI")

    if not cloud_uri:
        print("[ERROR] MONGO_URI not found in .env file")
        exit(1)

    geo_rag = SpatialHybridRAG(mongo_uri=cloud_uri, db_name=DB_NAME)

    print("\n" + "="*60)
    print("TEST 1: Direct point+radius pipeline (no zone data needed)")
    print("="*60)

    naics_query = "Find me all vacant spaces that are associated to offices within 300m Saint-Hyacinthe city center."

    naics_results = geo_rag.query_pipeline(
        user_query=naics_query,
        fallback_lat=45.625902,
        fallback_lon=-72.946463
    )

    if not naics_results:
        print("  No results found.")
    else:
        print(f"\n  Top 5 Results for '{naics_query}':\n")
        for i, match in enumerate(naics_results[:5]):
            doc_naics = match.metadata.get('naics_code', 'None')
            status = match.metadata.get('match_status', 'Standard Rank')
            distance = round(match.metadata.get('distance_m', 0))
            score = round(match.metadata.get('pareto_score', 0.0), 3)
            print(f"  {i+1}. [{status}] Score: {score} | Distance: {distance}m | NAICS: {doc_naics}")
            print(f"     Content: {match.page_content[:150]}...\n")