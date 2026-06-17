import re
import json
import pymongo
import os
import math
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# GLOBAL CONFIGURATION
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_RADIUS_METERS = 5000
NOMINATIM_USER_AGENT = "quebec_territorial_intelligence_rag"
DB_NAME = "overture_maps"  
NAICS_COLLECTION_NAME = "naics" # <-- Ensure this matches your DB collection name

class QueryExtractor:
    """
    Handles query understanding by combining low-latency Regular Expressions 
    with a local LLM fallback for conversational or complex user queries.
    Now includes a semantic lookup index for NAICS codes.
    """
    def __init__(self, model_name: str, db: pymongo.database.Database, embeddings):
        self.llm = ChatOllama(model=model_name, format="json", temperature=0)
        self.db = db  
        self.embeddings = embeddings
        
        # Initialize NAICS semantic lookup in memory
        self.naics_retriever = self._init_naics_index()
        
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
        
        print("[System Warning] NAICS index is empty. Check collection name and data.")
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
        
        demo_keywords = [
            "demographics", "demographic", "démographie", "population", "habitants", 
            "navettage", "commute", "commuting", "age", "âge", "stats", "statistiques"
        ]
        
        vacant_keywords = [
            "vacant", "vacants", "empty", "available", "libre", "libres", 
            "vide", "vides", "louer", "location", "rent", "lease", "leasing", 
            "inoccupé", "inoccupés", "disponible", "disponibles", "vendre"
        ]
        
        business_keywords = [
            "business", "businesses", "cafe", "café", "store", "stores", 
            "commerce", "commerces", "restaurant", "restaurants", "entreprise", 
            "entreprises", "magasin", "magasins", "boutique", "boutiques", 
            "retail", "shop", "shops", "office", "offices", "bureau", "bureaux", 
            "competitor", "concurrent", "concurrents", "clinic", "clinique", 
            "supermarket", "supermarché", "grocery", "épicerie", "bar", "bars"
        ]

        if any(word in query_clean for word in demo_keywords):
            intent = "demographics"
        elif any(word in query_clean for word in vacant_keywords):
            intent = "locaux_vacants"
        elif any(word in query_clean for word in business_keywords):
            intent = "quebec_businesses"
            
        # if explicit radius is mentionned short circuit Regex 
        if dist_match and loc_match and intent != "both":
            raw_val = float(dist_match.group(1))
            unit_str = dist_match.group(2)
            dist_meters = self._convert_to_meters(raw_val, unit_str)
            
            extracted_loc = loc_match.group(1).strip()
            extracted_loc = re.sub(r'\b(?:within|inside|in)\s+.*$', '', extracted_loc, flags=re.IGNORECASE).strip()
            
            print(f"[Query Understanding] Fast-path Regex activated. Location: '{extracted_loc}' | Radius: {dist_meters}m")
            
            # extract NAICS when using fast path REGEX
            naics_match = re.search(self.naics_pattern, query_clean)
            extracted_naics = naics_match.group(1) if naics_match else None

            if intent == "quebec_businesses" and not extracted_naics:
                print("[Query Understanding] Missing explicit NAICS number. Routing to LLM for semantic translation...")
                return self._fallback_llm_fields(query, extracted_loc, dist_meters, intent)

            # enforce pre-determined radius 
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
        print("[Query Understanding] Running LLM Schema Agent...")
        sample_doc = self.db["donnees"].find_one()
        available_columns = list(sample_doc.keys()) if sample_doc else []
        
        prompt = ChatPromptTemplate.from_template(
        "You are a precise geospatial and database routing agent.\n"
        "Analyze the following user query and extract the spatial constraints.\n"
        "Here is the list of available columns in our demographic database: {columns}\n\n"
        "Query: '{query}'\n\n"
        "Return ONLY a valid JSON object with exactly these five keys:\n"
        "- 'location': string containing the landmark/city/place name. CRITICAL: Provide the full anchor name (e.g., 'train station of Saint-Hyacinthe').\n"
        "- 'distance_meters': integer representing distance in meters. Return null if no explicit distance is mentioned.\n"
        "- 'collection_intent': string ('demographics', 'locaux_vacants', 'quebec_businesses', or 'both').\n"
        "- 'required_fields': an array of strings (max 6). Match keys exactly from the available columns list.\n"
        "- 'business_category': string. Extract the type of business, service, or industry requested (e.g., 'Soins de santé et assistance sociale', 'restaurants', 'gyms'). Return null if not applicable."
        )
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "query": query, 
                "columns": available_columns
            })
            cleaned_json_string = re.sub(r'```json\s*|```', '', response.content).strip()
            parsed_json = json.loads(cleaned_json_string)
            
            # Semantic Lookup for NAICS
            extracted_category = parsed_json.get("business_category")
            parsed_json["naics_code"] = None 
            
            if extracted_category and self.naics_retriever:
                naics_docs = self.naics_retriever.invoke(extracted_category)
                if naics_docs:
                    matched_code = naics_docs[0].metadata["naics"]
                    matched_label = naics_docs[0].page_content.split('.')[0]
                    parsed_json["naics_code"] = matched_code
                    print(f"[Query Understanding] Grounded intent '{extracted_category}' to NAICS {matched_code} ({matched_label})")

            # Route through REGEX overrides
            if pre_loc:
                parsed_json["location"] = pre_loc
            if pre_dist:
                parsed_json["distance_meters"] = pre_dist
            if intent != "both":
                parsed_json["collection_intent"] = intent
                
            return parsed_json
            
        except Exception as e:
            print(f"[Query Understanding Error] Failed to parse LLM structured output: {e}")
            return {"location": pre_loc, "distance_meters": pre_dist, "collection_intent": intent, "required_fields": [], "naics_code": None}


class SpatialHybridRAG:
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        self.geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # IMPORTANT: Pass embeddings to the extractor so it can build the NAICS index
        self.extractor = QueryExtractor(LLM_MODEL, self.db, self.embeddings) 

    def geocode_landmark(self, landmark_name: str) -> tuple:
        try:
            # 1. Clean conversational suffixes that break Nominatim
            clean_name = landmark_name.lower()
            suffixes_to_remove = [
                "city center", "city centre", "center", "centre", 
                "downtown", "centre-ville", "centre ville"
            ]
            for suffix in suffixes_to_remove:
                # Remove suffix if it appears at the end of the location string
                clean_name = re.sub(rf'\b{suffix}\b\s*$', '', clean_name).strip()
            
            # Capitalize words neatly for better API matching
            clean_name = clean_name.title() if clean_name else landmark_name
            
            # 2. Append territorial context
            search_context = f"{clean_name}, Quebec, Canada"
            print(f"[Geocoder] Attempting to ground sanitized string: '{search_context}'")
            
            location = self.geolocator.geocode(search_context, timeout=5)
            if location:
                print(f"[Geocoder] Grounded '{landmark_name}' -> '{clean_name}' to: {location.latitude}, {location.longitude}")
                return location.latitude, location.longitude
            else:
                print(f"[Geocoder Warning] Nominatim could not resolve '{search_context}'")
        except Exception as e:
            print(f"[Geocoder Error] Nominatim query failed: {e}")
        return None, None

    def hard_spatial_filter(self, collection_name: str, lon: float, lat: float, max_distance_meters: float) -> list:
        collection = self.db[collection_name]
        query = {
            "geometry": {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "$maxDistance": max_distance_meters
                }
            }
        }
        
        cursor = collection.find(query).limit(150)
        docs = []
        for doc in cursor:
            doc_lon, doc_lat = None, None
            if "geometry" in doc and "coordinates" in doc["geometry"]:
                doc_lon, doc_lat = doc["geometry"]["coordinates"]

            # SCHEMA 1: QUEBEC BUSINESSES
            if collection_name == "quebec_businesses":
                names_obj = doc.get("names", {})
                business_name = names_obj.get("primary", "Unknown Business")
                cats_obj = doc.get("categories", {})
                doc_type = cats_obj.get("primary", "Unknown Type")
                sector_obj = cats_obj.get("sector", {})
                sector = sector_obj.get("label", "Unknown Sector")
                extracted_naics = str(sector_obj.get("naics", ""))
                
                rich_text_content = f"Business Name: {business_name}. Type: {doc_type}. Sector: {sector}. NAICS: {extracted_naics}."
                metadata = {
                    "id": str(doc["_id"]),
                    "dauid": doc.get("DAUID", "Unknown"),
                    "source_collection": collection_name,
                    "lon": doc_lon,  
                    "lat": doc_lat,
                    "naics_code": extracted_naics
                }

            # SCHEMA 2: LOCAUX VACANTS (VACANT SPACES)
            else:
                type_local = doc.get("type_local", "Espace vacant")
                address = doc.get("adresse", "Adresse inconnue")
                
                # Build rich context so the semantic search understands the space type
                rich_text_content = f"Vacant Space Available. Property Type: {type_local}. Address: {address}."
                metadata = {
                    "id": str(doc["_id"]),
                    "source_collection": collection_name,
                    "lon": doc_lon,  
                    "lat": doc_lat,
                    "type_local": type_local
                }
            
            docs.append(Document(page_content=rich_text_content, metadata=metadata))
            
        return docs

    def hybrid_semantic_search(self, query: str, spatial_docs: list, top_k: int = 15) -> list:
        if not spatial_docs:
            return []

        nomic_query = f"search_query: {query}"
        vectorstore = FAISS.from_documents(spatial_docs, self.embeddings)
        dense_results = vectorstore.similarity_search(nomic_query, k=len(spatial_docs))

        bm25_retriever = BM25Retriever.from_documents(spatial_docs)
        bm25_retriever.k = len(spatial_docs)
        sparse_results = bm25_retriever.invoke(query)

        rrf_scores = {}
        doc_map = {doc.metadata["id"]: doc for doc in spatial_docs}
        k_rrf = 60

        for rank, doc in enumerate(dense_results):
            doc_id = doc.metadata["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + (rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc.metadata["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + (rank + 1))

        fused_docs = []
        for doc_id, rrf_score in rrf_scores.items():
            original_doc = doc_map[doc_id]
            normalized_semantic_score = rrf_score * 30.5
            original_doc.metadata["semantic_score"] = min(1.0, max(0.0, normalized_semantic_score))
            fused_docs.append(original_doc)

        fused_docs.sort(key=lambda x: x.metadata["semantic_score"], reverse=True)
        return fused_docs[:top_k]
    
    def pareto_rank(self, hybrid_results: list, ref_lat: float, ref_lon: float, max_radius: float, target_naics: str = None, target_category: str = None, w_spatial=0.6, w_semantic=0.4) -> list:
        ranked_results = []
        for doc in hybrid_results:
            doc_lon = doc.metadata.get("lon")
            doc_lat = doc.metadata.get("lat")
            
            # 1. Handle missing coordinates gracefully without crashing
            if doc_lon is None or doc_lat is None:
                doc.metadata["pareto_score"] = 0.0
                doc.metadata["distance_m"] = max_radius
                doc.metadata["match_status"] = "Missing Coordinates"
                ranked_results.append(doc)
                continue
                
            try:
                d_lat, d_lon, r_lat, r_lon = float(doc_lat), float(doc_lon), float(ref_lat), float(ref_lon)
            except (ValueError, TypeError):
                doc.metadata["pareto_score"] = 0.0
                doc.metadata["distance_m"] = max_radius
                doc.metadata["match_status"] = "Coordinate Error"
                ranked_results.append(doc)
                continue
                
            # 2. Calculate Distance
            R = 6371000
            phi1, phi2 = math.radians(r_lat), math.radians(d_lat)
            dphi = math.radians(d_lat - r_lat)
            dlam = math.radians(d_lon - r_lon)
            
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlam/2)**2
            distance_meters = R * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))
            doc.metadata["distance_m"] = distance_meters
            
            # 3. DEFINE THE MISSING SCORES
            spatial_score = max(0.0, min(1.0, 1.0 - (distance_meters / max_radius)))
            semantic_score = doc.metadata.get("semantic_score", 0.0)
            
            # 4. Calculate Final Hybrid Score
            final_score = (w_spatial * spatial_score) + (w_semantic * semantic_score)
            
            collection = doc.metadata.get("source_collection")
            
            # --- BOOST LOGIC FOR BUSINESSES ---
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
            
            # --- BOOST LOGIC FOR VACANT SPACES ---
            elif collection == "locaux_vacants" and target_category:
                doc_type = str(doc.metadata.get("type_local", "")).lower()
                req_type = target_category.lower()
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

            doc.metadata["pareto_score"] = final_score
            ranked_results.append(doc)
            
        ranked_results.sort(key=lambda x: x.metadata.get("pareto_score", 0.0), reverse=True)
        return ranked_results
    
    def get_smart_demographics(self, location_name: str, lat: float, lon: float, radius: float, requested_fields: list) -> list:
        def aggregate_data(cursor, location_desc):
            # lower-case mapping for case-sensitive database 
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
                            numeric_val = float(val)
                            aggregated_totals[tf_lower] += numeric_val
                        except (ValueError, TypeError):
                            pass

            if valid_doc_count == 0:
                return []

            output_metrics = {}
            for tf_lower, total_value in aggregated_totals.items():
                original_key = display_keys[tf_lower]
                if any(kw in tf_lower for kw in ["moyen", "average", "median", "taux", "rate"]):
                    output_metrics[original_key] = round(total_value / valid_doc_count, 2)
                else:
                    output_metrics[original_key] = int(total_value)

            data_str = ", ".join([f"{k}: {v}" for k, v in output_metrics.items()])
            context = f"Aggregated Demographics for {location_desc} (calculated from {valid_doc_count} geographic census blocks): {data_str}"
            
            return [Document(
                page_content=context, 
                metadata={"source": "aggregated_donnees", "blocks_merged": valid_doc_count}
            )]

        # 1. MACRO CHECK: city scale 
        city_doc = self.db["cities"].find_one({
            "name": {"$regex": f"^{location_name}$", "$options": "i"}
        })
        
        if city_doc:
            print(f"[GIS Router] '{location_name}' identified as a full CITY. Using sdrCode aggregation.")
            sdr_code = city_doc.get("sdrCode")
            cursor = self.db["donnees"].find({"sdrCode": sdr_code})
            return aggregate_data(cursor, f"the City of {location_name}")

        # 2. MICRO CHECK: neighborhood 
        if radius:
            print(f"[GIS Router] '{location_name}' is a LOCAL area. Searching polygons within {radius} meters.")
            spatial_query = {
                "geometry": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "$maxDistance": radius
                    }
                }
            }
            loc_desc = f"a {radius}m radius around {location_name}"
        else:
            print(f"[GIS Router] No radius specified for '{location_name}'. Finding EXACT intersecting polygon (Point-in-Polygon).")
            spatial_query = {
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        }
                    }
                }
            }
            loc_desc = f"the exact census block containing {location_name}"

        cursor = self.db["donnees"].find(spatial_query)
        return aggregate_data(cursor, loc_desc)

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
                
        print(f"[Pipeline Execute] Center=({lat}, {lon}) Radius={radius if radius else 'Exact Point'} | Target: {target_db.upper()}")
        if dynamic_fields:
            print(f"[Pipeline Execute] LLM Requested Data Fields: {dynamic_fields}")
        
        # Demographic routers 
        if target_db == "demographics":
            print(f"[Pipeline Execute] Rerouting to Spatial GIS lookup for Demographics: {landmark}")
            if not landmark or lat is None or lon is None:
                print("[Error] Missing spatial data for demographics.")
                return []
                
            return self.get_smart_demographics(landmark, lat, lon, radius, dynamic_fields)
            
        # Business/Vacant Properties
        search_radius = radius if radius is not None else DEFAULT_RADIUS_METERS
        search_radius = max(100.0, min(float(search_radius), 10000.0))
        all_spatial_candidates = []
        
        if target_db in ["locaux_vacants", "both"]:
            vacant_candidates = self.hard_spatial_filter("locaux_vacants", lon, lat, search_radius)
            all_spatial_candidates.extend(vacant_candidates)
            print(f"  -> Added {len(vacant_candidates)} vacant candidates.")
            
        if target_db in ["quebec_businesses", "both"]:
            business_candidates = self.hard_spatial_filter("quebec_businesses", lon, lat, search_radius)
            all_spatial_candidates.extend(business_candidates)
            print(f"  -> Added {len(business_candidates)} business candidates.")
        
        if not all_spatial_candidates:
            print("[Pipeline Pool] No candidates found in specified area.")
            return []
        
        hybrid_matches = self.hybrid_semantic_search(
            query=user_query, 
            spatial_docs=all_spatial_candidates, 
            top_k=len(all_spatial_candidates) # keep all of them before pareto ranking
        )
        final_docs = self.pareto_rank(hybrid_matches, lat, lon, search_radius, target_naics=naics_code, target_category=raw_category)
        return final_docs


# EXECUTION PIPELINE 
if __name__ == "__main__":
    load_dotenv()
    cloud_uri = os.getenv("MONGO_URI")
    
    if not cloud_uri:
        print("[ERROR] MONGO_URI not found in .env file")
        exit(1)

    geo_rag = SpatialHybridRAG(
        mongo_uri=cloud_uri,
        db_name=DB_NAME
    )
    
   # TEST 4: BUSINESS ROUTING WITH NAICS CODE (Soft Filter Boost)
    print("\n" + "="*60)
    print("TEST 4: BUSINESS ROUTING WITH NAICS CODE BOOST")
    print("="*60)
    
    # 62 represents the NAICS prefix for "Soins de santé et assistance sociale" 
    # (Healthcare and Social Assistance)
    naics_query = "Find me all vacant spaces that are associated to offices within 300m Saint-Hyacinthe city center."

    
    naics_results = geo_rag.query_pipeline(
        user_query=naics_query, 
        fallback_lat=45.625902, 
        fallback_lon=-72.946463
    )
    
    if not naics_results:
        print("  No results found. (Check if your test database has Montreal data)")
    else:
        print(f"\n  Top 5 Results for '{naics_query}':\n")
        for i, match in enumerate(naics_results[:5]):
            doc_id = match.metadata.get('id', 'Unknown')
            doc_naics = match.metadata.get('naics_code', 'None')
            is_match = match.metadata.get('naics_match', False)
            distance = round(match.metadata.get('distance_m', 0))
            score = round(match.metadata.get('pareto_score', 0.0), 3)
            
            match_status = "[★ NAICS BOOSTED]" if is_match and is_match != "No Match" and is_match != "N/A" else "[Standard Rank]"
            
            print(f"  {i+1}. {match_status} Score: {score} | Distance: {distance}m | NAICS: {doc_naics}")
            print(f"     Content: {match.page_content[:150]}...\n")