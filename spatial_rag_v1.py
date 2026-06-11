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

class QueryExtractor:
    """
    Handles query understanding by combining low-latency Regular Expressions 
    with a local LLM fallback for conversational or complex user queries.
    """
    def __init__(self, model_name: str, db: pymongo.database.Database):
        self.llm = ChatOllama(model=model_name, format="json", temperature=0)
        self.db = db  
        self.distance_pattern = r'(\d+(?:\.\d+)?)\s*(m|km|mi|meters?|kilometers?|miles?|mètres?|kilomètres?)\b'
        self.location_pattern = r'(?:near|close to|around|près de|à côté de|of|du|de la|des|de)\s+(?:the|le|la|l\')?\s*([^,.\n?]+)'

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
            
        #if explicit radius is mentionned short circuit Regex 
        if dist_match and loc_match and intent != "both":
            raw_val = float(dist_match.group(1))
            unit_str = dist_match.group(2)
            dist_meters = self._convert_to_meters(raw_val, unit_str)
            
            extracted_loc = loc_match.group(1).strip()
            extracted_loc = re.sub(r'\b(?:within|inside|in)\s+.*$', '', extracted_loc, flags=re.IGNORECASE).strip()
            
            print(f"[Query Understanding] Fast-path Regex activated. Location: '{extracted_loc}' | Radius: {dist_meters}m")
            
            #enforce pre-determined radius 
            if intent == "demographics":
                return self._fallback_llm_fields(query, extracted_loc, dist_meters, intent)
                
            return {
                "location": extracted_loc,
                "distance_meters": dist_meters,
                "collection_intent": intent,
                "required_fields": []
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
            "Return ONLY a valid JSON object with exactly these four keys:\n"
            "- 'location': string containing the landmark/city/place name. CRITICAL: If the query contains a specific localized building or anchor (e.g., 'train station of Saint-Hyacinthe'), your location value MUST be the full 'train station of Saint-Hyacinthe'. Do NOT truncate it to just the city name 'Saint-Hyacinthe'!\n"
            "- 'distance_meters': integer representing distance in meters. CRITICAL: Return null if no explicit distance, radius, or 'within X km' is mentioned in the prompt.\n"
            "- 'collection_intent': string ('demographics', 'locaux_vacants', 'quebec_businesses', or 'both').\n"
            "- 'required_fields': an array of strings (max 6). CRITICAL RULES: Match keys exactly from the available columns list. If the user asks for general 'demographics', always pick relevant base fields (e.g., 'Population_2021', 'age_moyen', 'revenu_moyen').\n"
        )
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "query": query, 
                "columns": available_columns
            })
            cleaned_json_string = re.sub(r'```json\s*|```', '', response.content).strip()
            parsed_json = json.loads(cleaned_json_string)
            
            #go through REGEX
            if pre_loc:
                parsed_json["location"] = pre_loc
            if pre_dist:
                parsed_json["distance_meters"] = pre_dist
            if intent != "both":
                parsed_json["collection_intent"] = intent
                
            return parsed_json
        except Exception as e:
            print(f"[Query Understanding Error] Failed to parse LLM structured output: {e}")
            return {"location": pre_loc, "distance_meters": pre_dist, "collection_intent": intent, "required_fields": []}


class SpatialHybridRAG:
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        self.geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
        self.extractor = QueryExtractor(LLM_MODEL, self.db) 
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    def geocode_landmark(self, landmark_name: str) -> tuple:
        try:
            search_context = f"{landmark_name}, Quebec, Canada"
            location = self.geolocator.geocode(search_context, timeout=5)
            if location:
                print(f"[Geocoder] Grounded '{landmark_name}' to coordinates: {location.latitude}, {location.longitude}")
                return location.latitude, location.longitude
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
            desc = doc.get("description", "")
            doc_type = doc.get("type_local", doc.get("categorie", ""))
            sector = doc.get("secteur", "")
            rich_text_content = f"search_document: Type: {doc_type}. Sector: {sector}. Details: {desc}"
        
            doc_lon = None
            doc_lat = None
            if "geometry" in doc and "coordinates" in doc["geometry"]:
                doc_lon, doc_lat = doc["geometry"]["coordinates"]
            
            metadata = {
                "id": str(doc["_id"]),
                "dauid": doc.get("DAUID", "Unknown"),
                "source_collection": collection_name,
                "lon": doc_lon,  
                "lat": doc_lat   
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
    
    def pareto_rank(self, hybrid_results: list, ref_lat: float, ref_lon: float, max_radius: float, w_spatial=0.6, w_semantic=0.4) -> list:
        ranked_results = []
        for doc in hybrid_results:
            doc_lon = doc.metadata.get("lon")
            doc_lat = doc.metadata.get("lat")
            
            if doc_lon is None or doc_lat is None:
                doc.metadata["pareto_score"] = 0.0
                doc.metadata["distance_m"] = max_radius
                ranked_results.append(doc)
                continue
                
            try:
                d_lat = float(doc_lat)
                d_lon = float(doc_lon)
                r_lat = float(ref_lat)
                r_lon = float(ref_lon)
            except (ValueError, TypeError):
                doc.metadata["pareto_score"] = 0.0
                doc.metadata["distance_m"] = max_radius
                ranked_results.append(doc)
                continue
                
            R = 6371000
            phi1, phi2 = math.radians(r_lat), math.radians(d_lat)
            dphi = math.radians(d_lat - r_lat)
            dlam = math.radians(d_lon - r_lon)
            
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlam/2)**2
            a = min(1.0, max(0.0, a))
            distance_meters = R * 2 * math.asin(math.sqrt(a))
            
            doc.metadata["distance_m"] = distance_meters
            spatial_score = 1.0 - (distance_meters / max_radius)
            spatial_score = max(0.0, min(1.0, spatial_score)) 
            
            semantic_score = doc.metadata.get("semantic_score", 0.0)
            final_score = (w_spatial * spatial_score) + (w_semantic * semantic_score)
            
            doc.metadata["pareto_score"] = final_score
            ranked_results.append(doc)
            
        ranked_results.sort(key=lambda x: x.metadata.get("pareto_score", 0.0), reverse=True)
        return ranked_results

    def get_smart_demographics(self, location_name: str, lat: float, lon: float, radius: float, requested_fields: list) -> list:
        def aggregate_data(cursor, location_desc):
            #lower-case mapping for case-sensitive database 
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

        #1. MACRO CHECK: city scale 
        city_doc = self.db["cities"].find_one({
            "name": {"$regex": f"^{location_name}$", "$options": "i"}
        })
        
        if city_doc:
            print(f"[GIS Router] '{location_name}' identified as a full CITY. Using sdrCode aggregation.")
            sdr_code = city_doc.get("sdrCode")
            cursor = self.db["donnees"].find({"sdrCode": sdr_code})
            return aggregate_data(cursor, f"the City of {location_name}")

        #2. MICRO CHECK: neighborhood 
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
        
        radius = float(raw_radius) if raw_radius is not None else None
        
        lat, lon = fallback_lat, fallback_lon
        if landmark:
            g_lat, g_lon = self.geocode_landmark(landmark)
            if g_lat is not None and g_lon is not None:
                lat, lon = g_lat, g_lon
                
        print(f"[Pipeline Execute] Center=({lat}, {lon}) Radius={radius if radius else 'Exact Point'} | Target: {target_db.upper()}")
        if dynamic_fields:
            print(f"[Pipeline Execute] LLM Requested Data Fields: {dynamic_fields}")
        
        #Demographic routers 
        if target_db == "demographics":
            print(f"[Pipeline Execute] Rerouting to Spatial GIS lookup for Demographics: {landmark}")
            if not landmark or lat is None or lon is None:
                print("[Error] Missing spatial data for demographics.")
                return []
                
            return self.get_smart_demographics(landmark, lat, lon, radius, dynamic_fields)
            
        #Business/Vacant Properties
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
        
        hybrid_matches = self.hybrid_semantic_search(user_query, all_spatial_candidates)
        final_docs = self.pareto_rank(hybrid_matches, lat, lon, search_radius)
        return final_docs


#EXECUTION PIPELINE 
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
    
    print("\n" + "="*60)
    print("INITIATING DEMOGRAPHICS ROUTING TEST")
    print("="*60)

    #TEST 1: MACRO (CITY LEVEL)
    print("\n TEST 1: MACRO CITY QUERY")
    city_query = "What are the demographics of Les Îles-de-la-Madeleine?"
    macro_results = geo_rag.query_pipeline(user_query=city_query, fallback_lat=47.3822, fallback_lon=-61.8596)
    for match in macro_results[:1]: 
        print(f"  Content: {match.page_content}")

    #TEST 2: MICRO (NEIGHBORHOOD LEVEL WITH RADIUS)
    print("\n TEST 2: MICRO NEIGHBORHOOD (1KM RADIUS)")
    micro_query_radius = "What is the total commuter population within 1km of the train station of Saint-Hyacinthe?"
    micro_radius_results = geo_rag.query_pipeline(user_query=micro_query_radius, fallback_lat=45.6275, fallback_lon=-72.9286)
    for match in micro_radius_results[:1]:
        print(f"  Content: {match.page_content}")

    #TEST 3: MICRO (EXACT SINGLE CENSUS BLOCK)
    print("\nTEST 3: MICRO NEIGHBORHOOD (EXACT POINT-IN-POLYGON)")
    micro_query_exact = "What are the exact commuter demographics at the train station of Saint-Hyacinthe?"
    micro_exact_results = geo_rag.query_pipeline(user_query=micro_query_exact, fallback_lat=45.6275, fallback_lon=-72.9286)
    for match in micro_exact_results[:1]:
        print(f"  Content: {match.page_content}")
        
    print("\n" + "="*60 + "\n")