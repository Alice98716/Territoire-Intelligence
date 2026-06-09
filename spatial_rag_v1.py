import re
import json
import pymongo
from geopy.geocoders import Nominatim
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

#GLOBAL CONFIGURATION
LLM_MODEL = "llama3.2:3b"
DEFAULT_RADIUS_METERS = 5000
NOMINATIM_USER_AGENT = "quebec_territorial_intelligence_rag"
DB_NAME = "overture_maps"  #target database for user's query 


class QueryExtractor:
    """
    Handles query understanding by combining low-latency Regular Expressions 
    with a local LLM fallback for conversational or complex user queries.
    """
    def __init__(self, model_name: str):
        self.llm = ChatOllama(model=model_name, format="json", temperature=0)
        self.distance_pattern = r'(\d+(?:\.\d+)?)\s*(m|km|mi|meters?|kilometers?|miles?|mètres?|kilomètres?)\b'
        self.location_pattern = r'(?:near|close to|around|près de|à côté de|of|du|de la|des|de)\s+(?:the|le|la|l\')?\s*([^,.\n?]+)'

    def _convert_to_meters(self, value: float, unit: str) -> float:
        """Standardizes extracted distance metrics into meters."""
        unit = unit.lower()
        if unit in ['km', 'kilometer', 'kilometers', 'kilomètre', 'kilomètres']:
            return value * 1000
        elif unit in ['mi', 'mile', 'miles']:
            return value * 1609.34
        return value

    def extract(self, query: str) -> dict:
        """Extracts spatial attributes using Regex or an LLM fallback."""
        query_clean = query.lower()
        dist_match = re.search(self.distance_pattern, query_clean)
        loc_match = re.search(self.location_pattern, query_clean)
        
        if dist_match and loc_match:
            print("[Query Understanding] Success using fast-path Regex.")
            raw_val = float(dist_match.group(1))
            unit_str = dist_match.group(2)
            
            extracted_loc = loc_match.group(1).strip()
            extracted_loc = re.sub(r'\b(?:within|inside|in)\s+.*$', '', extracted_loc, flags=re.IGNORECASE).strip()
            
            return {
                "location": extracted_loc,
                "distance_meters": self._convert_to_meters(raw_val, unit_str)
            }

        print("[Query Understanding] Fuzzy or incomplete parsing. Invoking Ollama fallback...")
        prompt = ChatPromptTemplate.from_template(
            "You are a precise geospatial entity parsing tool.\n"
            "Analyze the following user query and extract the spatial constraints.\n\n"
            "Query: '{query}'\n\n"
            "Return ONLY a valid JSON object with exactly these two keys:\n"
            "- 'location': string containing the landmark/place name (or null if none found)\n"
            "- 'distance_meters': integer representing distance in meters (convert km/miles if necessary, use {default_radius} if not specified)\n\n"
            "JSON Output:"
        )
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"query": query, "default_radius": DEFAULT_RADIUS_METERS})
        
            cleaned_json_string = re.sub(r'```json\s*|```', '', response.content).strip()
            return json.loads(cleaned_json_string)
        except Exception as e:
            print(f"[Query Understanding Error] Failed to parse LLM structured output: {e}")
            return {"location": None, "distance_meters": DEFAULT_RADIUS_METERS}


class SpatialHybridRAG:
    def __init__(self, mongo_uri: str, db_name: str):
        #connection to MongoDB
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        self.geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
        self.extractor = QueryExtractor(LLM_MODEL)
        self.embeddings = OllamaEmbeddings(model=LLM_MODEL)

    def geocode_landmark(self, landmark_name: str) -> tuple:
        """Resolves raw place description text to coordinates via Nominatim."""
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
        """Phase 1: Filter documents from a specific collection using a 2dsphere index."""
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
            text_content = doc.get("description", "")
            # We preserve the collection source name in metadata to distinguish them later
            metadata = {
                "id": str(doc["_id"]),
                "dauid": doc.get("DAUID", "Unknown"),
                "source_collection": collection_name 
            }
            docs.append(Document(page_content=text_content, metadata=metadata))
            
        return docs

    def hybrid_semantic_search(self, query: str, spatial_docs: list) -> list:
        """Phase 2: Perform Dense Semantic Search (Bypassing EnsembleRetriever)."""
        if not spatial_docs:
            return []

        # Create the vector database from MongoDB spatial candidates
        vectorstore = FAISS.from_documents(spatial_docs, self.embeddings)
        
        # Search it using the Ollama embeddings
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        return faiss_retriever.invoke(query)

    def pareto_rank(self, hybrid_results: list, w_spatial=0.4, w_semantic=0.6) -> list:
        """Phase 3: Multi-objective ranking combining spatial exactness and semantic relevance."""
        ranked_results = []
        for doc in hybrid_results:
            final_score = (w_spatial * 1.0) + (w_semantic * 1.0) 
            doc.metadata["pareto_score"] = final_score
            ranked_results.append(doc)
            
        ranked_results.sort(key=lambda x: x.metadata["pareto_score"], reverse=True)
        return ranked_results

    def query_pipeline(self, user_query: str, fallback_lat: float, fallback_lon: float) -> list:
        """Executes the multi-collection merging Spatial-RAG operational pipeline."""
        
        #1. Structural Variable Extraction
        extracted_constraints = self.extractor.extract(user_query)
        landmark = extracted_constraints.get("location")
        raw_radius = extracted_constraints.get("distance_meters", DEFAULT_RADIUS_METERS)
        radius = max(100.0, min(float(raw_radius), 10000.0))
        
        #2. Spatial Anchoring
        lat, lon = fallback_lat, fallback_lon
        if landmark:
            g_lat, g_lon = self.geocode_landmark(landmark)
            if g_lat is not None and g_lon is not None:
                lat, lon = g_lat, g_lon
                
        print(f"[Pipeline Execute] Search parameters settled: Center=({lat}, {lon}) Radius={radius}m")
        
        #3. Step 2 Merge Strategy: Query both collections independently using the 2dsphere index
        vacant_candidates = self.hard_spatial_filter("locaux_vacants", lon, lat, radius)
        business_candidates = self.hard_spatial_filter("quebec_businesses", lon, lat, radius)
        
        #Combine the distinct Document sets together
        all_spatial_candidates = vacant_candidates + business_candidates
        print(f"[Pipeline Pool] Merged pool size: {len(all_spatial_candidates)} total candidates "
              f"({len(vacant_candidates)} vacant, {len(business_candidates)} active businesses).")
        
        #4. Dense/Sparse Embedding Pass over combined documents
        hybrid_matches = self.hybrid_semantic_search(user_query, all_spatial_candidates)
        
        #5. Multi-Objective Scoring Optimization
        final_docs = self.pareto_rank(hybrid_matches)
        return final_docs

import os
from dotenv import load_dotenv

# --- Execution Pipeline ---
if __name__ == "__main__":
    # 1. Load variables from the .env file
    load_dotenv()
    
    # 2. Grab the URI securely from the environment
    cloud_uri = os.getenv("MONGO_URI")
    
    if not cloud_uri:
        print("[ERROR] MONGO_URI not found in .env file!")
        exit(1)
        
    # 3. Initialize the RAG with the secure URI
    geo_rag = SpatialHybridRAG(
        mongo_uri=cloud_uri,
        db_name="overture_maps"
    )

    
    # Define a default center point (e.g., Downtown Saint-Hyacinthe)
    DEFAULT_LAT = 45.6275
    DEFAULT_LON = -72.9286

    print("\n" + "="*50)
    print("TEST 1: The 'Fast Regex' Dual Search")
    print("Testing: Explicit distance (800m) and clear landmark.")
    print("="*50)
    results_1 = geo_rag.query_pipeline(
        user_query="Are there any industrial zones about 15min drive from old port montreal ",
        fallback_lat=DEFAULT_LAT,
        fallback_lon=DEFAULT_LON
    )
    print("\n[Test 1 Results]")
    for i, match in enumerate(results_1[:5]): # Show top 5
        print(f"  {i+1}. [Source: {match.metadata['source_collection']}] ID: {match.metadata['id']}")


    print("\n" + "="*50)
    