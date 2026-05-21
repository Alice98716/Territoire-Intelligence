
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def create_geospatial_index():
    raw_uri = os.getenv("MONGO_URI")
    client = MongoClient(raw_uri)
    db = client.overture_maps 
    
    # Create the 2dsphere index for geolocation
    db.locaux_vacants.create_index([("geometry", "2dsphere")])
    print("✅ Index 2dsphere créé avec succès sur locaux_vacants !")

if __name__ == "__main__":
    create_geospatial_index()