import os
import requests
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. API KEYS (set these in your .env file - see .env.example)
# ==========================================
KEYS = {
    "mapbox": os.getenv("MAPBOX_API_KEY", "YOUR_MAPBOX_KEY"),
    "geocodio": os.getenv("GEOCODIO_API_KEY", "YOUR_GEOCODIO_KEY"),
    "geoapify": os.getenv("GEOAPIFY_API_KEY", "YOUR_GEOAPIFY_KEY"),
}

# ==========================================
# 2. EXTENDED TEST ADDRESSES (Quebec Edge Cases)
# ==========================================
TEST_ADDRESSES = [
    # --- 1. LE PIÈGE DES FUSIONS MUNICIPALES (Homonymes exacts) ---
    # Il y a 5 "Rue de l'Église" différentes sur l'île de Montréal !
    "100 Rue de l'Église, Montréal, QC",                # Lequel va-t-il choisir sans code postal ?
    "100 Rue de l'Église, Verdun, QC",                  # Test : Comprend-il l'ancien arrondissement ?
    "100 Rue de l'Église, Saint-Laurent, QC",           # Test : Fait-il bien la distinction avec Verdun ?
    
    # --- 2. L'AMBIGUÏTÉ DIRECTIONNELLE (Est vs Ouest) ---
    "200 Rue Sainte-Catherine, Montréal, QC",           # Test : Sans préciser "Est" ou "Ouest", où place-t-il le point ? Au centre ? Rejette-t-il l'adresse ?
    
    # --- 3. LES ERREURS PHONÉTIQUES ET TYPOGRAPHIQUES COMPLEXES ---
    "Avenue Delorimier, Mtl",                           # Typo fréquente : "Delorimier" collé au lieu de "De Lorimier".
    "Boul Pie 9, Montreal",                             # Phonétique : Comprend-il que "Pie 9" signifie "Pie-IX" (chiffres romains) ?
    "Chemin de la Cote des neiges, QC",                 # Omission totale des accents et des traits d'union.
    
    # --- 4. CONFUSION ENTRE RUE, VILLE ET RÉGION ---
    "Mont-Royal, QC",                                   # Test : Place-t-il le point au centre de la ville de Mont-Royal ?
    "Avenue du Mont-Royal, Montréal, QC",               # Test : Fait-il bien la différence avec la rue dans Le Plateau ?
    "Bellechasse, QC",                                  # La municipalité régionale de comté (MRC).
    "Rue de Bellechasse, Montréal",                     # La rue à Montréal.
    
    # --- 5. SYNTAXE RURALE (Les "Rangs") ---
    "2e Rang, Saint-Hyacinthe",                         # Syntaxe 1 : "2e Rang"
    "Rang 2, Saint-Hyacinthe"                           # Syntaxe 2 : "Rang 2" (Le moteur comprend-il que c'est exactement la même chose ?)
]

# ==========================================
# 3. GEOCODING FUNCTIONS
# ==========================================
def geocode_mapbox(address, key):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={key}&country=ca"
    start = time.time()
    res = requests.get(url).json()
    latency = round((time.time() - start) * 1000)
    
    try:
        feature = res['features'][0]
        lon, lat = feature['center']
        match_type = feature.get('accuracy', feature.get('place_type', [''])[0])
        return {"Lat": lat, "Lon": lon, "Type": match_type, "Latency (ms)": latency}
    except:
        return {"Lat": None, "Lon": None, "Type": "Not Found", "Latency (ms)": latency}

def geocode_geocodio(address, key):
    url = f"https://api.geocod.io/v1.7/geocode?q={address}&api_key={key}"
    start = time.time()
    res = requests.get(url).json()
    latency = round((time.time() - start) * 1000)
    
    try:
        result = res['results'][0]
        lat, lon = result['location']['lat'], result['location']['lng']
        match_type = result.get('accuracy_type', 'Unknown')
        return {"Lat": lat, "Lon": lon, "Type": match_type, "Latency (ms)": latency}
    except:
        return {"Lat": None, "Lon": None, "Type": "Not Found", "Latency (ms)": latency}

def geocode_geoapify(address, key):
    url = f"https://api.geoapify.com/v1/geocode/search?text={address}&apiKey={key}&format=json"
    start = time.time()
    res = requests.get(url).json()
    latency = round((time.time() - start) * 1000)
    
    try:
        result = res['results'][0]
        lat, lon = result['lat'], result['lon']
        match_type = result.get('match_type', 'Unknown')
        return {"Lat": lat, "Lon": lon, "Type": match_type, "Latency (ms)": latency}
    except:
        return {"Lat": None, "Lon": None, "Type": "Not Found", "Latency (ms)": latency}

def geocode_nominatim(address):
    # Nominatim requires a valid User-Agent to prevent blocking
    headers = {"User-Agent": "QuebecGeocodingBenchmark/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    
    start = time.time()
    res = requests.get(url, headers=headers).json()
    latency = round((time.time() - start) * 1000)
    
    # Nominatim Acceptable Use Policy requires 1 second delay between requests
    time.sleep(1) 
    
    try:
        result = res[0]
        lat, lon = float(result['lat']), float(result['lon'])
        match_type = result.get('addresstype', result.get('type', 'Unknown'))
        return {"Lat": lat, "Lon": lon, "Type": match_type, "Latency (ms)": latency}
    except:
        return {"Lat": None, "Lon": None, "Type": "Not Found", "Latency (ms)": latency}

# ==========================================
# 4. RUN BENCHMARK
# ==========================================
def run_benchmark():
    print("Starting Geocoding Benchmark for Quebec...\n")
    results = []

    for address in TEST_ADDRESSES:
        print(f"Testing Address: '{address}'")
        
        # 1. Mapbox
        mb = geocode_mapbox(address, KEYS["mapbox"]) if KEYS["mapbox"] != "YOUR_MAPBOX_KEY" else {"Lat": "Skipped", "Lon": "Skipped", "Type": "No Key", "Latency (ms)": 0}
        
        # 2. Geocodio
        gc = geocode_geocodio(address, KEYS["geocodio"]) if KEYS["geocodio"] != "YOUR_GEOCODIO_KEY" else {"Lat": "Skipped", "Lon": "Skipped", "Type": "No Key", "Latency (ms)": 0}
        
        # 3. Geoapify
        ga = geocode_geoapify(address, KEYS["geoapify"]) if KEYS["geoapify"] != "YOUR_GEOAPIFY_KEY" else {"Lat": "Skipped", "Lon": "Skipped", "Type": "No Key", "Latency (ms)": 0}
        
        # 4. Nominatim
        nom = geocode_nominatim(address)

        # Append to results
        for provider, data in [("Mapbox", mb), ("Geocodio", gc), ("Geoapify", ga), ("Nominatim", nom)]:
            results.append({
                "Address": address,
                "Provider": provider,
                "Latitude": data["Lat"],
                "Longitude": data["Lon"],
                "Match Type": data["Type"],
                "Latency (ms)": data["Latency (ms)"]
            })

    # Display results as a cleanly formatted DataFrame
    df = pd.DataFrame(results)
    df.set_index(["Address", "Provider"], inplace=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n--- BENCHMARK RESULTS ---")
    print(df)

if __name__ == "__main__":
    run_benchmark()
