import geopandas as gpd
import pandas as pd
import os

folder_path1 = r"C:/Users/alice/OneDrive/Bureau/Github Territoire Intelligence/lad_000a21a_f"
folder_path2 = r"C:/Users/alice/OneDrive/Bureau/Github Territoire Intelligence/2021_92-151_X"

#doc 1 
shp_path = os.path.join(folder_path1, "lad_000a21a_f.shp")
boundaries = gpd.read_file(shp_path)

#doc 2
attr_zip_path = os.path.join(folder_path2, "2021_92-151_X.csv")
attributes = pd.read_csv(attr_zip_path, encoding="iso-8859-1")


#columns to check names
#print(boundaries.columns.tolist())
#print(attributes.columns.tolist())


boundaries['ADIDU'] = boundaries['ADIDU'].astype(str)
attributes['DAUID_ADIDU'] = attributes['DAUID_ADIDU'].astype(str)

full_data = boundaries.merge(attributes, left_on='ADIDU', right_on='DAUID_ADIDU')


def get_location_from_id(da_id):
    da_id = str(da_id)
    result = full_data[full_data['ADIDU'] == da_id]
    
    if not result.empty:

        # 'CSDNAME_SDRNOM' is the City/Municipality name
        # 'PRNAME_PRNOM' is the Province name
        city = result.iloc[0]['CSDNAME_SDRNOM']
        province = result.iloc[0]['PRNAME_PRNOM']
        
        return f"City: {city}, Province: {province}"
    else:
        return "ID not found."



def get_ids_from_city(city_name):
    #filter
    result = full_data[full_data['CSDNAME_SDRNOM'].str.contains(city_name, case=False, na=False)]
    
    if not result.empty:
        #liste des aires de diffusions
        da_ids = result['ADIDU'].unique().tolist()
        return da_ids
    else:
        return "City not found."


ids_in_montreal = get_ids_from_city("Montréal")
print(f"Found {len(ids_in_montreal)} Dissemination Areas in Montréal.")
print(ids_in_montreal[:10])  #10 premiers