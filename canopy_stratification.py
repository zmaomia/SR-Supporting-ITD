# Canopy stratification based on tree height and crown diameter
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


def standardize(values):
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)  #standard deviation
    standardized_values = [(x - mean) / std_dev for x in values]
    return standardized_values, mean, std_dev

# read Shapefile
shapefile_path = './ITD.shp'  # Individual tree detection results
gdf = gpd.read_file(shapefile_path)

gdf['canopy'] = 0


for index, feature in gdf.iterrows():

    geom = feature.geometry 
    # buffer 10 m
    buffer_geom = geom.buffer(5)
    nearby_features = gdf[gdf.intersects(buffer_geom)]
    
    # read CHM and diameter
    chm_values = nearby_features['chm'].tolist()
    diameter_values = nearby_features['Diameter'].tolist()

    standardized_chm, chm_mean, chm_std = standardize(chm_values)
    standardized_diameters, diameters_mean, diameters_std = standardize(diameter_values)

    gdf.at[index, 'chm_mean'] = chm_mean
    gdf.at[index, 'chm_std'] = chm_std
    gdf.at[index, 'dia_mean'] = diameters_mean
    gdf.at[index, 'dia_std'] = diameters_std

    A = 0.5
    B = 0.5
    composite_index_list = [A * h + B * d for h, d in zip(standardized_chm, standardized_diameters)]

    tree_height = feature['chm']
    tree_dia = feature['Diameter']

    composite_index  = A * (tree_height - chm_mean)/chm_std + B * (tree_dia - diameters_mean)/diameters_std 
    gdf.at[index, 'comp_index'] = composite_index

    
    composite_mean = np.mean(composite_index_list)
    composite_std = np.std(composite_index_list, ddof=1)  

    # composite index and thresholds
    #thre_A = composite_mean + composite_std 
    thre_B = composite_mean - composite_std

    # 4 CCG
    #if composite_index > composite_mean + composite_std :
       # gdf.at[index, 'canopy'] = 4
    #elif composite_mean + composite_std  >= composite_index > composite_mean:
    #    gdf.at[index, 'canopy'] = 3
    #elif composite_mean >= composite_index > composite_mean - composite_std:
    #    gdf.at[index, 'canopy'] = 2
    #else:
    #    gdf.at[index, 'canopy'] = 1

    # 3 CCG
    if composite_index >= composite_mean:
        gdf.at[index, 'canopy'] = 3
    elif composite_mean > composite_index >= thre_B:
        gdf.at[index, 'canopy'] = 2
    else:
        gdf.at[index, 'canopy'] = 1

# save Shapefile
output_path = './Canopy_grading.shp'
gdf.to_file(output_path)
