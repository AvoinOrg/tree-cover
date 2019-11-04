#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:09:26 2019
- clean NAs
- method to add a rectangle with sides 30m into each direction as "geometry" column
- method to generate a subset of the selected data

@author: fynn
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import json

clean_file_location = 'data/bastin_db_cleaned.csv'

def clean_data():
    data = pd.read_csv('../paper_a/Bastin_Database-S1.csv', sep=';')
    data.isnull().sum()
    
    data = data.dropna(subset=['longitude', 'latitude'])
    data.to_csv(clean_file_location)

def add_rectangle_points(data, drop_cols=True):
    """
    adds points in 30m distance of the initial point in order to obtain a 9x9 grid
    """
    r_earth = 6371008.7714150598 # in m
    d_add = (30/r_earth)*(180/np.pi)
    
    data['xMin'] = data['longitude'] - d_add
    data['xMax'] = data['longitude'] + d_add
    data['yMin'] = data['latitude'] - d_add/np.cos(data['longitude']*np.pi/180)
    data['yMax'] = data['latitude'] + d_add/np.cos(data['longitude']*np.pi/180)
    data['geometry'] = data.apply(lambda row: json.dumps({
            'type': 'Polygon', 
            'coordinates': [[[ row['xMin'], row['yMin'] ], [ row['xMax'],row['yMin'] ],
                             [ row['xMax'], row['yMax'] ], [ row['xMin'],row['yMax'] ],
                             [ row['xMin'], row['yMin'] ]
                           ]]}), axis=1)
    if drop_cols:
        data.drop(['xMin', 'xMax', 'yMin', 'yMax'],inplace=True, axis=1)
    
 
def generate_subset(data, index=0, region='Australia', n=1000, fmt='csv'):
    end = data.shape[0] if data.shape[0] < (index+1)*n else (index+1)*n
    sub_df = data[data['dryland_assessment_region'] == region].iloc[index:end,:]
    # copy = sub_df.drop(['longitude', 'latitude'], axis=1)
    if fmt == 'csv':
        sub_df.to_csv(f'{region}_{index*n}_to_{end}.csv', header=True, index=None)
    elif fmt == 'shp':
        sub_df['geometry'] = sub_df.apply(lambda x: Point(float(x.longitude), float(x.latitude)), axis=1)
        sub_df = gpd.GeoDataFrame(sub_df, geometry='geometry', crs="+init=epsg:4326")
        sub_df.to_file(f'{region}_{index*n}_to_{end}.shp', driver='ESRI Shapefile')
    else:
        print(f"Don't know {fmt}.")
        
    
# Convert the long/lat - points into a 70 * 70 area around it for region selection
data = pd.read_csv(clean_file_location)
#add_rectangle_points(data)

generate_subset(data, index=0, n=500, fmt='shp') # change params here


