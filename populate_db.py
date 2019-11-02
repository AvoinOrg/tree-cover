""" Transforms 2015_full.csv to proper format and pipes it into db.csv.
    2015_full.csv Format: 
    ['Unnamed: 0', 'B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 
    'B7', 'pixel_qa', 'radsat_qa', 'sr_aerosol', '.geo', 'longitude_x', 
    'latitude_x', 'longitude_y', 'latitude_y', 'dryland_assessment_region', 
    'Aridity_zone', 'land_use_category', 'tree_cover']
    to output format:
    ['longitude', 'latitude', 'dryland_assessment_region', 'Aridity_zone',
    'land_use_category', 'tree_cover', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B10', 'B11', 'pixel_qa', 'radsat_qa', 'sr_aerosol']
"""

import pandas as pd
import os
import itertools

def transform(df):
    """ Sticks to the hardcoded format;
        pretty much a one-off script but m/b requires reuse.
    """
    cols = df.columns.tolist()
    cols_new = cols[16:22] + [cols[1]] + cols[4:10] + cols[2:4] + cols[10:13]
    df = df[cols_new]
    df.rename(columns={'longitude_y': 'longitude', 'latitude_y': 'latitude'}, inplace=True)
    return df

def main():
    os.chdir('/home/dario/_py/tree-cover')
    path = 'data/2015_full.csv'
    df = pd.read_csv(path, sep=',')
    df = transform(df)
    df.to_csv('data/db.csv', index = False)

if __name__ == '__main__':
    main()
