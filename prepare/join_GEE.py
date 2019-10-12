# -*- coding: utf-8 -*-

"""
Join the data exported by GEE with the data from the table.

@author: fynn
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def join_data():
    gee_data = pd.read_csv('../data/data.csv')
    paper_data = pd.read_csv('../data/bastin_db_cleaned.csv')
    
    # oops - values differ slightly, eg -22.65348 vs -22.653845 
    # and 113.87942859999998 vs 113.87943003678768
    gee_data.drop(labels='system:index', axis=1, inplace=True)
    gee_data['longitude'] = gee_data.apply(lambda row: json.loads(row['.geo'])['coordinates'][0], axis=1)
    gee_data['latitude'] = gee_data.apply(lambda row: json.loads(row['.geo'])['coordinates'][1], axis=1)
    
    gee_indices = []
    paper_indices = []
    for i, lon, lat in gee_data[['longitude', 'latitude']].itertuples():
        lon_close = np.isclose(lon, paper_data['longitude'])
        lat_close = np.isclose(lat, paper_data['latitude'])
        match = lon_close & lat_close
        if match.sum() > 1:
            print (f'{i} - too many: {match.sum()}')
        elif match.sum() == 0:
            print(f'{i} - no match found!')
        else:
            gee_indices.append(i)
            paper_indices.append(paper_data.index[match][0])
    index_df = pd.DataFrame(data={'gee': gee_indices, 'paper': paper_indices})
    gee_close =   pd.merge(index_df, gee_data  , left_on='gee' , right_index=True).reindex(columns=gee_data.columns)
    paper_close = pd.merge(index_df, paper_data, left_on='paper', right_index=True).reindex(columns=paper_data.columns)
    joint_df = pd.merge(gee_close, paper_close, left_index=True, right_index=True)
    return joint_df

def train_dump_forest(joint_df):
    bands = [c for c in joint_df.columns if c.startswith('B')]
    joint_df.dropna(inplace=True)
    X = joint_df[bands].to_numpy()
    y = joint_df['tree_cover']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))

joint_df = join_data()
train_dump_forest(joint_df)

