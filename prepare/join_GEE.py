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

regions = [
    "region_Australia",
    "region_CentralAsia",
    "region_EastSouthAmerica",
    "region_Europe",
    "region_HornAfrica",
    "region_MiddleEast",
    "region_NorthAmerica",
    "region_NorthernAfrica",
    "region_Sahel",
    "region_SouthernAfrica",
    "region_SouthWestAsia",
    "region_WestSouthAmerica",
]

columns = [
    "system:index",
    "B1",
    "B10",
    "B11",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "pixel_qa",
    "radsat_qa",
    "sr_aerosol",
    ".geo",
]


def join_data_by_region_and_row(year):
    """
    expected structure: in data folder, have the splits by region in `by_region` and 
    the year-df exports in `year`.
    """
    df_list = []
    for region in regions:
        reg_df = pd.read_csv(f"data/by_region/{region}.csv")
        sat_df = pd.read_csv(f"data/{year}/export_{region}_{year}.csv").reindex(columns, axis="columns")
        # merged = reg_df.merge(sat_df, how='inner', left_index=True, right_index=True)
        merged = join_data_by_location(sat_df, reg_df)
        df_list.append(merged)
    return pd.concat(df_list, ignore_index=True)


def join_data_by_location(gee_data, paper_data):
    # gee_data = pd.read_csv(gee_file)
    # paper_data = pd.read_csv(paper_file)

    # oops - values differ slightly, eg -22.65348 vs -22.653845
    # and 113.87942859999998 vs 113.87943003678768
    gee_data.drop(labels="system:index", axis=1, inplace=True)
    gee_data["longitude"] = gee_data.apply(lambda row: json.loads(row[".geo"])["coordinates"][0], axis=1)
    gee_data["latitude"] = gee_data.apply(lambda row: json.loads(row[".geo"])["coordinates"][1], axis=1)

    gee_indices = []
    paper_indices = []
    curr_lon = None
    curr_lat = None
    curr_idx = None
    for i, lon, lat in gee_data[["longitude", "latitude"]].itertuples():
        if curr_lon == lon and curr_lat == lat:
            # speedup for sequential data
            gee_indices.append(i)
            paper_indices.append(curr_idx)
            continue
        lon_close = np.isclose(lon, paper_data["longitude"])
        lat_close = np.isclose(lat, paper_data["latitude"])
        match = lon_close & lat_close
        if match.sum() > 1:
            print(f"{i} - too many: {match.sum()}")
        elif match.sum() == 0:
            print(f"{i} - no match found!")
        else:
            gee_indices.append(i)
            curr_idx = paper_data.index[match][0]
            paper_indices.append(curr_idx)
        if i % 10000 == 0:
            print(f"{i} ...")
    index_df = pd.DataFrame(data={"gee": gee_indices, "paper": paper_indices})
    gee_close = pd.merge(index_df, gee_data, left_on="gee", right_index=True).reindex(columns=gee_data.columns)
    paper_close = pd.merge(index_df, paper_data, left_on="paper", right_index=True).reindex(columns=paper_data.columns)
    joint_df = pd.merge(gee_close, paper_close, left_index=True, right_index=True)
    return joint_df


def train_dump_forest(joint_df):
    bands = [c for c in joint_df.columns if c.startswith("B")]
    joint_df.dropna(inplace=True)
    X = joint_df[bands].to_numpy()
    y = joint_df["tree_cover"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))


def run():
    # joint_df = join_data_by_location()
    # train_dump_forest(joint_df)
    join_data_by_region_and_row(2013).to_csv("data/2013_full.csv")
    join_data_by_region_and_row(2014).to_csv("data/2014_full.csv")
    join_data_by_region_and_row(2015).to_csv("data/2015_full.csv")
