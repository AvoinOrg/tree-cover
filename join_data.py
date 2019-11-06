""" Read some csv with latitude and longitude and match the GEE extracted 10'er 
    buckets. In this particular case I add the extraction from the 
    ee.ImageCollection("MODIS/006/MCD64A1") dataset.
"""
import ee
import os
import pandas as pd
import backoff
import itertools
import json

# import pickle
# import time

# df = pd.read_csv("tree-cover/data/aam6527_Bastin_Database-S1.csv", sep=";")
# df = pd.read_csv("data/bastin_db_cleaned.csv", sep=",")
df = pd.read_csv("tree-cover/data/2015_full.csv")
df = df.loc[df["dryland_assessment_region"] == "Australia"]


df = df.assign(no_burn=None)
df = df.copy()

# 9 + 10 * idx%10
# df = df.iloc[0:30]
for i in range(0, len(df)):
    #    print(i)
    if (i - 1) // 10 != i // 10:
        if i > 15099:
            print(f"Loading file no.15103")
            burn = pd.read_csv(f"tree-cover/data/15103_2015-01-01_2015-12-31.csv", sep=",", header=1)
        else:
            print(f"Loading file no.{9+10*(i//10)}")
            burn = pd.read_csv(f"tree-cover/data/{9+10*(i//10)}_2015-01-01_2015-12-31.csv", sep=",", header=1)
    dist = (
        (
            burn.longitude
            - df.longitude_x[i]
            # df.location_x[i]
        )
        ** 2
        + (
            burn.latitude
            - df.latitude_x[i]
            # df.location_y[i]
        )
        ** 2
    )
    idx_low = dist.idxmin()
    idx_up = idx_low + 12
    #    print(f'Indices: {idx_low} to {idx_up-1}.')
    no_burn_count = burn.iloc[idx_low:idx_up].BurnDate.isnull().sum().copy()
    df.no_burn.iloc[i] = no_burn_count

df.to_csv(f"tree-cover/data/df_extended.csv")
