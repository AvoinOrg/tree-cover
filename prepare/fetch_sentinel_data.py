# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
import os
import sqlite3 as lite

from utils import timer


bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA10", "QA20", "QA60"]
band_types = ["uint32"]*14 + ["uint8"]*6 + ["uint32"]*3

stmt = """CREATE TABLE sentinel(id INT,longitude FLOAT, latitude FLOAT, time DATETIME, B1 SMALLINT, B2 SMALLINT, 
       B3 SMALLINT, B4 SMALLINT, B5 SMALLINT, B6 SMALLINT, B7 SMALLINT, B8 SMALLINT, B8A SMALLINT, B9 SMALLINT,
       B11 SMALLINT, B12 SMALLINT, AOT SMALLINT, WVP SMALLINT, SCL SMALLINT, TCI_R SMALLINT, TCI_G SMALLINT,
       TCI_B SMALLINT, MSK_CLDPRB SMALLINT, MSK_SNWPRB SMALLINT, HAS_CLOUDFLAG BOOLEAN)"""


def clean_df(df):
    df['HAS_CLOUDFLAG'] = (df['QA60'] == 1024.) | (df['QA60'] == 2048.)
    # https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
    df.longitude = df.longitude.round(7)
    df.latitude = df.latitude.round(7)
    df.time = pd.to_datetime(df.time, unit="ms").apply(lambda t: t.value//10**9)
    df.drop(['QA10', 'QA20', 'QA60'], axis=1, inplace=True)
    return df

def retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end):
    """
    appends to the dataframe of retrieved items (retrieved) and returns the next index and the current error counter
    """
    GEOM = ee.Geometry.Point([df["longitude"].iloc[i], df["latitude"].iloc[i]]).buffer(35).bounds()
    dataset = (
        ee.ImageCollection(collection)
        .filterDate(start, end)
        .filterBounds(GEOM)
        .cast(dict(zip(bands, band_types)), bands)
        .getRegion(GEOM, 10))
    try:
        e = dataset.getInfo()
        err_cnt = 0
        fetched = pd.DataFrame(e[1:], columns=e[0])
        fetched["id"] = df.index[i]  # index in bastin_cleaned.csv
        fetched = clean_df(fetched)
        if retrieved is None:
            retrieved = fetched
        else:
            retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
    except Exception as ex:
        print(f"i:{i} attempt {err_cnt+1} failed with: ", ex)
        if "Too many values:" in str(ex):
            print("Continue with next image due to too much data. Todo: split and try again")
        else:
            time.sleep(2**(err_cnt + 1))
            err_cnt += 1
            if err_cnt > 9:
                print("Stopping execution, error occured 10 times")
                raise ex
    return retrieved, err_cnt


@timer
def fetch_and_write_sqlite(con, df, start, end, collection="COPERNICUS/S2_SR", i=0, i_max=None):
    """ 
    retrieves sentinel data and writes it to the SQLite DB
    con -- the sqlite connection object
    df  -- the dataframe with longitude and latitude column
    start -- start date as string
    end   -- end date as string
    
    Keyword arguments:
    collection -- image collection (default: COPERNICUS/S2_SR)
    i     -- start index (default: 0)
    i_max -- end index (default: length of the dataframe)
    """

    if i_max is None:
        i_max = df.shape[0]
    err_cnt = 0
    retrieved = None
    while i < i_max:
        
        retrieved, i, err_cnt = retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end)
    
        if retrieved is not None and i % 10 == 0:
            if i % 100 == 0:
                print(f"{i}: writing fetched data to file.")

            retrieved.to_sql('sentinel', con, if_exists='append', index=False)
            retrieved = None
    
    if i % 10 != 0:
        retrieved.to_sql('sentinel', con, if_exists='append', index=False)

def run(area='NorternAfrica'):
    
    ee.Initialize()
    start = "2017-06-01"  # only available since 2018-12...
    end = "2019-08-31"

    regions = [
        "Australia",
        "CentralAsia",
        "EastSouthAmerica",
        "Europe",
        "HornAfrica",
        "MiddleEast",
        "NorthAmerica",
        "NorthernAfrica",
        "Sahel",
        "SouthernAfrica",
        "SouthWestAsia",
        "WestSouthAmerica",
    ]

    region_to_batch = {}
    bastin_df = pd.read_csv("data/bastin_db_cleaned.csv", usecols=["longitude", "latitude", "dryland_assessment_region"])
    
    for region in regions:
        filtered = bastin_df[bastin_df["dryland_assessment_region"] == region]
        region_to_batch[region] = (filtered.index.min(), filtered.index.max() + 1)
    
    bastin_df.drop("dryland_assessment_region", axis=1, inplace=True)

    i = region_to_batch[area][0]
    f_name = f"data/sentinel/{area}.db"
    
    db_exists = os.path.isfile(f_name)
    with lite.connect(f_name) as con:
        if db_exists:
            last_id = con.execute('SELECT MAX(id) FROM sentinel').fetchone()[0]
            if last_id is not None:
                i = last_id+1
        else:
            con.execute(stmt)
        print('starting at index ', i)
        fetch_and_write_sqlite(con, bastin_df, start, end, i=i, i_max=region_to_batch[area][1])
            


