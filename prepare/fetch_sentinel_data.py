# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
import os
import sqlite3 as lite

ee.Initialize()

area = "HornAfrica"
start = "2017-06-01"  # only available since 2018-12...
end = "2019-08-31"
collection = "COPERNICUS/S2_SR"
print(f'fetching data for {area} from {start} to {end} for {collection}')
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


# Australia from 0 to 15103
# CentralAsia from 15104 to 35444
# EastSouthAmerica from 35445 to 50463
# Europe from 50464 to 65425
# HornAfrica from 65426 to 85461
# MiddleEast from 85462 to 100517
# NorthAmerica from 100518 to 115525
# NorthernAfrica from 115526 to 130638
# Sahel from 130639 to 156723
# SouthernAfrica from 156724 to 177951
# SouthWestAsia from 177952 to 198234
# WestSouthAmerica from 198235 to 213792
regions = [
    "Australia", # done - except first 3000
    "CentralAsia", # running on my pc now.
    "EastSouthAmerica", # done
    "Europe",
    "HornAfrica", # running on luca
    "MiddleEast",
    "NorthAmerica", # done
    "NorthernAfrica",
    "Sahel",
    "SouthernAfrica", # done
    "SouthWestAsia", # running on instance
    "WestSouthAmerica", # done
]
region_to_batch = {}
bastin_df = pd.read_csv("data/bastin_db_cleaned.csv", usecols=["longitude", "latitude", "dryland_assessment_region"])

for region in regions:
    filtered = bastin_df[bastin_df["dryland_assessment_region"] == region]
    region_to_batch[region] = (filtered.index.min(), filtered.index.max() + 1)

bastin_df.drop("dryland_assessment_region", axis=1, inplace=True)
lon_counts = dict()
lat_counts = dict()
retrieved = None
t_start = time.time()
saved = []
err_cnt = 0
consecutive_incompat_bands = 0
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
    while i < region_to_batch[area][1]:
    
        # iis=list(range((i-1)*10, i*10))
        # boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
        # GEOM = ee.Geometry.MultiPolygon(boxes)
    
        GEOM = ee.Geometry.Point([bastin_df["longitude"].iloc[i], bastin_df["latitude"].iloc[i]]).buffer(35).bounds()
    
        dataset = (
            ee.ImageCollection(collection)
            .filterDate(start, end)
            .filterBounds(GEOM)
            # .map(maskS2clouds)
            .getRegion(GEOM, 10)
            # getRegion: Output an array of values for each [pixel, band, image] tuple in an ImageCollection.
            # The output contains rows of id, lon, lat, time, and all bands for each image that intersects each pixel in the given region.
        )
        try:
            time.sleep(1)
            e = dataset.getInfo()
            err_cnt = 0
            fetched = pd.DataFrame(e[1:], columns=e[0])
            fetched["id"] = bastin_df.index[i]  # index in bastin_cleaned.csv
            fetched = clean_df(fetched)
            lons = pd.unique(fetched["longitude"]).size
            lats = pd.unique(fetched["latitude"]).size
            lon_counts[lons] = lon_counts.get(lons, 0) + 1
            lat_counts[lats] = lat_counts.get(lats, 0) + 1
            counts = fetched.groupby(["latitude", "longitude"]).count().shape[0]
            i += 1
            if retrieved is None:
                retrieved = fetched
            else:
                retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
        except Exception as ex:
            print(f"i:{i} attempt {err_cnt+1} failed with: ", ex)
            if "Expected a homogeneous image collection" in str(ex) and consecutive_incompat_bands < 100:
                print("Continue with next image due to incompatible bands")
                i += 1
                consecutive_incompat_bands += 1
            else:
                time.sleep(2**(err_cnt + 1))
                err_cnt += 1
                if err_cnt > 9:
                    print("Stopping execution, error occured 10 times")
                    raise ex
    
        if retrieved is not None and i % 10 == 0:
            if i % 100 == 0:
                print(f"{i}: writing fetched data to file.")
            # print(f"Fetched data in {(time.time()-t_start)/60} minutes")
            # print("lon counts:", lon_counts)
            # print("lat counts:", lat_counts)
            retrieved.to_sql('sentinel', con, if_exists='append', index=False)
            retrieved = None
    
    if i % 10 != 0:
        retrieved.to_sql('sentinel', con, if_exists='append', index=False)
        
# con closes when exiting with 
print(f"Fetched data in {(time.time()-t_start)/60} minutes")
print("lon counts:", lon_counts)
print("lat counts:", lat_counts)
print("\a\a\a")  # done now


def maskS2clouds(image):
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 2 ** 10
    cirrusBitMask = 2 ** 11

    # Both flags should be set to zero, indicating clear conditions.
    mask1 = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask2 = qa.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask1).updateMask(mask2).divide(10000)
