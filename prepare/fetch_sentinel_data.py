# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
import os

ee.Initialize()

area = "SouthernAfrica"
start = "2017-06-01"  # only available since 2018-12...
end = "2019-08-31"
collection = "COPERNICUS/S2_SR"
print(f'fetching data for {area} from {start} to {end} for {collection}')

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
    "CentralAsia",
    "EastSouthAmerica", # done
    "Europe",
    "HornAfrica",
    "MiddleEast",
    "NorthAmerica",
    "NorthernAfrica",
    "Sahel",
    "SouthernAfrica", # done
    "SouthWestAsia",
    "WestSouthAmerica", # done
]
region_to_batch = {}
df = pd.read_csv("data/bastin_db_cleaned.csv", usecols=["longitude", "latitude", "dryland_assessment_region"])

for region in regions:
    filtered = df[df["dryland_assessment_region"] == region]
    region_to_batch[region] = (filtered.index.min(), filtered.index.max() + 1)

df = df.drop("dryland_assessment_region", axis=1)
lon_counts = dict()
lat_counts = dict()
retrieved = None
t_start = time.time()
saved = []

f_name = f"data/sentinel_{start}-{end}_{area}_from_{region_to_batch[area][0]}.csv"
err_cnt = 0

i = region_to_batch[area][0]
while i < region_to_batch[area][1]:

    # iis=list(range((i-1)*10, i*10))
    # boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
    # GEOM = ee.Geometry.MultiPolygon(boxes)

    GEOM = ee.Geometry.Point([df["longitude"].iloc[i], df["latitude"].iloc[i]]).buffer(35).bounds()

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
        fetched["time"] = pd.to_datetime(fetched["time"], unit="ms")
        lons = pd.unique(fetched["longitude"]).size
        lats = pd.unique(fetched["latitude"]).size
        lon_counts[lons] = lon_counts.get(lons, 0) + 1
        lat_counts[lats] = lat_counts.get(lats, 0) + 1
        counts = fetched.groupby(["latitude", "longitude"]).count().shape[0]
        fetched["id"] = df.index[i]  # index in bastin_cleaned.csv
        i += 1
        if retrieved is None:
            retrieved = fetched
        else:
            retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
    except Exception as ex:
        print(f"i:{i} attempt {err_cnt+1} failed with: ", ex)
        time.sleep(2**(err_cnt + 1))
        err_cnt += 1
        if err_cnt > 9:
            print("Stopping execution, error occured 10 times")
            raise ex

    if i % 10 == 0:
        if i % 1000 == 0:
            f_name = f"data/sentinel_{start}-{end}_{area}_from_{i}.csv"
        if i % 100 == 0:
            print(f"{i}: writing fetched data to file.")
        # print(f"Fetched data in {(time.time()-t_start)/60} minutes")
        # print("lon counts:", lon_counts)
        # print("lat counts:", lat_counts)
        if os.path.isfile(f_name):
            with open(f_name, "a") as file:
                retrieved.to_csv(file, header=False, index=False)
        else:
            retrieved.to_csv(f_name, index=False)
        retrieved = None

if i % 10 != 0:
    with open(f_name, "a") as file:
        retrieved.to_csv(file, header=False, index=False)
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
