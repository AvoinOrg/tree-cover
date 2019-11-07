# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
import os

ee.Initialize()


start = "2017-06-01"  # only available since 2018-12...
end = "2019-08-31"
area = "Australia"
collection = "COPERNICUS/S2_SR"

# last index of oz is 15103, so need 15104 rows here.
df = pd.read_csv("data/bastin_db_cleaned.csv", usecols=["longitude", "latitude"], nrows=15104)

lon_counts = dict()
lat_counts = dict()
retrieved = None
t_start = time.time()
saved = []


def maskS2clouds(image):
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 2 ** 10
    cirrusBitMask = 2 ** 11

    # Both flags should be set to zero, indicating clear conditions.
    mask1 = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask2 = qa.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask1).updateMask(mask2).divide(10000)


f_name = f"data/sentinel_{start}-{end}_{int(t_start)}.csv"
err_cnt = 0
# todo: cloud mask? Nah better filter later and resample single entries with nonzero clouds or whole entries
for i in range(13001, 13013):  # df.shape[0]): # df.shape[0]

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
        if retrieved is None:
            retrieved = fetched
        else:
            retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
        # df = pd.DataFrame(e)
        # df.to_csv(f'data/SKYSAT_{start}_{end}_{iis[-1]}.csv')
    except Exception as ex:
        print("Couldnt retrieve", ex)
        time.sleep(60)
        if err_cnt == 0:
            i -= 1
        err_cnt += 1
        if err_cnt >= 5:
            print("Stopping execution, error occured 5 times")
            raise ex

    if i % 10 == 0:
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
        time.sleep(1)

if i % 10 != 0:
    with open(f_name, "a") as file:
        retrieved.to_csv(file, header=False, index=False)
print(f"Fetched data in {(time.time()-t_start)/60} minutes")
print("lon counts:", lon_counts)
print("lat counts:", lat_counts)
print("\a\a\a")  # done now
