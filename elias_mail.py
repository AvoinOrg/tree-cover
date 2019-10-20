import ee
import os
import pandas as pd
import backoff
import itertools
import json
#import pickle
#import time

@backoff.on_exception(backoff.expo,
                      ee.EEException,
                      max_tries=5)
def get_stuff(dataset):
    print('GET')
    return dataset.getInfo()


def chunk_iter(it, size):
    while True:
        chunk = itertools.islice(it, size)
        if not chunk:
            return
        yield chunk


# TODO: break into smaller chunks on failure to retrieve
def fetch_points(df, start='2015-01-01', end='2015-12-31'):
    # lon, lat = df.location_x, df.location_y
    lon, lat = df.longitude, df.latitude

    for iis in chunk_iter(iter(range(len(df))), 10):
        iis=list(iis)

        if len(iis) > 0:
            print(iis[-1])
            if os.path.exists(f'{iis[-1]}.csv'):
                continue
        else:
            continue

        boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
        GEOM = ee.Geometry.MultiPolygon(boxes)

        dataset = (
            ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
            .filterDate(start, end)
            .filterBounds(GEOM)
            .getRegion(GEOM, 30)
        )

        print('current chunk until index:', iis[-1])
        try:
            e = get_stuff(dataset)
            print(iis[-1], 'evaluated')
            df = pd.DataFrame(e)
            df.to_csv(f'{iis[-1]}.csv')
        except Exception as ex:
            print('Couldnt retrieve', ex)


ee.Initialize()

# df = pd.read_csv("/home/dario/_dsp/data/aam6527_Bastin_Database-S1.csv", sep=";")
df = pd.read_csv("data/bastin_db_cleaned.csv", sep=",")
df = df.loc[df["dryland_assessment_region"] == 'Australia']
# fetch_points(df.iloc[:20,:])
fetch_points(df)


