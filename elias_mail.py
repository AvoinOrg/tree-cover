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
        if not chunk: return
        yield chunk

# TODO: break into smaller chunks on failure to retrieve
def fetch_points(df, start='2015-01-01', end='2015-12-31'):
    lon, lat = df.location_x, df.location_y

    for iis in chunk_iter(iter(range(len(df))), 10):
        iis=list(iis)

        print(iis[-1])
        if os.path.exists(f'{iis[-1]}.json'): continue

        GEOM = ee.Geometry.MultiPoint(
            [ [lon[i], lat[i]] for i in iis ]
        ).buffer(60)

        dataset = (
            ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
            .filterDate(start, end)
            .filterBounds(GEOM)
            .getRegion(GEOM, 30)
        )

        print(iis[-1])
        try:
            e = get_stuff(dataset)
            print(iis[-1], 'evaluated')
            with open(f'{iis[-1]}.json', 'wt') as fh:
                json.dump(e, fh)
        except Exception as ex:
            print('Couldnt retrieve', ex)

try:
	os.chdir('/home/dario/_dsp/code')
except:
	pass

ee.Initialize()

df = pd.read_csv("/home/dario/_dsp/data/aam6527_Bastin_Database-S1.csv", sep=";")
df = df.loc[df["dryland_assessment_region"] == 'Australia']
fetch_points(df)


