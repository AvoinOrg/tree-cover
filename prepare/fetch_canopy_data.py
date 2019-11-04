# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
ee.Initialize()


df = pd.read_csv('data/bastin_db_cleaned.csv')
df['canopy'] = pd.Series()

df.reset_index(inplace=True)
img = ee.Image("NASA/JPL/global_forest_canopy_height_2005")

lon, lat = df.longitude, df.latitude

# todo: proper iter here
start = time.time()
errs = []
for i in range(10000, df.shape[0]):
    
    #iis=list(range((i-1)*10, i*10))
    #boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
    #GEOM = ee.Geometry.MultiPolygon(boxes)
    
    GEOM = ee.Geometry.Point([lon[i], lat[i]])
    
    if i % 5000 == 0 and i != 0:
        print(f'{i}: sleeping a while after fetching 5000.')
        print(f'Fetched data in {time.time()-start} seconds')
        start = time.time()
        df.to_csv(f'data/canopy.csv')
        time.sleep(120)

    # one arc-second is approximately 30 m. This has 30 arc seconds... so very imprecise.
    try:
        data = img.reduceRegion(ee.Reducer.median(), GEOM, 30).getInfo()
        df['canopy'][i] = data['1']

    except Exception as ex:
        errs.append(i)
        print('Couldnt retrieve', ex)

print('all errors: ', errs)
df.to_csv(f'data/canopy.csv')

