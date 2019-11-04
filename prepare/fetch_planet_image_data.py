# -*- coding: utf-8 -*-
import ee
import pandas as pd
ee.Initialize()


data_full = pd.read_csv('data/bastin_db_cleaned.csv')
df = data_full[data_full['dryland_assessment_region'] == 'Europe'].iloc[:200, :]

df.reset_index(inplace=True)
start='2014-07-03'
end='2016-12-24'
collection="SKYSAT/GEN-A/PUBLIC/ORTHO/MULTISPECTRAL"

lon, lat = df.longitude, df.latitude

# todo: proper iter here
for i in range(1,200):
    
    #iis=list(range((i-1)*10, i*10))
    #boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
    #GEOM = ee.Geometry.MultiPolygon(boxes)
    
    GEOM = ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds()

    dataset = (
         ee.ImageCollection(collection) 
        .filterDate(start, end)
        .filterBounds(GEOM)
        .getRegion(GEOM, 10)
    )

    print('index: ', i)
    # print('current chunk until index:', iis[-1])
    try:
        e = dataset.getInfo()
        # print(iis[-1], 'evaluated')
        print(e)
        #df = pd.DataFrame(e)
        #df.to_csv(f'data/SKYSAT_{start}_{end}_{iis[-1]}.csv')
    except Exception as ex:
        print('Couldnt retrieve', ex)
