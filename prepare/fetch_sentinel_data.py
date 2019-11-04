# -*- coding: utf-8 -*-
import ee
import pandas as pd
import time
ee.Initialize()


data_full = pd.read_csv('data/bastin_db_cleaned.csv')

start='2018-12-01'
end='2019-02-28'
area = 'Australia'
collection="COPERNICUS/S2_SR"
df = data_full[data_full['dryland_assessment_region'] == area]

lon_counts = dict()
lat_counts = dict()
retrieved = None
t_start = time.time()

def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 2**10
    cirrusBitMask = 2**11

    # Both flags should be set to zero, indicating clear conditions.
    mask1 = qa.bitwiseAnd(cloudBitMask).eq(0) 
    mask2 = qa.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask1).updateMask(mask2).divide(10000)


# todo: cloud mask? Nah better filter later.
for i in range(300,df.shape[0]):
    
    #iis=list(range((i-1)*10, i*10))
    #boxes = [ee.Geometry.Point([lon[i], lat[i]]).buffer(35).bounds() for i in iis]
    #GEOM = ee.Geometry.MultiPolygon(boxes)
    
    GEOM = ee.Geometry.Point([df['longitude'].iloc[i], df['latitude'].iloc[i]]).buffer(35).bounds()

    dataset = (
         ee.ImageCollection(collection) 
        .filterDate(start, end)
        .filterBounds(GEOM)
        #.map(maskS2clouds)
        .getRegion(GEOM, 10)
        # getRegion: Output an array of values for each [pixel, band, image] tuple in an ImageCollection. 
        # The output contains rows of id, lon, lat, time, and all bands for each image that intersects each pixel in the given region.
    )
    try:
        e = dataset.getInfo()
        fetched = pd.DataFrame(e[1:], columns=e[0])
        fetched['time'] = pd.to_datetime(fetched['time'], unit='ms')
        lons = pd.unique(fetched['longitude']).size
        lats = pd.unique(fetched['latitude']).size
        lon_counts[lons] = lon_counts.get(lons, 0) + 1
        lat_counts[lats] = lat_counts.get(lats, 0) + 1
        counts = fetched.groupby(['latitude', 'longitude']).count().shape[0]
        fetched['id'] = df.index[i] # index in bastin_cleaned.csv
        if retrieved is None:
            retrieved = fetched
        else:
            retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
        #df = pd.DataFrame(e)
        #df.to_csv(f'data/SKYSAT_{start}_{end}_{iis[-1]}.csv')
    except Exception as ex:
        print('Couldnt retrieve', ex)
        
    if i % 1000 == 0 and i != 0:
        print(f'{i}: sleeping a while after fetching 5000.')
        print(f'Fetched data in {time.time()-start} seconds')
        start = time.time()
        retrieved.to_csv(f'data/sentinel_{start}-{end}_{i}.csv')
        retrieved = None
        time.sleep(120)

# 689 secs for 300 data points
print(f'took {time.time()-t_start} seconds to retrieve data.')
print('lon counts:', lon_counts)
print('lat counts:', lat_counts)