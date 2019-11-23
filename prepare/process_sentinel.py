# -*- coding: utf-8 -*-
"""
Methods to process the data retrieved by `fetch_sentinel.py`. You need to adjust:
    - location of compiled libsqlitefunctions.so
    - location of bastin-db and where the sql-dbs to process are
"""

import pandas as pd
import sqlite3 as lite
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from utils import timer 
import prepare.image_reshaper as ir

# gcc -g -fPIC -shared extension-functions.c -o libsqlitefunctions.so -lm
lib_extension_path = 'libsqlitefunctions.so'
bastin_db = pd.read_csv('data/bastin_db_cleaned.csv')
db_folder = 'data/sentinel/'

val_cols = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9","B11", "B12"]
csv_cols = ["id", "longitude", "latitude", "time", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA60"]
fetch_cols = ["id", "longitude", "latitude",  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL"]    

funs = ['min', 'max', 'avg', 'stdev', 'median', 'lower_quartile', 'upper_quartile']

# initialise scaler manually.
ndvi_scaler = MinMaxScaler(feature_range=(0,255))
ndvi_scaler.scale_=255/2
ndvi_scaler.data_min_=-1,
ndvi_scaler.data_max=1
ndvi_scaler.min_=255/2

date_ranges = (('2018-12-01', '2019-02-28'), ('2019-03-01', '2019-05-31'), ('2019-06-01', '2019-08-31'))
# if wanting to look at diff wet/dry for the data that I have
"""
region_to_wet = {    "Australia": 0,
    "CentralAsia",
    "EastSouthAmerica": 0,
    "Europe",
    "HornAfrica": 1,
    "MiddleEast",
    "NorthAmerica",
    "NorthernAfrica": 0,
    "Sahel",
    "SouthernAfrica",
    "SouthWestAsia",
    "WestSouthAmerica": 0,}
region_to_try = {}
"""

region_to_dry = {}

def raw_df_to_db_db(df):
    """
    Used for the first iteration of data retrieval. Not relevant anymore.
    """
    df['HAS_CLOUDFLAG'] = (df['QA60'] == 1024.) | (df['QA60'] == 2048.)
    # https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
    df.longitude = df.longitude.round(7)
    df.latitude = df.latitude.round(7)
    df.time = pd.to_datetime(df.time).apply(lambda t: t.value//10**9)
    df.drop('QA60', axis=1, inplace=True)
    return df

def import_full_csv():
    """
    Imports the fetched CSV to a sqlite3 db. Not needed anymore.
    """
    area = 'NorthAmerica'
    stmt = """CREATE TABLE sentinel(id INT,longitude FLOAT, latitude FLOAT, time DATETIME, B1 SMALLINT, B2 SMALLINT, 
          B3 SMALLINT, B4 SMALLINT, B5 SMALLINT, B6 SMALLINT, B7 SMALLINT, B8 SMALLINT, B8A SMALLINT, B9 SMALLINT,
          B11 SMALLINT, B12 SMALLINT, AOT SMALLINT, WVP SMALLINT, SCL SMALLINT, TCI_R SMALLINT, TCI_G SMALLINT,
          TCI_B SMALLINT, MSK_CLDPRB SMALLINT, MSK_SNWPRB SMALLINT, HAS_CLOUDFLAG BOOLEAN)"""
        
    with lite.connect(db_folder + area + '.db') as con:
        con.execute(stmt)
        for file in Path(Path.cwd(), db_folder, area).glob('**/*.csv'):
            df = pd.read_csv(file, usecols=csv_cols)
            df = raw_df_to_db_db(df)
            curr_max = df[val_cols].max().max()
            print(f'max {curr_max} for file {file}')
            df.to_sql('sentinel', con, if_exists='append', index=False)
        
        # only a good idea once the file is complete. Else, this slows down inserts drastically.
        con.execute('CREATE INDEX id_index ON sentinel(id)')
        con.commit()
        
def gen_fetch_stmt_and_headers():
    stmt = "select id, "
    headers = []
    for band in val_cols + ['NDVI']:
        for fun in funs:
            stmt += f'{fun}({band}_m), '
            headers.append(f'{fun}({band}_m)')
    stmt += "count(case SCL_m when 4 then 1 else null end)/count(SCL_m) as veg_pc from (select "
    stmt += "id, longitude, latitude, " + ', '.join(f'median({b}) as {b}_m' for b in val_cols) 
    stmt += ", mode(SCL) as SCL_m, (1.0*median(B8)-median(B5))/(median(B8)+median(B5)) as NDVI_m from sentinel where "
    stmt += "id in ? and (HAS_CLOUDFLAG = 0 and MSK_CLDPRB is null or MSK_CLDPRB <0.1 and MSK_SNWPRB < 0.1) and "
    stmt += "date(time, 'unixepoch') between ? and ? group by id, longitude, latitude) group by id"
    headers.append('veg_pc')
    return stmt, headers

def compute_features(t_start, t_end, part=None):
    """
    Computes min, max, mean, std, lower& upper quartile for the median values of the retrieved band values and the NDVI
    within the passed timeframe (as datestring 'YYYY-MM-DD'). Note that data is only available from ~2018-12-15 to
    2019-08-31. Currently used to extract the seasons within that timeframe.
    """
    region_to_bounds = {}
    err_rows = []
    all_regs = pd.unique(bastin_db.dryland_assessment_region)
    for region in all_regs:
        filtered = bastin_db[bastin_db["dryland_assessment_region"] == region]
        region_to_bounds[region] = (filtered.index.min(), filtered.index.max() + 1)
    fetch_stmt, new_cols = gen_fetch_stmt_and_headers()
    if part is not None:
        new_cols = [f"{c}_{part}" for c in new_cols]
    reg_abbr = []
    ret_df = bastin_db.reindex(bastin_db.columns.tolist() + new_cols,axis='columns', copy=True)
    ret_df.drop(columns=bastin_db.columns, inplace=True)
    for reg in all_regs:
        reg_abbr.append(''.join(x for x in reg if x.isupper()))
        idx_start = region_to_bounds[reg][0]
        idx_end = region_to_bounds[reg][1]
        with lite.connect(db_folder + reg + '.db') as con:
            con.enable_load_extension(True)
            con.load_extension(lib_extension_path)
            print(f'Computing features for {reg} from {t_start} to {t_end}')
            try:
                # will any be invalid if I just join by date? -> no, luckily never happens for this set.
                # inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
                # if len(inv_ids) > 0:
                #     raise(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
                curr_stmt = fetch_stmt.replace('?', str(tuple(range(idx_start, idx_end))), 1)
                data_rows = con.execute(curr_stmt, (t_start, t_end)).fetchall()
                data_df = pd.DataFrame(data_rows).set_index([0],drop=True).round(5)
                ret_df.loc[data_df.index, new_cols] = data_df.to_numpy()

            except Exception as e:
                print(reg, e)
                err_rows.append(reg)

    print(f'Completed feature extraction from {t_start} to {t_end}')
    if len(err_rows) > 0:
        print('Had errors for regions: ', err_rows)
    return ret_df

@timer
def compute_three_seasons_features():
    """
    computes the features from the raw postgres data and saves them as CSV. The columns are named as the query function,
    but with a _0, _1, _2 appended.
    """
    all_cols = []
    all_dfs = []
    for i, date_range in enumerate(date_ranges):
        date_df = compute_features(date_range[0], date_range[1], i)
        all_cols += date_df.columns.to_list()
        all_dfs.append(date_df)
    
    bastin_extended = bastin_db.reindex(bastin_db.columns.tolist() + all_cols,axis='columns', copy=True)
    for date_df in all_dfs:
        bastin_extended[date_df.columns] = date_df.to_numpy()
    bastin_extended.to_parquet(db_folder + f'features_three_months_full.parquet')
    
# todo: scale them all -> but better do that in the train method.
def enhance_three_seasons_features():
    """ using the importances from the boosted tree: exclude useless features, calculated some differences. """
    orig_df = pd.read_parquet('data/features_three_months_full.parquet')
    
    # these were not so useful
    drop_avg_q = [c for c in orig_df.columns if ('quartile' in c or 'avg' in c) and not any(b in c for b in ('B12', 'B4', 'NDVI'))]
    drop_bands = [c for c in orig_df.columns if any(b in c for b in ('veg_pc', 'B7', 'B8', 'B11', 'B6'))]
    improved_df = orig_df[orig_df.columns.symmetric_difference(drop_avg_q+drop_bands)]
    
    
    improved_df = pd.read_parquet('data/features_three_months_improved.parquet')
    # for the most useful bands, add global diff
    for b in ('B1', 'B2', 'B12', 'B4', 'NDVI', 'B3'):
        for f in funs:
            if b in ('B1', 'B2', 'B3') and f in ('lower_quartile', 'upper_quartile', 'avg'):
                continue
            row_names = [f'{f}({b}_m)_{i}' for i in range(3)]
            improved_df.loc[:,f'diff_{f}({b})'] = improved_df.apply(lambda row: np.abs(np.max(row[row_names])-np.min(row[row_names])), axis=1)
    improved_df.to_parquet('data/features_three_months_improved.parquet')    

def prepare_image_data(t_start, t_end, reg, idx_start, idx_end, full_df):
    """ Modifies the passed df to append the RGB and NDVI values for the 8x7 pixels, scaled to [-1,1] """ 
    mode = 'interp'
    params = ['nearest']
    size = [8,7]
    include_ndvi = True
    missing = []

    img_cols = ["TCI_R", "TCI_G", "TCI_B"]
    stmt = "select id, longitude, latitude, " + ', '.join(f'median({b}) as {b}_m' for b in img_cols) 
    if include_ndvi:
        stmt += ", (1.0*median(B8)-median(B5))/(median(B8)+median(B5)) as NDVI_m"
    stmt += " from sentinel where "
    stmt += "id in ? and (HAS_CLOUDFLAG = 0 and MSK_CLDPRB is NULL or MSK_CLDPRB <0.1 and MSK_SNWPRB < 0.1) and "
    stmt += "date(time, 'unixepoch') between ? and ? group by id, longitude, latitude order by id, longitude, latitude"
    
    with lite.connect(db_folder + reg + '.db') as con:
        con.enable_load_extension(True)
        con.load_extension(lib_extension_path)
        print(f'Computing features for {reg} from {t_start} to {t_end}')
        curr_stmt = stmt.replace('?', str(tuple(range(idx_start, idx_end))), 1)
        data_rows = con.execute(curr_stmt, (t_start, t_end)).fetchall()
        data_df = pd.DataFrame(data_rows).set_index([0],drop=True)
        for i in pd.unique(data_df.index):
            img_df = data_df.loc[i]
            try:
                img_df.iloc[:,-1] = ndvi_scaler.transform(img_df.iloc[:,-1].to_numpy().reshape(-1,1)).round()
                img_arr = img_df.iloc[:, 2:].to_numpy(dtype=np.uint8)
                dim = (pd.unique(img_df.iloc[:, 0]).size, pd.unique(img_df.iloc[:,1]).size, img_arr.shape[1])
                test = img_arr.reshape(dim)
            except (ValueError, pd.core.indexing.IndexingError): # or did sth else go wrong if just 1 value???
                print(f'i={i}: could not reshape due to missing values because of cloud filter')
                missing.append(i)
                continue
            # convert into 8x7 if necessary
            reshaped = ir.reshape_image_array(test, mode, params, size)
            img_row = prepare_img_for_svm(reshaped)
            full_df.iloc[i, 6:] = img_row
    print(f'{len(missing)} images are missing due to cloud filter: {missing}')

def prepare_img_for_svm(img: np.array):
    """ transforms the image into single values for each pixel in range [-1,1] """ 
    return ndvi_scaler.inverse_transform(img.reshape(1, 8*7*4))[0]
    
# wet season & summer in OZ
t_start = '2018-12-01'
t_end = '2019-02-28'
@timer
def generate_RGB_ndvi_raw_data():
    """ 
    generates the df with the raw band values scaled to -1, 1 for logistic regression. The columns are named according
    to the schema: {band name}_{x_index}_{y_index} where x and y start at the top left corner of the image.
    """
    bands = ["R", "G", "B", "ndvi"]
    col_names = []
    for lon_idx in range(8):
        for lat_idx in range(7):
            for band in bands:
                col_names.append(f'{band}_{lon_idx}_{lat_idx}')
        
    region_to_bounds = {}
    # let's take regions during wet season for starters.
    all_regs = ['Australia', 'SouthernAfrica', 'NorthernAfrica', "EastSouthAmerica", "WestSouthAmerica"]  
    # pd.unique(bastin_db.dryland_assessment_region)
    for region in all_regs:
        filtered = bastin_db[bastin_db["dryland_assessment_region"] == region]
        region_to_bounds[region] = (filtered.index.min(), filtered.index.max() + 1)

    ret_df = bastin_db.reindex(bastin_db.columns.tolist() + col_names,axis='columns', copy=True)
    for reg in all_regs:
        idx_start = region_to_bounds[reg][0]
        idx_end = region_to_bounds[reg][1]
        prepare_image_data(t_start, t_end, reg, idx_start, idx_end, ret_df)
    ret_df.to_parquet(db_folder + f'features_WetSeason_test.parquet')

        
generate_RGB_ndvi_raw_data()
    