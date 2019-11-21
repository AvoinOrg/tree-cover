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

import prepare.image_reshaper as ir

lib_extension_path = '/home/fynn/Apps/anaconda3/include/libsqlitefunctions.so'
bastin_db = pd.read_csv('data/bastin_db_cleaned.csv')
db_folder = 'data/sentinel/'

val_cols = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9","B11", "B12"]
csv_cols = ["id", "longitude", "latitude", "time", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA60"]
fetch_cols = ["id", "longitude", "latitude",  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL"]    

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
    funs = ['min', 'max', 'avg', 'stdev', 'median', 'lower_quartile', 'upper_quartile']
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
                # will any be invalid if I just join by date?
                inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
                if len(inv_ids) > 0:
                    raise(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
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


def compute_three_seasons_features():
    """
    computes the features from the raw postgres data and saves them as CSV. The columns are named as the query function,
    but with a _0, _1, _2 appended.
    """
    date_ranges = (('2018-12-01', '2019-02-28'), ('2019-03-01', '2019-05-31'), ('2019-06-01', '2019-08-31'))
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


# todo: standardise data, save the MinMaxScaler for training. Do I still have enough values if I filtering for clouds?
def prepare_image(t_start, t_end):
    reg = 'Australia'
    mode = 'interp'
    params = ['nearest']
    size = [7,7]
    include_ndvi = True
    ndvi_scaler = MinMaxScaler(feature_range=(0,255))
    missing = []
    
    img_cols = ["TCI_R", "TCI_G", "TCI_B"]
    stmt = "select id, longitude, latitude, " + ', '.join(f'median({b}) as {b}_m' for b in img_cols) 
    if include_ndvi:
        stmt += ", (1.0*median(B8)-median(B5))/(median(B8)+median(B5)) as NDVI_m"
    stmt += " from sentinel where "
    stmt += "id in ? and (HAS_CLOUDFLAG = 0 and MSK_CLDPRB is NULL or MSK_CLDPRB <0.1 and MSK_SNWPRB < 0.1) and "
    stmt += "date(time, 'unixepoch') between ? and ? group by id, longitude, latitude order by id, longitude, latitude"
    all_imgs = []
    with lite.connect(db_folder + reg + '.db') as con:
        con.enable_load_extension(True)
        con.load_extension(lib_extension_path)
        print(f'Computing features for {reg} from {t_start} to {t_end}')
        curr_stmt = stmt.replace('?', str(tuple(range(idx_start, idx_end))), 1)
        data_rows = con.execute(curr_stmt, (t_start, t_end)).fetchall()
        data_df = pd.DataFrame(data_rows).set_index([0],drop=True)
        for i in pd.unique(data_df.index):
            img_df = data_df.loc[i]
            img_df.iloc[:,-1] = ndvi_scaler.fit_transform(img_df.iloc[:,-1].to_numpy().reshape(-1,1)).round()
            img_arr = img_df.iloc[:, 2:].to_numpy(dtype=np.uint8)
            dim = (pd.unique(img_df.iloc[:, 0]).size, pd.unique(img_df.iloc[:,1]).size, img_arr.shape[1])
            # reshape into 8x7x13
            ok = False
            try:
                test = img_arr.reshape(dim)
                ok = True
            except ValueError:
                print(f'i={i}: could not reshape due to missing values because of cloud filter')
                missing.append(i)
                continue
            # convert into 7x7
            test2 = ir.reshape_image_array(test, mode, params, size)
            all_imgs.append(test2)
    print('missing due to cloud filter:', missing)