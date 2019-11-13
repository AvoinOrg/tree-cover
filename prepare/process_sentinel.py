# -*- coding: utf-8 -*-
"""
Methods to process the data retrieved by `fetch_sentinel.py`. You need to adjust:
    - location of compiled libsqlitefunctions.so
    - location of bastin-db and where the sql-dbs to process are
"""

import pandas as pd
import sqlite3 as lite
from pathlib import Path

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
    stmt = "select id"
    funs = ['min', 'max', 'avg', 'stdev', 'median', 'lower_quartile', 'upper_quartile']
    headers = []
    for band in val_cols + ['NDVI']:
        for fun in funs:
            stmt += f', {fun}({band}_m)'
            headers.append(f'{fun}({band}_m)')
    stmt += ", count(case SCL_m when 4 then 1 else null end)/count(SCL_m) as veg_pc from (select "
    stmt += "id, longitude, latitude, " + ', '.join(f'median({b}) as {b}_m' for b in val_cols) 
    stmt += ", mode(SCL) as SCL_m, (1.0*median(B8)-median(B5))/(median(B8)+median(B5)) as NDVI_m from sentinel where "
    stmt += "id = ? and (HAS_CLOUDFLAG = 0 or MSK_CLDPRB <0.1) and date(time, 'unixepoch') between "
    stmt += "? and ? group by id, longitude, latitude)"
    headers.append('veg_pc')
    return stmt, headers


t_start = '2018-12-01'
t_end = '2019-02-28'
def compute_features(t_start, t_end, export_file):
    """
    Computes min, max, mean, std, lower& upper quartile for the median values of the retrieved band values and the NDVI
    within the passed timeframe (as datestring 'YYYY-MM-DD'). Note that data is only available from ~2018-12-15 to
    2019-08-31. Currently used to extract the seasons within that timeframe.
    """
    region_to_bounds = {}
    ok_rows = 0
    err_rows = []
    # don't have data for all yet... # pd.unique(bastin_db.dryland_assessment_region)
    all_regs = ["Australia", "EastSouthAmerica", "NorthAmerica", "SouthernAfrica", "WestSouthAmerica"] 
    for region in all_regs:
        filtered = bastin_db[bastin_db["dryland_assessment_region"] == region]
        region_to_bounds[region] = (filtered.index.min(), filtered.index.max() + 1)
    fetch_stmt, new_cols = gen_fetch_stmt_and_headers()
    bastin_extended = bastin_db.reindex(bastin_db.columns.tolist() + new_cols,axis='columns', copy=True)
    for reg in all_regs:
        with lite.connect(db_folder + reg + '.db') as con:
            con.enable_load_extension(True)
            con.load_extension(lib_extension_path)
            print(f'Computing features for {reg} from {t_start} to {t_end}')
            # will any be invalid if I just join by date?
            inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
            if len(inv_ids) > 0:
                print(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
                continue
            for i in range(region_to_bounds[reg][0], region_to_bounds[reg][1]):
                try:
                    data_row = con.execute(fetch_stmt, (i, t_start, t_end)).fetchall()[0]
                    bastin_extended.loc[0, new_cols] = data_row[1:] # no need to re-set the ID
                    ok_rows+=1
                except Exception as e:
                    print(f'i: {i} - ', e)
                    err_rows.append(i)
                if i % 5000 == 0:
                    print('i:', i)
    print(f'Completed feature extraction for {ok_rows} rows.')
    if len(err_rows) > 0:
        print('Had errors for indices: ', err_rows)
    with_features = bastin_extended[new_cols].round(5)
    with_features.to_csv(db_folder + f'features_{t_start}-{t_end}.csv', sep=',')


#with_features = compute_features('2018-12-01', '2019-02-28')
#with_features.to_csv(export_file, sep=',')
