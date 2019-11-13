# -*- coding: utf-8 -*-
"""
Methods to process the data retrieved by `fetch_sentinel.py`:
    - brings the format in correct type, ignores irrelvant columns.
    - write to sqlite database. One DB per region.
    - expects all files to be in a folder called like the area & will write a db with that name
"""

area = 'NorthAmerica'

import pandas as pd
import sqlite3 as lite
from pathlib import Path

lib_extension_path = '/home/fynn/Apps/anaconda3/include/libsqlitefunctions.so'
bastin_db = pd.read_csv('data/bastin_db_cleaned.csv')
db_folder = 'data/sentinel/'
val_cols = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9","B11", "B12"]


# 1. drop QA columns, just take boolean column „cloud_qa”.
# todo: ever > 32767 must warn!!!
csv_cols = ["id", "longitude", "latitude", "time", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA60"]
fetch_cols = ["id", "longitude", "latitude",  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL"]    

def raw_df_to_db_db(df):
    df['HAS_CLOUDFLAG'] = (df['QA60'] == 1024.) | (df['QA60'] == 2048.)
    # https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
    df.longitude = df.longitude.round(7)
    df.latitude = df.latitude.round(7)
    df.time = pd.to_datetime(df.time).apply(lambda t: t.value//10**9)
    df.drop('QA60', axis=1, inplace=True)
    return df

def import_full_csv():
    stmt = """CREATE TABLE sentinel(id INT,longitude FLOAT, latitude FLOAT, time DATETIME, B1 SMALLINT, B2 SMALLINT, 
          B3 SMALLINT, B4 SMALLINT, B5 SMALLINT, B6 SMALLINT, B7 SMALLINT, B8 SMALLINT, B8A SMALLINT, B9 SMALLINT,
          B11 SMALLINT, B12 SMALLINT, AOT SMALLINT, WVP SMALLINT, SCL SMALLINT, TCI_R SMALLINT, TCI_G SMALLINT,
          TCI_B SMALLINT, MSK_CLDPRB SMALLINT, MSK_SNWPRB SMALLINT, HAS_CLOUDFLAG BOOLEAN)"""
        
    with lite.connect(db_folder + area + '.db') as con:
        # print(con.execute('select count(id) from sentinel').fetchone())
        # test = con.execute('select distinct(id) from sentinel').fetchall()
        con.execute(stmt)
        file = '' # put here
        
        for file in Path(Path.cwd(), db_folder, area).glob('**/*.csv'):
            df = pd.read_csv(file, usecols=csv_cols)
            df = raw_df_to_db_db(df)
            curr_max = df[val_cols].max().max()
            print(f'max {curr_max} for file {file}')
            df.to_sql('sentinel', con, if_exists='append', index=False)
        
        # only a good idea once the file is complete. Else, this slows down inserts drastically.
        con.execute('CREATE INDEX id_index ON sentinel(id)')
        con.commit()
        
def gen_fetch_stmt():
    stmt = "select id"
    funs = ['min', 'max', 'avg', 'stdev', 'median', 'lower_quartile', 'upper_quartile']
    for band in val_cols + ['NDVI']:
        for fun in funs:
            stmt += f', {fun}({band}_m)'
    stmt += ", count(case SCL_m when 4 then 1 else null end)/count(SCL_m) as veg_pc from (select "
    stmt += "id, longitude, latitude, " + ', '.join(f'median({b}) as {b}_m' for b in val_cols) 
    stmt += ", mode(SCL) as SCL_m, (1.0*median(B8)-median(B5))/(median(B8)+median(B5)) as NDVI_m from sentinel where "
    stmt += "id = ? and (HAS_CLOUDFLAG = 0 or MSK_CLDPRB <0.1) and date(time, 'unixepoch') between "
    stmt += "? and ? group by id, longitude, latitude)"
    return stmt


def compute_features(t_start, t_end):
    region_to_bounds = {}
    all_regs = pd.unique(bastin_db.dryland_assessment_region)
    for region in all_regs:
        filtered = bastin_db[bastin_db["dryland_assessment_region"] == region]
        region_to_bounds[region] = (filtered.index.min(), filtered.index.max() + 1)
    fetch_stmt = gen_fetch_stmt()
    for reg in all_regs:
        with lite.connect(db_folder + reg + '.db') as con:
            con.enable_load_extension(True)
            con.load_extension(lib_extension_path)
            print(f'Computing features for {reg} from {t_start} to {t_end}')
            # will any be invalid if I just join by date?
            inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
            if len(inv_ids) > 0:
                raise(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
            for i in range(region_to_bounds[reg][0], region_to_bounds[reg][1]):
                data_row = con.execute(fetch_stmt, (i, t_start, t_end)).fetchone()[0]
                return data_row

testrow = compute_features('2018-12-01', '2019-02-28')