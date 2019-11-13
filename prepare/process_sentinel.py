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
        
def gen_fetch_stmt(fun, date_start, date_end):
    pass

def compute_features():
    with lite.connect(db_folder + area + '.db') as con:
        con.enable_load_extension(True)
        con.load_extension(lib_extension_path)
        # will any be invalid if I just join by date?
        inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
        if len(inv_ids) > 0:
            print('WARN: need to re-run again for ids around midnight: ', inv_ids)
        
        test_m = con.execute('SELECT median(B1) from sentinel').fetchone()[0]
            