# -*- coding: utf-8 -*-
"""
   -- input: A CSV file with the (at least) the columns `longitude`, `latitude` and `Aridity_zone`
   -- libsqlite: Path to the compiled `libsqlitefunction.so`. 

   Optional:
   -- output: A CSV file with these 3 columns + the required columns for the landsat predictor.
   -- db: sqlite db to which the data should be written
   
   This program fetches the raw data from Google Earth Engine and writes the necessary features into the output file.
   NOTE: It's smart to split the input data and run several instances in parallel (but you cannot write to the same db)
   With chunk size 10, it can run on the smalles GoogleCloudCompute instance.
"""
import ee
import pandas as pd
import os
import sqlite3 as lite
import argparse
from tqdm import tqdm
import time
import functools


# Change these parameters to the next year if reusing exactly the same model or to the desired range
# if retrieving raw features for new model training
start = "2018-12-01"
end = "2019-08-31"
collection = "COPERNICUS/S2_SR"

# you need to compile the included SQLite extension and then specify its location here.
# gcc -g -fPIC -shared extension-functions.c -o libsqlitefunctions.so -lm
lib_extension_path = './libsqlitefunctions.so'


val_cols = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9","B11", "B12"]
csv_cols = ["id", "longitude", "latitude", "time", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA60"]
fetch_cols = ["id", "longitude", "latitude",  "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "AOT", "WVP", "SCL"]

bands = val_cols + ["AOT", "WVP", "SCL", "TCI_R", "TCI_G", "TCI_B", "MSK_CLDPRB", "MSK_SNWPRB", "QA10", "QA20", "QA60"]
band_types = ["uint32"]*14 + ["uint8"]*6 + ["uint32"]*3

# for sqlite:
funs = ['min', 'max', 'avg', 'stdev', 'median', 'lower_quartile', 'upper_quartile']
date_ranges = (('2018-12-01', '2019-02-28'), ('2019-03-01', '2019-05-31'), ('2019-06-01', '2019-08-31'))
stmt = """CREATE TABLE sentinel(id INT,longitude FLOAT, latitude FLOAT, time DATETIME, B1 SMALLINT, B2 SMALLINT, 
       B3 SMALLINT, B4 SMALLINT, B5 SMALLINT, B6 SMALLINT, B7 SMALLINT, B8 SMALLINT, B8A SMALLINT, B9 SMALLINT,
       B11 SMALLINT, B12 SMALLINT, AOT SMALLINT, WVP SMALLINT, SCL SMALLINT, TCI_R SMALLINT, TCI_G SMALLINT,
       TCI_B SMALLINT, MSK_CLDPRB SMALLINT, MSK_SNWPRB SMALLINT, HAS_CLOUDFLAG BOOLEAN)"""


def timer(f):
    """ Add this @decorator to a function to print its runtime after completion """
    @functools.wraps(f)
    def t_wrap(*args, **kwargs):
        t_start = time.perf_counter()
        ret = f(*args, **kwargs)
        t_run = round((time.perf_counter() - t_start)/60)
        print(f"{f.__name__} completed in {t_run} minutes.")
        return ret
    return t_wrap


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


def fetch_sqlite(f_name, infile, libsqlite, outfile=None):
    """ fetches data from df, starting at the largest present id """ 
    db_exists = os.path.isfile(f_name)
    i=0
    df = pd.read_csv(infile)

    with lite.connect(f_name) as con:
        if db_exists:
            last_id = con.execute('SELECT MAX(id) FROM sentinel').fetchone()[0]
            if last_id is not None:
                i = last_id+1
                print('Found existing database ', f_name, 'continue from index', i)
        else:
            con.execute(stmt)
        con.enable_load_extension(True)
        con.load_extension(libsqlite)
        fetch_and_write_sqlite(con, df, start, end, i=i, i_max=df.shape[0], collection=collection)
        if outfile is not None:
            print(f'Now computing features for {df.shape[0]} rows from sqlite db: {f_name}, will save to: {outfile}')
            feature_stmt, columns = gen_fetch_stmt_and_headers()
            new_cols = [f"{c}_{j}" for j, _ in enumerate(date_ranges) for c in columns]
            feature_df = df.reindex(df.columns.tolist() + new_cols, axis='columns', copy=True)
            feature_df = compute_features(feature_df, feature_stmt, columns, 0, df.shape[0], start, end, con)
            feature_df.to_csv(outfile, header=True, index=False)
            print('Export completed.')


@timer
def fetch_data(df, output, libsqlite, write_chunk=100):
    feature_stmt, columns = gen_fetch_stmt_and_headers()    
    if 'plot_id' not in df.columns:
        df['plot_id'] = df.index

    with lite.connect(":memory:") as con:
        con.execute(stmt)
        con.enable_load_extension(True)
        con.load_extension(libsqlite)

        i = 0
        n = df.shape[0]

        new_cols = [f"{c}_{j}" for j, _ in enumerate(date_ranges) for c in columns]
        feature_df = df.reindex(df.columns.tolist() + new_cols, axis='columns', copy=True)
        used_cols = [c for c in feature_df.columns if not c.startswith('veg_pc')]

        if output is not None:
            fetch_and_write_to_file(df, feature_df, i, n, write_chunk, feature_stmt, columns, con, output, used_cols)
        else:
            return fetch_and_return(df, feature_df, i, n, feature_stmt, columns, con, used_cols)


def fetch_and_return(df, feature_df, i, n, feature_stmt, columns, con, used_cols):
    err_cnt = 0
    retrieved = None
    i_start = i
    while i < n:
        #print('retrieving for i=', i)
        retrieved, err_cnt = retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end)
        if err_cnt == 0:
            i += 1
    if retrieved is not None:
        retrieved.to_sql('sentinel', con, if_exists='append', index=False)
        feature_df = compute_features(feature_df, feature_stmt, columns, i_start, i, start, end, con)

    return feature_df[used_cols]


def fetch_and_write_to_file(df,feature_df, i, n, write_chunk, feature_stmt, columns, con, output, used_cols):
    write_header = True
    if os.path.exists(output):
        i = pd.read_csv(output).shape[0]
        write_header = False
        print('Will continue from index', i)

    if write_chunk > n - i:
        write_chunk = n - i - 1

    err_cnt = 0
    retrieved = None
    i_start = i

    with tqdm(total=n) as pbar:
        pbar.update(i)

        while i < n:
            # print('retrieving for i=', i)
            retrieved, err_cnt = retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end)

            if retrieved is not None and i % write_chunk == 0 and i != i_start:
                # print(f"{i}: writing fetched data to file.")
                retrieved.to_sql('sentinel', con, if_exists='append', index=False)
                feature_df = compute_features(feature_df, feature_stmt, columns, i - write_chunk, i, start, end, con)
                with open(output, 'a') as f:
                    feature_df[used_cols][i - write_chunk:i].to_csv(f, header=write_header, index=False, mode='a')
                    write_header = False
                retrieved = None

            if err_cnt == 0:
                pbar.update()
                i += 1

        if retrieved is not None:
            retrieved.to_sql('sentinel', con, if_exists='append', index=False)
            feature_df = compute_features(feature_df, feature_stmt, columns, i // write_chunk * write_chunk, i, start,
                                          end, con)
            with open(output, 'a') as f:
                feature_df[used_cols][i // write_chunk * write_chunk:i].to_csv(f, header=False, index=False, mode='a')
    print('Features exported to ', output)

            
def compute_features(feature_df, fetch_stmt, fetch_cols, idx_start, idx_end, t_start, t_end, con):

    for i, date_range in enumerate(date_ranges):
        curr_cols = [f"{c}_{i}" for c in fetch_cols]
        try:
            # TODO: will any be invalid if I just join by date?
            # inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
            # if len(inv_ids) > 0:
            #     raise(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
            range_tuple = tuple(range(idx_start, idx_end))
            if len(range_tuple) > 1:
                curr_stmt = fetch_stmt.replace('?', str(range_tuple), 1)
            else:
                curr_stmt = fetch_stmt.replace('in ?', f'is {idx_start}')
            data_rows = con.execute(curr_stmt, (t_start, t_end)).fetchall()
            data_df = pd.DataFrame(data_rows).set_index([0],drop=True).round(5)
            feature_df.loc[data_df.index, curr_cols] = data_df.to_numpy()

        except Exception as e:
            print('ERROR in sentinel.py -> compute_features')
            print(e)
            
    return feature_df


def clean_df(df):
    df['HAS_CLOUDFLAG'] = (df['QA60'] == 1024.) | (df['QA60'] == 2048.)
    # https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
    df.longitude = df.longitude.round(7)
    df.latitude = df.latitude.round(7)
    df.time = pd.to_datetime(df.time, unit="ms").apply(lambda t: t.value // 10 ** 9)
    df.drop(['QA10', 'QA20', 'QA60'], axis=1, inplace=True)
    return df


def retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end):
    """
    appends to the dataframe of retrieved items (retrieved) and returns the next index and the current error counter
    """
    GEOM = ee.Geometry.Point([df["longitude"].iloc[i], df["latitude"].iloc[i]]).buffer(35).bounds()
    #print('fetching: ', [df["longitude"].iloc[i], df["latitude"].iloc[i]])
    dataset = (
        ee.ImageCollection(collection)
            .filterDate(start, end)
            .filterBounds(GEOM)
            .cast(dict(zip(bands, band_types)), bands)
            .getRegion(GEOM, 10))
    try:
        e = dataset.getInfo()
        err_cnt = 0
        fetched = pd.DataFrame(e[1:], columns=e[0])
        fetched["id"] = df.index[i]  # index in bastin_cleaned.csv
        fetched = clean_df(fetched)
        #print('For i=', i, 'retrieved data of shape:', fetched.shape)
        if retrieved is None:
            retrieved = fetched
        else:
            retrieved = pd.concat((retrieved, fetched), axis=0, copy=False)
    except Exception as ex:
        print(f"i:{i} attempt {err_cnt + 1} failed with: ", ex)
        if "Too many values:" in str(ex):
            print("Continue with next image due to too much data. Todo: split and try again")
        else:
            time.sleep(2 ** (err_cnt + 1))
            err_cnt += 1
            if err_cnt > 9:
                print("Stopping execution, error occured 10 times")
                raise ex
    return retrieved, err_cnt


def fetch_and_write_sqlite(con, df, start, end, collection="COPERNICUS/S2_SR", i=0, i_max=None):
    """
    retrieves sentinel data and writes it to the SQLite DB
    con -- the sqlite connection object
    df  -- the dataframe with longitude and latitude column
    start -- start date as string
    end   -- end date as string

    Keyword arguments:
    collection -- image collection (default: COPERNICUS/S2_SR)
    i     -- start index (default: 0)
    i_max -- end index (default: length of the dataframe)
    """

    if i_max is None:
        i_max = df.shape[0]
    err_cnt = 0
    retrieved = None
    with tqdm(total=i_max - i) as pbar:
        pbar.update(i)
        while i < i_max:
            retrieved, err_cnt = retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end)

            if retrieved is not None and i % 10 == 0:
                retrieved.to_sql('sentinel', con, if_exists='append', index=False)
                retrieved = None
            if err_cnt == 0:
                i += 1
                pbar.update()

    if retrieved is not None and i % 10 != 0:
        retrieved.to_sql('sentinel', con, if_exists='append', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve the SENTINEL features for tree cover prediction')
    parser.add_argument('infile', metavar='INPUT', help='Input file in .csv format with columns longitude, latitude, Aridity_zone (, plot_id)')
    parser.add_argument('--output', default=None, help='Output file path for the csv with the features. May be left out if only raw data shall be fetched into the specified --db')
    parser.add_argument('--libsqlite', default='./libsqlitefunctions.so', 
                        help='path to the compiled `libsqlitefunction.so`. Also required without raw feature storage.')
    parser.add_argument('--db', default=None, 
                        help='Specify the path to the sqlite database for raw feature retrieval (default: None, only save them in the memory before writing to CSV)')
    parser.add_argument('--chunk', default=100, type=int, 
                        help='Computes (and writes features) from the raw data after fetching (default: 100) points')

    args=parser.parse_args()
    ee.Initialize()

    if args.db is None:
        df = pd.read_csv(args.infile, usecols=["longitude", "latitude", "Aridity_zone", "tree_cover"])
        fetch_data(df, args.output, libsqlite=args.libsqlite, write_chunk=args.chunk)
    else:
        fetch_sqlite(args.db, args.infile, args.libsqlite, outfile=args.output)

    
    
