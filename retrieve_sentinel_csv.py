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

from .prepare.fetch_sentinel_data import retrieve_single_point, fetch_and_write_sqlite, stmt
from .prepare.process_sentinel import gen_fetch_stmt_and_headers, date_ranges
from .utils import timer


# Change these parameters to the next year if reusing exactly the same model or to the desired range
# if retrieving raw features for new model training
start = "2018-12-01"
end = "2019-08-31"
collection = "COPERNICUS/S2_SR"


def fetch_sqlite(f_name, df):
    """ fetches data from df, starting at the largest present id """ 
    db_exists = os.path.isfile(f_name)
    i=0
    with lite.connect(f_name) as con:
        if db_exists:
            last_id = con.execute('SELECT MAX(id) FROM sentinel').fetchone()[0]
            if last_id is not None:
                i = last_id+1
        else:
            con.execute(stmt)
        print('starting at index ', i)
        fetch_and_write_sqlite(con, df, start, end, i=i, i_max=df.shape[0], collection=collection)


@timer
def fetch_data(df, output, libsqlite, write_chunk=100):
    feature_stmt, columns = gen_fetch_stmt_and_headers()    
    if 'plot_id' not in df.columns:
        df['plot_id'] = df.index

    with lite.connect(":memory:") as con:
        con.execute(stmt)
        con.enable_load_extension(True)
        con.load_extension(libsqlite)

        ee.Initialize()
        i = 0
        n = df.shape[0]
        write_header = True
        if os.path.exists(output):
            i = pd.read_csv(output).shape[0]
            write_header = False
            print('Will continue from index', i)
            
        if write_chunk > n-i:
            write_chunk=n-i-1
            
        new_cols = [f"{c}_{i}" for i, _ in enumerate(date_ranges) for c in columns]
        feature_df = df.reindex(df.columns.tolist() + new_cols, axis='columns', copy=True)
        used_cols = [c for c in feature_df.columns if not c.startswith('veg_pc')]

        if output is not None:
            fetch_and_write_to_file(df, feature_df, i, n, write_chunk, feature_stmt, columns, con, output, used_cols, write_header)
        else:
            return fetch_and_return(df, feature_df, i, n, feature_stmt, columns, con, used_cols)


def fetch_and_return(df, feature_df, i, n, feature_stmt, columns, con, used_cols):
    err_cnt = 0
    retrieved = None
    while i < n:
        print('retrieving for i=', i)
        retrieved, err_cnt = retrieve_single_point(retrieved, i, err_cnt, df, collection, start, end)
        if retrieved is not None:
            retrieved.to_sql('sentinel', con, if_exists='append', index=False)
            feature_df = compute_features(feature_df, feature_stmt, columns, i - write_chunk, i, start, end, con)
            if err_cnt == 0:
                i += 1

    return feature_df[used_cols]


def fetch_and_write_to_file(df,feature_df, i, n, write_chunk, feature_stmt, columns, con, output, used_cols, write_header):
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
            curr_stmt = fetch_stmt.replace('?', str(tuple(range(idx_start, idx_end))), 1)
            data_rows = con.execute(curr_stmt, (t_start, t_end)).fetchall()
            data_df = pd.DataFrame(data_rows).set_index([0],drop=True).round(5)
            feature_df.loc[data_df.index, curr_cols] = data_df.to_numpy()

        except Exception as e:
            print(e)
            
    return feature_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve the SENTINEL features for tree cover prediction')
    parser.add_argument('infile', metavar='INPUT', help='input file in .csv format')
    parser.add_argument('--output', default='features.csv', 
                        help='output file path (default: features.csv)')
    parser.add_argument('--libsqlite', default='./libsqlitefunctions.so', 
                        help='path to the compiled `libsqlitefunction.so`. Also required without raw feature storage.')
    parser.add_argument('--db', default=None, 
                        help='Specify the path to the sqlite database for raw feature retrieval (default: None, only save them in the memory before writing to CSV)')
    parser.add_argument('--chunk', default=100, type=int, 
                        help='Computes (and writes features) from the raw data in memory after fetching (default: 100) points')

    args=parser.parse_args()

    if args.db is None:
        df = pd.read_csv(args.infile, usecols=["longitude", "latitude", "Aridity_zone", "tree_cover"])
        fetch_data(df, args.output, libsqlite=args.libsqlite, write_chunk=args.chunk)
    else:
        fetch_sqlite(args.db, args.infile)

    
    
