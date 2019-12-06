# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3 as lite
import numpy as np
import ee
import os
from treecover.sentinel import stmt, date_ranges, funs, val_cols, gen_fetch_stmt_and_headers, fetch_and_write_sqlite

"""
The methods in this file are not used anymore. compute_features is used in a similar manner in sentinel.py.
"""
bastin_db = pd.read_csv('data/bastin_db_cleaned.csv')
db_folder = 'data/sentinel/'
lib_extension_path = './libsqlitefunctions.so'


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


def run(area='NorternAfrica'):
    ee.Initialize()
    start = "2017-06-01"  # only available since 2018-12...
    end = "2019-08-31"

    regions = [
        "Australia",
        "CentralAsia",
        "EastSouthAmerica",
        "Europe",
        "HornAfrica",
        "MiddleEast",
        "NorthAmerica",
        "NorthernAfrica",
        "Sahel",
        "SouthernAfrica",
        "SouthWestAsia",
        "WestSouthAmerica",
    ]

    region_to_batch = {}
    bastin_df = pd.read_csv("data/bastin_db_cleaned.csv",
                            usecols=["longitude", "latitude", "dryland_assessment_region"])

    for region in regions:
        filtered = bastin_df[bastin_df["dryland_assessment_region"] == region]
        region_to_batch[region] = (filtered.index.min(), filtered.index.max() + 1)

    bastin_df.drop("dryland_assessment_region", axis=1, inplace=True)

    i = region_to_batch[area][0]
    f_name = f"data/sentinel/{area}.db"

    db_exists = os.path.isfile(f_name)
    with lite.connect(f_name) as con:
        if db_exists:
            last_id = con.execute('SELECT MAX(id) FROM sentinel').fetchone()[0]
            if last_id is not None:
                i = last_id + 1
        else:
            con.execute(stmt)
        print('starting at index ', i)
        fetch_and_write_sqlite(con, bastin_df, start, end, i=i, i_max=region_to_batch[area][1])


def compute_three_seasons_features():
    """
    computes the features from the raw postgres data and saves them. The columns are named as the query function,
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
    

def compute_monthly_features():
    """ 
    exports the vegetation index & band 1,5,9,12 - based features. Saves them both by month for the LSTM and aggregated
    for the GBR (min, max, median) in the hope it would pick up changes over the year from that.
    """
    bastin_db = pd.read_csv('data/bastin_db_cleaned.csv')
    region_to_bounds = {}
    calc_all = True
    err_rows = []
    all_regs = pd.unique(bastin_db.dryland_assessment_region)
    for region in all_regs:
        filtered = bastin_db[bastin_db["dryland_assessment_region"] == region]
        region_to_bounds[region] = (filtered.index.min(), filtered.index.max() + 1)
    fetch_stmt, new_cols = gen_temporal_fetch_stmt_and_headers(calc_all)
    min_cols = [f'{col}_MIN' for col in new_cols]
    max_cols = [f'{col}_MAX' for col in new_cols]
    med_cols = [f'{col}_MED' for col in new_cols]
    gbr_cols = min_cols + max_cols + med_cols

    ret_df = bastin_db.reindex(bastin_db.columns.tolist() + gbr_cols,axis='columns', copy=True)
    full_data = None
    for reg in all_regs:
        idx_start = region_to_bounds[reg][0]
        idx_end = region_to_bounds[reg][1]
        with lite.connect(db_folder + reg + '.db') as con:
            con.enable_load_extension(True)
            con.load_extension(lib_extension_path)
            print(f'Computing features for {reg}')
            try:
                # will any be invalid if I just join by date? -> no, luckily never happens for this set.
                # inv_ids = con.execute("select distinct(id) from sentinel where strftime('%H:%M', time, 'unixepoch') in('23:59','00:00')").fetchall()
                # if len(inv_ids) > 0:
                #     raise(f'WARN: need to re-run again for ids around midnight: {inv_ids}')
                curr_stmt = fetch_stmt.replace('?', str(tuple(range(idx_start, idx_end))), 1)
                data_rows = con.execute(curr_stmt).fetchall()
                data_df = pd.DataFrame(data_rows).set_index([0],drop=True).round(5)
                data_df.columns = ['year_month'] + new_cols
                
                # extract min, max, median from the fetched data, leaving out the first column (year_month)
                grouped = data_df.iloc[:,1:].groupby([0]) # id
                uniq_idx = pd.unique(data_df.index) # .id
                ret_df.loc[uniq_idx, min_cols] = grouped.min().to_numpy()
                ret_df.loc[uniq_idx, max_cols] = grouped.max().to_numpy()
                ret_df.loc[uniq_idx, med_cols] = grouped.median().to_numpy()

                data_df['id'] = data_df.index
                if full_data is None:
                    full_data = data_df
                else:
                    full_data = full_data.append(data_df, ignore_index=True)
                
            except Exception as e:
                print(reg, e)
                err_rows.append(reg)

    if len(err_rows) > 0:
        print('Had errors for regions: ', err_rows)
    ret_df.to_parquet('data/vegetation_index_features_aggregated_all.parquet', index=True)
    full_data.to_parquet('data/vegetation_index_features_full_all.parquet', index=False)
    print(f'Completed feature extraction.')

    
def gen_temporal_fetch_stmt_and_headers(use_all = False):
    """ fetches the statistics over the image region based on the monthly medians + the calculated indices """
    # blue: B2, green: B3, red: B4, NIR: B8
    stmt = "select id, year_month"
    headers = []
    cols = val_cols if use_all else  ["B1", "B5", "B9", "B12"]
    for band in cols  + ['ENDVI', "NDVI", "GDVI", "MSAVI2", "SAVI"]:
        for fun in ['min', 'max', 'avg','lower_quartile', 'upper_quartile', 'stdev']:
            stmt += f', {fun}({band}_m)'
            headers.append(f'{fun}({band}_m)')
    stmt += " from (select "
    stmt += "id, strftime('%Y-%m', time, 'unixepoch') as year_month, longitude, latitude, " 
    stmt += ', '.join(f'median({b}) as {b}_m' for b in cols) 
    stmt += ", (1.0*(median(B8)+median(B3))-2*median(B2)) / (median(B8)+median(B3)+2*median(B2)) as ENDVI_m"
    stmt += ", 1.0*(median(B8)-median(B4))/(median(B8)+median(B4)) as NDVI_m "
    stmt += ", 1.0*median(B8)-median(B3) as GDVI_m "
    stmt += ", 0.5*((2*median(B8)+1)-sqrt( square(2*median(B8)+1)-8*(median(B8)-median(B4) ))) as MSAVI2_m"
    stmt += ", 1.5*(median(B8)-median(B4))/(median(B8)+median(B4)+0.5) as SAVI_m" # L = 0.5
    stmt += " from sentinel where"
    stmt += " id in ? and (HAS_CLOUDFLAG = 0 and MSK_CLDPRB is null or MSK_CLDPRB <0.1 and MSK_SNWPRB < 0.1)"
    stmt += " group by year_month, id, longitude, latitude) group by id, year_month"
    return stmt, headers

"""
# not used -> but might be useful for learning on images.
def prepare_image_data(t_start, t_end, reg, idx_start, idx_end, full_df):
    # Modifies the passed df to append the RGB and NDVI values for the 8x7 pixels, scaled to [-1,1]
    import .image_reshaper as ir
    from sklearn.preprocessing import MinMaxScaler
    
    # initialise scaler manually.
    ndvi_scaler = MinMaxScaler(feature_range=(0,255))
    ndvi_scaler.scale_=255/2
    ndvi_scaler.data_min_=-1,
    ndvi_scaler.data_max=1
    ndvi_scaler.min_=255/2

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
            img_row = ndvi_scaler.inverse_transform(reshaped.reshape(1, 8*7*4))[0]
            full_df.iloc[i, 6:] = img_row
    print(f'{len(missing)} images are missing due to cloud filter: {missing}')


# wet season & summer in OZ
t_start = '2018-12-01'
t_end = '2019-02-28'
def generate_RGB_ndvi_raw_data():
    #generates the df with the raw band values scaled to -1, 1 for logistic regression. The columns are named according
    #to the schema: {band name}_{x_index}_{y_index} where x and y start at the top left corner of the image.
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
"""
        
