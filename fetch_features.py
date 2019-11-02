""" Takes an arriving input.csv file and first checks if the respective 
    locations have already been fetched and reside in the db.csv.
    Otherwise proceed to fetch the points which are not yet in db.csv and
    add them to db.csv.
    Currently just fetches landsat data, more to come.
    Todo:
        - Generalize fetching to add more data sets and specific columns.
        - Fix the fetching function from elias
            -- Currently does not finish after fetching all the points.
                Probably some inifinite loop.
        - Improve performance overall
            -- currently raping pandas beauty 
                b/c I used R the last 2 semesters X_X
        
"""
import ee
import os
import pandas as pd
import backoff
import itertools
#import elias_mail

def not_in_db(df, db):
    """ Takes the target data frame and the existing database object.
        Returns the splitted dataframe for existing and non-existant entries.
    """
    df_ex = df[~(df[['longitude', 'latitude']]
            .isin(db[['longitude', 'latitude']]))
        .any(axis=1)]
    df_in = df[(df[['longitude', 'latitude']]
            .isin(db[['longitude', 'latitude']]))
        .any(axis=1)]
    df_ex = df_ex.reset_index()
    df_in = df_ex.reset_index()
    return df_ex, df_in

def extend_df_landsat(df, chunk_size = 2):
    """ Needs generalization to incorporate multiple data sets.
        Currently adds landsat data.
    """
    cols = ['B1','B2','B3','B4','B5','B6','B7','B10','B11','sr_aerosol',
            'pixel_qa', 'radsat_qa']
    i = chunk_size-1 
    for _ in cols:
        df[_] = None
    while i < len(df):
        gee_data = pd.read_csv(f"data/{i}_2015-01-01_2015-12-31.csv", sep=",", header=1)
        print(f"File: {i}_2015-01-01_2015-12-31.csv")
        #print(gee_data[cols])
        for j in range(i-chunk_size-1,i+1):
            dist = ((gee_data.longitude - df.longitude[j])**2 + (gee_data.latitude - df.latitude[j])**2 )
            values = gee_data.loc[dist[dist == dist.min()].index,cols].median()
            for k in range(len(cols)):
                df[cols[k]].iloc[j] = values[k]
            print(values)

        i += min(chunk_size, max(len(df)-1-i,1))
    return df

def main():
    os.chdir('/home/dario/_py/tree-cover')
    ee.Initialize()
    df = pd.read_csv("data/input.csv", sep=",")
    db = pd.read_csv("data/db.csv", sep=",")
    df_ex, df_in = not_in_db(df, db)
    # elias_mail.fetch_points(df_ex) #Does not finish?
    df_ex = extend_df_landsat(df_ex)
    db.append(df_ex)
    # Lets not overwrite our dummy data just yet.
    # db.to_csv('data/db.csv', sep=',') 

if __name__ == '__main__':
    main()

