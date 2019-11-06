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

# import elias_mail


def not_in_db(df, db):
    """ Takes the target data frame and the existing database object.
        Returns the splitted dataframe for existing and non-existant entries.
    """
    df_in = df[df.set_index(["longitude", "latitude"]).index.isin(db.set_index(["longitude", "latitude"]).index)]
    df_ex = df[~df.set_index(["longitude", "latitude"]).index.isin(db.set_index(["longitude", "latitude"]).index)]
    df_ex = df_ex.reset_index()
    df_in = df_in.reset_index()
    return df_ex, df_in


def extend_df_landsat(df, chunk_size=2):
    """ Needs generalization to incorporate multiple data sets.
        Currently adds landsat data.
    """
    cols = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "sr_aerosol", "pixel_qa", "radsat_qa"]
    i = chunk_size - 1
    for _ in cols:
        df[_] = None
    while i < len(df):
        gee_data = pd.read_csv(f"data/{i}_2015-01-01_2015-12-31.csv", sep=",", header=1)
        print(f"File: {i}_2015-01-01_2015-12-31.csv")
        # print(gee_data[cols])
        for j in range(i - chunk_size - 1, i + 1):
            dist = (gee_data.longitude - df.longitude[j]) ** 2 + (gee_data.latitude - df.latitude[j]) ** 2
            values = gee_data.loc[dist[dist == dist.min()].index, cols].median()
            for k in range(len(cols)):
                df[cols[k]].iloc[j] = values[k]
            print(values)

        i += min(chunk_size, max(len(df) - 1 - i, 1))
    return df


def single_point_populater(df, prep_method, aggregate_method, verbose=False):
    """ Takes single points of an arriving dataframe for a given fetching target.
        Prepares the data for a given method of preparation.
        Fetches the data point by point accordingly.
        Aggregates the single point data and adds it to the dataframe.
        Returns the populated dataframe.
    """
    if verbose:
        print("Running single_point_populater().")
    df, info = prep_method(df, verbose=verbose)
    points = zip(df.longitude, df.latitude)
    for lon, lat in points:
        fetched_data = single_fetch(lon, lat, collection=info["collection"], verbose=verbose)
        values = aggregate_method(fetched_data, lon, lat, info["fetch_cols"], verbose=verbose)
        df.loc[(df["latitude"] == lat) & (df["longitude"] == lon), info["agg_cols"]] = [values]
    return df


def landsat_aggregate_method(fetched_data, lon, lat, fetch_cols, verbose=False):
    if verbose:
        print("Running landsat_aggregate_method(lon={lon},lat={lat}).")
    dist = (fetched_data.longitude - lon) ** 2 + (fetched_data.latitude - lat) ** 2
    values = fetched_data.loc[dist[dist == dist.min()].index, fetch_cols].median()
    return values


def landsat_prepare_method(df, verbose=False):
    """ Takes any dataframe, gives it the respective columns 
        with default None values and returns the requested columns
        of landsat data. Also overrides old columns.
        Gives a fetching function the required information (columns)
        and prepares the dataframe into a clean format where the 
        columns will a) exist and b) be None.
    """
    if verbose:
        print("Running landsat_prepare_method().")
    info = {
        "collection": "LANDSAT/LC08/C01/T1_SR",
        "fetch_cols": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "sr_aerosol", "pixel_qa", "radsat_qa"],
        "agg_cols": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "sr_aerosol", "pixel_qa", "radsat_qa"]
    }
    for _ in info["agg_cols"]:
        df[_] = None
    return df, info


def single_fetch(lon, lat, start="2015-01-01", end="2015-12-31", collection="LANDSAT/LC08/C01/T1_SR", verbose=False):
    """ Fetches data for a single point given by lon(gitude) and lat(itude).
        Returns a pandas dataframe.
    """
    if verbose:
        print("f'Running single_fetch(lon={lon},lat={lat}).'")
    pt = ee.Geometry.Point([lon, lat]).buffer(35).bounds()
    GEOM = ee.Geometry.MultiPolygon([pt])

    dataset = (ee.ImageCollection(collection).filterDate(start, end).filterBounds(GEOM).getRegion(GEOM, 30)).getInfo()
    dataset = pd.DataFrame(dataset[1:], columns=dataset[0])
    return dataset


def main():
    os.chdir("/home/dario/_py/tree-cover")
    ee.Initialize()
    df = pd.read_csv("data/input.csv", sep=",")
    db = pd.read_csv("data/db.csv", sep=",")
    df_ex, df_in = not_in_db(df, db)
    # elias_mail.fetch_points(df_ex) #Does not finish?
    artificial_data = {"index": 1337, "longitude": 13.37, "latitude": 13.37}
    df_ex = df_ex.append(artificial_data, ignore_index=True)
    if len(df_ex) & len(df_ex) < 100:
        df_ex = single_point_populater(df_ex, landsat_prepare_method, landsat_aggregate_method, verbose=True)
        db.append(df_ex)
    elif: len(df_ex) >= 100: print("Data not in the database is too large. Please populate the database manually.")
    # Lets not overwrite our dummy data just yet.
    # db.to_csv('data/db.csv', sep=',')


if __name__ == "__main__":
    main()
