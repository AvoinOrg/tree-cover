""" Creates the boosting model based on the format in db.csv.
"""
import sklearn as sk
import numpy as np
import pandas as pd
import os
import itertools

def load_data(path,cols):
    df = pd.read_csv(path, sep=',')
    t, X = df.tree_cover, df[cols]
    #t, X = sk.utils.shuffle(df.tree_cover, df[cols], random_state=1)
    #split = int(X.shape[0] * 0.9)
    #X_train, t_train = X[:split], t[:split]
    #X_test, t_test = X[split:], t[split:]
    #return X_train, t_train, X_test, t_test
    return t, X

def train(features, targets)
    

def main():
    os.chdir('/home/dario/_py/tree-cover')
    path = 'data/db.csv'
    feat = ['dryland_assessment_region', 'Aridity_zone', 'land_use_category', 
            'tree_cover', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 
            'B11', 'sr_aerosol', 'pixel_qa', 'radsat_qa']
    X, t  = load_data(path=path, cols=feat)

if __name__ == '__main__':
    main()
