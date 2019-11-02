""" Creates the boosting model based on the format in db.csv.
    To-Do:
        - Add gridsearch option for deeper training
            -- Shallow training (ie best previous hyper parameters)
                should be an option, currently those of the R model.
            -- Deeper training with automatic hyperparameter optimiziation
                should be available as well.
"""
import sklearn as sk
from sklearn import ensemble as en
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import os
import itertools
import joblib as jl

def load_data(path,cols):
    df = pd.read_csv(path, sep=',')
    df = df.dropna()
    t, X = df.tree_cover, df[cols]
    #t, X = sk.utils.shuffle(df.tree_cover, df[cols], random_state=1)
    #split = int(X.shape[0] * 0.9)
    #X_train, t_train = X[:split], t[:split]
    #X_test, t_test = X[split:], t[split:]
    #return X_train, t_train, X_test, t_test
    return t, X

def train(features, targets, gridsearch=False):
    X, t = features, targets
    params = {'n_estimators': 2000, 'max_depth': 5, 'min_samples_split': 10,
          'learning_rate': 0.1, 'loss': 'ls'}
    #cat = OneHotEncoder()
    #X = cat.fit_transform(X)
    if gridsearch==True:
        # Hyperparameter tuning
        pass
    clf = en.GradientBoostingRegressor(**params)
    clf.fit(X, t)
    os.rename('model.joblib', 'model.joblib.bk')
    jl.dump(clf, 'model.joblib') 
    return clf

def predict(X, model):
    cat = OneHotEncoder()
    #X = cat.fit_transform(X)
    if model == None:
        p = np.random.uniform(size = X.shape[0])
    elif model == 'load':
        model = jl.load('model.joblib') 
        p = model.predict(X)
    else:
        p = model.predict(X)
    return p

def main():
    os.chdir('/home/dario/_py/tree-cover')
    path = 'data/db.csv'
    feat = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 
            'B11', 'sr_aerosol', 'pixel_qa', 'radsat_qa']
    t, X = load_data(path=path, cols=feat)
    model = train(features=X, targets=t)
    p = predict(X, model=model)
    print(f'RMSE in %: {round(np.sqrt(sum((p-t)**2)/len(p))*100,4)}')

if __name__ == '__main__':
    main()
