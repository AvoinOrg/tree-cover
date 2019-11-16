""" Creates the boosting model based on the format in db.csv.
    To-Do:
        - Add gridsearch option for deeper training
            -- Shallow training (ie best previous hyper parameters)
                should be an option, currently those of the R model.
            -- Deeper training with automatic hyperparameter optimiziation
                should be available as well.
"""
from sklearn import ensemble as en
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import joblib as jl
import time

# global params as I'm too lazy to build a CLI
do_train = False
np.random.seed(42)
w_dir = '/home/dario/_py/tree-cover'


def load_data(path, cols):
    df = pd.read_csv(path, sep=",")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    t, X = df.tree_cover, df[cols]
    cat = X.columns[X.dtypes == "object"]
    X = pd.get_dummies(X,cat,drop_first=True)
    # t, X = sk.utils.shuffle(df.tree_cover, df[cols], random_state=1)
    # split = int(X.shape[0] * 0.9)
    # X_train, t_train = X[:split], t[:split]
    # X_test, t_test = X[split:], t[split:]
    # return X_train, t_train, X_test, t_test
    return t, X

def prep(X):
    cat = X.columns[X.dtypes == "object"]
    X = pd.get_dummies(X,cat,drop_first=True)
    return X


def train(X, t, gridsearch=False):
    params = {
        "n_estimators": 2000,
        "max_depth": 5,
        "min_samples_split": 3,
        "learning_rate": 0.1,
        "loss": "ls",
    }  # try: 'lad', 'criterion': 'mae'
#    cat = OneHotEncoder()
#    X_num, X_cat = X.loc[:,X.dtypes!="object"], X.loc[:,X.dtypes=="object"]
#    X_cat = cat.fit_transform(X_cat)
    if gridsearch == True:
        # Hyperparameter tuning
        pass
    clf = en.GradientBoostingRegressor(**params)
    clf.fit(X, t)
    os.rename('model.joblib', 'model.joblib.bk')
    jl.dump(clf, "model.joblib")
    return clf


def predict(X, model):
    # cat = OneHotEncoder()
    # X = cat.fit_transform(X)
    if model == None:
        p = np.random.uniform(size=X.shape[0])
    elif model == "load":
        model = jl.load("model.joblib")
        p = model.predict(X)
    else:
        p = model.predict(X)
    return p


def main():
    os.chdir(w_dir)
    path = "data/df_empty_dummy.csv"
    feat = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "sr_aerosol", "pixel_qa",
            "radsat_qa", "B1_sd", "B2_sd", "B3_sd", "B4_sd", "B5_sd", "B6_sd", "B7_sd", "B10_sd",
            "B11_sd", "sr_aerosol_sd", "pixel_qa_sd", "radsat_qa_sd",
            "dryland_assessment_region", "Aridity_zone"]
    t, X = load_data(path=path, cols=feat)
    X = prep(X)
    X_train, X_test, y_train, y_test = train_test_split(X, t, 
                                                        test_size=0.2,
                                                        random_state=42)
    do_train = 1
    if do_train:
        print(f'Starting training at {time.time()}')
        t_start = time.time()
        model = train(X_train, y_train)
        print(f"training model took {(time.time()-t_start)/60} minutes")
        p = predict(X_test, model=model)
    else:
        p = predict(X_test, model="load")
    rmse = round(np.sqrt(sum((p - y_test) ** 2) / len(p)) * 100, 4)
    r_squared = round(model.score(X_test, y_test), 4)
    print(f"RMSE in %: {rmse}, R^2: {r_squared}")


if __name__ == "__main__":
    main()
