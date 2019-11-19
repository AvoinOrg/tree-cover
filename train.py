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
import matplotlib.pyplot as plt

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
    #os.rename('model.joblib', 'model.joblib.bk')
    jl.dump(clf, "model_landsat_median_sds.joblib")
    return clf


def predict(X, model):
    # cat = OneHotEncoder()
    # X = cat.fit_transform(X)
    if model == None:
        p = np.random.uniform(size=X.shape[0])
    elif model == "load":
        model = jl.load("model_landsat_median_sds.joblib")
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
    #feat = ['dryland_assessment_region', 'Aridity_zone',
    #   'min(B1_m)', 'max(B1_m)',
    #   'avg(B1_m)', 'stdev(B1_m)', 'median(B1_m)', 'lower_quartile(B1_m)',
    #   'upper_quartile(B1_m)', 'min(B2_m)', 'max(B2_m)', 'avg(B2_m)',
    #   'stdev(B2_m)', 'median(B2_m)', 'lower_quartile(B2_m)',
    #   'upper_quartile(B2_m)', 'min(B3_m)', 'max(B3_m)', 'avg(B3_m)',
    #   'stdev(B3_m)', 'median(B3_m)', 'lower_quartile(B3_m)',
    #   'upper_quartile(B3_m)', 'min(B4_m)', 'max(B4_m)', 'avg(B4_m)',
    #   'stdev(B4_m)', 'median(B4_m)', 'lower_quartile(B4_m)',
    #   'upper_quartile(B4_m)', 'min(B5_m)', 'max(B5_m)', 'avg(B5_m)',
    #   'stdev(B5_m)', 'median(B5_m)', 'lower_quartile(B5_m)',
    #   'upper_quartile(B5_m)', 'min(B6_m)', 'max(B6_m)', 'avg(B6_m)',
    #   'stdev(B6_m)', 'median(B6_m)', 'lower_quartile(B6_m)',
    #    'upper_quartile(B6_m)', 'min(B7_m)', 'max(B7_m)', 'avg(B7_m)',
    #    'stdev(B7_m)', 'median(B7_m)', 'lower_quartile(B7_m)',
    #    'upper_quartile(B7_m)', 'min(B8_m)', 'max(B8_m)', 'avg(B8_m)',
    #    'stdev(B8_m)', 'median(B8_m)', 'lower_quartile(B8_m)',
    #    'upper_quartile(B8_m)', 'min(B8A_m)', 'max(B8A_m)', 'avg(B8A_m)',
    #    'stdev(B8A_m)', 'median(B8A_m)', 'lower_quartile(B8A_m)',
    #    'upper_quartile(B8A_m)', 'min(B9_m)', 'max(B9_m)', 'avg(B9_m)',
    #    'stdev(B9_m)', 'median(B9_m)', 'lower_quartile(B9_m)',
    #    'upper_quartile(B9_m)', 'min(B11_m)', 'max(B11_m)', 'avg(B11_m)',
    #    'stdev(B11_m)', 'median(B11_m)', 'lower_quartile(B11_m)',
    #    'upper_quartile(B11_m)', 'min(B12_m)', 'max(B12_m)', 'avg(B12_m)',
    #    'stdev(B12_m)', 'median(B12_m)', 'lower_quartile(B12_m)',
    #    'upper_quartile(B12_m)', 'min(NDVI_m)', 'max(NDVI_m)', 'avg(NDVI_m)',
    #    'stdev(NDVI_m)', 'median(NDVI_m)', 'lower_quartile(NDVI_m)',
    #    'upper_quartile(NDVI_m)', 'veg_pc']
    t, X = load_data(path=path, cols=feat)
    t[t==0] = 0.0001
    t[t==1] = 0.9999
    t = np.log(t/(1-t))
    X = prep(X)
    X_train, X_test, y_train, y_test = train_test_split(X, t, 
                                                        test_size=0.2,
                                                        random_state=42)
    do_train =0 
    if do_train:
        print(f'Starting training at {time.time()}')
        t_start = time.time()
        model = train(X_train, y_train)
        print(f"training model took {(time.time()-t_start)/60} minutes")
        p = predict(X_test, model=model)
    else:
        p = predict(X_test, model="load")
    rmse = round(np.sqrt(sum((p - y_test) ** 2) / len(p)) * 100, 4)
    #r_squared = round(model.score(X_test, y_test), 4) 
    print(f"RMSE in %: {rmse}, R^2: {r_squared}")
    diff = 1/(1+np.exp(-p))-1/(1+np.exp(-y_test))
    median = sorted(np.sqrt(diff**2))[int(len(diff)/2)]
    mean = sum(np.sqrt(diff**2))/len(diff)
    for i in range(0,100):
        print(str(i)+"% percentile : " +str(sorted(np.sqrt(diff**2))[int((len(diff))*i/100)]))
    plt.plot(sorted(np.sqrt((1/(1+np.exp(-y_test)))**2)))
    plt.plot(sorted(np.sqrt(diff**2)))
    plt.show()
   

if __name__ == "__main__":
    main()
