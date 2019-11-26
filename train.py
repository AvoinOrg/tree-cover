""" Creates the boosting model based on the format in db.csv.
    To-Do:
        - Add gridsearch option for deeper training
            -- Shallow training (ie best previous hyper parameters)
                should be an option, currently those of the R model.
            -- Deeper training with automatic hyperparameter optimiziation
                should be available as well.
"""
from sklearn import ensemble as en
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import os
import joblib as jl
import matplotlib.pyplot as plt
from utils import timer

# global params as I'm too lazy to build a CLI
path = "data/df_empty_dummy.csv" # 'data/features_three_months_improved.parquet' #  
np.random.seed(42)
w_dir = '/home/dario/_py/tree-cover' # '.' # 
model_name = "model.joblib" # "model_sentinel_enhanced_svr_rbf_0.5.joblib" # 

do_train = True
do_transform = True # logarithmic transform of y
do_scale_X = False # use a MinMaxScaler to bring the data into a range bewteen -1 and 1
do_weight = False # assign a weight to each feature s.t. those occuring less frequent will have higher weights
do_stratify = False # only take an approximately equal amount for each tree-cover level into account
method = 'boost' # 'svr' # 
kernel = 'rbf' # for svr


bastin_cols = ['longitude','latitude','dryland_assessment_region','Aridity_zone','land_use_category','tree_cover']


def load_data(path, cols=None):
    if path.endswith('.csv'):
        df = pd.read_csv(path, sep=",")
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    if cols is None:
        cols = [col for col in df.columns if col not in set(bastin_cols)]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    t, X = df.tree_cover, df[cols]
    cat = X.columns[X.dtypes == "object"]
    X = pd.get_dummies(X,cat,drop_first=True)
    cover_to_count = df.groupby('tree_cover').count().iloc[:,0].to_dict()
    # t, X = sk.utils.shuffle(df.tree_cover, df[cols], random_state=1)
    # split = int(X.shape[0] * 0.9)
    # X_train, t_train = X[:split], t[:split]
    # X_test, t_test = X[split:], t[split:]
    # return X_train, t_train, X_test, t_test
    return t, X, cover_to_count, cols

def prep(X):
    cat = X.columns[X.dtypes == "object"]
    X = pd.get_dummies(X,cat,drop_first=True)
    return X

@timer
def train(X, t, gridsearch=False, weights=None):
    params = {
        "n_estimators": 550,
        "max_depth": 8,
        "min_samples_split": 3,
        "learning_rate": 0.01,
        "loss": "ls" # "huber", # 
    }
#    cat = OneHotEncoder()
#    X_num, X_cat = X.loc[:,X.dtypes!="object"], X.loc[:,X.dtypes=="object"]
#    X_cat = cat.fit_transform(X_cat)
    if gridsearch == True:
        params = {
            "n_estimators": [550],
            "max_depth": [10,12],
            "learning_rate": [0.01,0.03],
            "loss": ["ls"],
        }   # Hyperparameter tuning
        clf = en.GradientBoostingRegressor()
        cv = GridSearchCV(clf,params,cv=3, 
                          n_jobs=4, verbose=1)
        cv.fit(X,t)
        #os.rename('model.joblib', 'model.joblib.bk')
        jl.dump(cv, model_name)
        return cv 
    else:
        clf = en.GradientBoostingRegressor(**params)
        clf.fit(X, t, sample_weight=weights)
        #os.rename('model.joblib', 'model.joblib.bk')
        jl.dump(clf, model_name)
        return clf

@timer
def train_svr(X, y, weights=None):
    """ 
    trains based on svm. scales in len(y)^2, so only use with do_stratify=True. Advantage of SVM: useful in high-dim
    spaces, so might use it for the raw img data. Not scale invariant! Must scale x to [-1,+1] e.g.
    """ 
    svr = SVR(kernel=kernel, C=0.5, cache_size=1000, gamma='scale')
    svr.fit(X, y, sample_weight=weights)
    jl.dump(svr, model_name)
    return svr

def predict(X, model):
    # cat = OneHotEncoder()
    # X = cat.fit_transform(X)
    if model == None:
        p = np.random.uniform(size=X.shape[0])
    elif model == "load":
        model = jl.load(model_name)
        p = model.predict(X)
    else:
        p = model.predict(X)
    return p, model

def get_weights(cnt_dict, vec, n_total):
    """ returns the weights according to the frequency in vec s.t. each value of vec has the same avg weight """
    weights = np.zeros(vec.size)
    for val, cnt in cnt_dict.items():
        weights[vec==val] = cnt/n_total
    return weights

def stratify(cnt_dict, X, y, scale=4):
    """ return a subsample of X and y where each class only appears maximum `scale x the minimum count` """
    allowed_size = min(cnt_dict.values()) * scale
    indices = []
    for val, cnt in cnt_dict.items():
        if cnt < allowed_size:
            indices += y[y==val].index.to_list()
        else:
            indices += y[y==val].sample(allowed_size).index.to_list()
    return X.loc[indices], y.loc[indices]
    

def evaluate(p, y_train_pred, y_test, y_train, w_test, w_train):
    """ calculates RMSE, R^2, mean and median error; prints & plots error percentiles """
    if do_transform:
        # bt means back transformed
        y_t_bt = 1/(1+np.exp(-y_test))
        y_train_bt = 1/(1+np.exp(-y_train))
        y_train_pred_bt = 1/(1+np.exp(-y_train_pred))
        p_bt = 1/(1+np.exp(-p))
        diff = p_bt-y_t_bt
    else:
        diff = p-y_test
        y_t_bt = y_test
        p_bt = p
        y_train_bt = y_train
        y_train_pred_bt = y_train_pred
        
    # handle overshooting
    p_bt[p_bt>0.95] = 0.95
    p_bt[p_bt<0] = 0
    y_train_pred_bt[y_train_pred_bt>0.95] = 0.95
    y_train_pred_bt[y_train_pred_bt<0] = 0
    
    rmse = round(np.sqrt(mean_squared_error(y_t_bt, p_bt)), 4)
    r_squared = round(r2_score(y_t_bt, p_bt, sample_weight=w_test),4) 
    print(f"For model: ", model_name)
    print(f"Test set - RMSE: {rmse}, R^2: {r_squared}")
    
    # overfitting?
    rmse_train = round(np.sqrt(mean_squared_error(y_train_bt, y_train_pred_bt, sample_weight=w_train)),4)
    r_sq_train = round(r2_score(y_train_bt, y_train_pred_bt, sample_weight=w_train), 4)
    print(f"Training set - RMSE: {rmse_train}, R^2: {r_sq_train}")
    
    median = sorted(np.sqrt(diff**2))[int(len(diff)/2)]
    mean = sum(np.sqrt(diff**2))/len(diff)
    for i in range(0,100, 10):
        print(str(i)+"% percentile : " +str(sorted(np.sqrt(diff**2))[int((len(diff))*i/100)]))
    print(f'Median error: {median:4f}, Mean error: {mean:4f}')
    plt.plot(sorted(np.sqrt(y_t_bt)**2))
    plt.plot(sorted(np.sqrt(diff**2)))
    plt.show()


def main():
    os.chdir(w_dir)
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
    if path.endswith('.parquet'):
        t, X, cnt_dict, feat = load_data(path=path)
    else:
        t, X, cnt_dict, feat = load_data(path=path, cols=feat)
    
    if do_stratify:
        X, t = stratify(cnt_dict, X, t)
        
    if do_transform:
        t[t==0] = 0.0001
        t[t==1] = 0.9999
        t = np.log(t/(1-t))
        
    X = prep(X)
    
    if do_scale_X:
        mm_scaler = MinMaxScaler(feature_range=(-1,1))
        X.iloc[:,:] = mm_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, t, 
                                                        test_size=0.2,
                                                        random_state=42)
    w_train, w_test = None, None
    if do_weight:
        w_train = get_weights(cnt_dict, y_train, t.size)
        w_test = get_weights(cnt_dict, y_test, t.size)

    if do_train:
        if method == 'boost':
            model = train(X_train, y_train, weights=w_train)
        elif method == 'svr':
            model = train_svr(X_train, y_train, weights=w_train)
        p, model = predict(X_test, model=model)
        y_train_pred, _ = predict(X_train, model=model)
    else:
        p, model = predict(X_test, model="load")
        y_train_pred, _ = predict(X_train, model="load")
    
    evaluate(p, y_train_pred, y_test, y_train, w_test, w_train)
    
    if do_stratify:
        # check the error on the discarded samples
        if path.endswith('.parquet'):
            t_full, X_full, cnt_dict, feat = load_data(path=path)
        else:
            t_full, X_full, cnt_dict, feat = load_data(path=path, cols=feat)
        X_rest = X_full.loc[X_full.index.symmetric_difference(X.index)]
        y_rest = t_full.loc[X_full.index.symmetric_difference(X.index)]
        if do_transform:
            y_rest[y_rest==0] = 0.0001
            y_rest[y_rest==1] = 0.9999
            y_rest = np.log(y_rest/(1-y_rest))
        p_rest, _ = predict(X_rest, model=model)
        if do_transform:
            # bt means back transformed
            p_r_bt = 1/(1+np.exp(-p_rest))
            y_r_bt = 1/(1+np.exp(-y_rest))
        else:
            p_r_bt = p_rest
            y_r_bt = y_rest
        
        rmse_train = round(np.sqrt(mean_squared_error(y_r_bt, p_r_bt)),4)
        r_sq_train = round(r2_score(y_r_bt, p_r_bt), 4)
        print(f"On left out data - RMSE: {rmse_train}, R^2: {r_sq_train}")

    # print sorted feature importances
    if method == 'boost':
        feat_df=pd.DataFrame(zip(feat, model.feature_importances_), columns=['feature', 'importance'])
        feat_df.sort_values('importance', ascending=False, inplace=True)
        feat_df.reset_index(inplace=True)
        print(feat_df[['feature', 'importance']].to_string())
    

if __name__ == "__main__":
    main()
