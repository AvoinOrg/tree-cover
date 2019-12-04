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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import joblib as jl
import matplotlib.pyplot as plt
from utils import timer

#plt.rcParams["figure.figsize"] = (30,12)
#plt.rcParams["font.size"] = 20

# global params as I'm too lazy to build a CLI
path = 'data/features_three_months_full.parquet' # 'data/features_three_months_improved.parquet' # 'data/vegetation_index_features_aggregated_all.parquet' # 'data/vegetation_index_features_aggregated.parquet' # "data/df_empty_dummy.csv" #   
np.random.seed(42)
w_dir = '.' # '/home/dario/_py/tree-cover' # 
model_name = "model_sentinel_logtrans_stratified_huber_3months_2000_60leaves.joblib" #  "model_sentinel_logtrans_stratified_mae_allveg_lgbm_depth12_2000.joblib" # 


do_train = False
do_transform = True # logarithmic transform of y
do_stratify = True # only take an approximately equal amount for each tree-cover level into account
use_lgbm = True # faster than sklearn
target= 'tree_cover' # 'land_use_category' # 

do_gridsearch = False
do_scale_X = False # use a MinMaxScaler to bring the data into a range bewteen -1 and 1 -> no need.
do_weight = False # assign a weight to each feature s.t. those occuring less frequent will have higher weights
method = 'boost' # 'svr' # 
kernel = 'rbf' # for svr

# params for lgb:
objective= 'huber' # 'mean_absolute_error'


# cols to be dropped from the training data. Aridity Zone can be kept.
bastin_cols = ['longitude','latitude','dryland_assessment_region','land_use_category','tree_cover'] # 'Aridity_zone'


def load_data(path, cols=None):
    if path.endswith('.csv'):
        df = pd.read_csv(path, sep=",")
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    if cols is None:
        cols = [col for col in df.columns if col not in set(bastin_cols) and not col.startswith('veg_pc')]
        
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    t, X = df.tree_cover, df[cols]
    
    if use_lgbm:
        if 'Aridity_zone' in X.columns:
            X.Aridity_zone = pd.Categorical(X.Aridity_zone)
        if 'dryland_assessment_region' in X.columns:
            X.dryland_assessment_region = pd.Categorical(X.dryland_assessment_region)
    else:
        X = prep(X)
    
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
    print(f'Now training {method} model for data {path} with parameters:')
    print(f'do_transform = {do_transform}, do_scale_X = {do_scale_X}, do_weight = {do_weight}, do_stratify = {do_stratify}')
    if do_train and os.path.exists(model_name):
        print(f'Warning: model {model_name} already exists and will be overwritten after training!')
    else:
        print('Model will be saved as:', model_name)
        # warn if present!!!
    
    if gridsearch == True:
        if use_lgbm:
            depth = 12
            # score: 0.38406944683157623
            # best for depth 8: {'min_data_in_leaf': 50, 'n_estimators': 1000, 'num_leaves': 64}
            # run with mae: {'min_data_in_leaf': 60, 'n_estimators': 2000, 'num_leaves': 32}
            param_grid = {
                'num_leaves': [16, 24, 32, 40, 50, 62, 80] #[2**(depth-3), 2**(depth-2), 2**(depth-1)],
            }
            clf = lgb.LGBMRegressor(objective=objective, boosting_type='dart', alpha=0.9, learning_rate=0.1, 
                                    random_state=42, subsample=1.0, n_estimators=2000, min_child_samples=50)
            cv = GridSearchCV(clf, param_grid, n_jobs=2, cv=3, verbose=1, scoring='neg_mean_absolute_error')
        else:
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
        jl.dump(cv, model_name)
        return cv 
    else:
        if target=='land_use_category':
            kwargs = dict(loss='deviance', criterion='friedman_mse', init=None,
                          learning_rate=0.01, max_depth=8,  n_estimators=550,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0,
                          n_iter_no_change=None, presort='auto',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
            clf = en.GradientBoostingClassifier(**kwargs) if not use_lgbm else lgb.LGBMClassifier(**kwargs)
        else:
            if use_lgbm:
                kwargs = dict(
                        objective=objective, # regression, regression_l1, ...
                        boosting_type='dart',
                        alpha=0.9,
                        learning_rate=0.1, max_depth=-1, num_leaves=60,
                        min_child_samples=50,
                        n_estimators=2000,
                        random_state=42, subsample=1.0)
            else:
                kwargs = dict(alpha=0.9, criterion='friedman_mse', init=None,
                              learning_rate=0.1, loss='huber', max_depth=5, # rate 0.01, depth 8
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=3, # split 2
                              min_weight_fraction_leaf=0.0, n_estimators=200,
                              n_iter_no_change=None, presort='auto',
                              random_state=None, subsample=1.0, tol=0.0001,
                              validation_fraction=0.1, verbose=0, warm_start=False)

            clf = en.GradientBoostingRegressor(**kwargs) if not use_lgbm else lgb.LGBMRegressor(**kwargs)

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
    

def evaluate(p, y_train_pred, y_test, y_train, w_test, w_train, second_run=False):
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
    if second_run:
        print('\n--- Run on data removed by stratification ---')
    print(f"For model: ", model_name)
    print(f"Test set - RMSE: {rmse}, R^2: {r_squared}")
    
    # overfitting?
    rmse_train = round(np.sqrt(mean_squared_error(y_train_bt, y_train_pred_bt, sample_weight=w_train)),4)
    r_sq_train = round(r2_score(y_train_bt, y_train_pred_bt, sample_weight=w_train), 4)
    if not second_run:
        print(f"Training set - RMSE: {rmse_train}, R^2: {r_sq_train}")
    
    median = sorted(np.sqrt(diff**2))[int(len(diff)/2)]
    mean = sum(np.sqrt(diff**2))/len(diff)
    for i in range(0,100, 10):
        print(str(i)+"% percentile : " +str(round(sorted(np.sqrt(diff**2))[int((len(diff))*i/100)], 3)))
    print(f'Median error: {median:4f}, Mean error: {mean:4f}')
    plt.plot(sorted(np.sqrt(y_t_bt)**2))
    plt.plot(sorted(np.sqrt(diff**2)))
    plt.show()
    
    # how is it performing on the different tree cover estimates?
    print('MAE on the different buckets:')
    for i in range(0,10):
        denom = len(diff[np.logical_and((i/10 + 0.1) > y_t_bt, (i/10) < y_t_bt)])
        if denom == 0:
            continue
        nom = sum(np.abs(diff[np.logical_and((i/10 + 0.1) > y_t_bt, (i/10) < y_t_bt)]))
        print(f'{i}: {round(nom/denom,3)}')
        

def main():
    os.chdir(w_dir)
    feat = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11", "sr_aerosol", "pixel_qa",
            "radsat_qa", "B1_sd", "B2_sd", "B3_sd", "B4_sd", "B5_sd", "B6_sd", "B7_sd", "B10_sd",
            "B11_sd", "sr_aerosol_sd", "Aridity_zone"] # "pixel_qa_sd", "radsat_qa_sd", "dryland_assessment_region", 

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
        
    if do_scale_X:
        mm_scaler = MinMaxScaler(feature_range=(-1,1))
        X.iloc[:,:] = mm_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, t, 
                                                        test_size=0.2,
                                                        random_state=42)
    
    # df.loc[X_train.sample(500).index].to_csv('testset_part.csv', index=False)
    w_train, w_test = None, None
    if do_weight:
        w_train = get_weights(cnt_dict, y_train, t.size)
        w_test = get_weights(cnt_dict, y_test, t.size)

    if do_train:
        if method == 'boost':
            model = train(X_train, y_train, weights=w_train, gridsearch=do_gridsearch)
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
        
        # p, y_train_pred, y_test, y_train
        evaluate(p_r_bt, y_train_pred , y_r_bt, y_train, w_test, w_train, True)
        #rmse_train = round(np.sqrt(mean_squared_error(y_r_bt, p_r_bt)),4)
        #r_sq_train = round(r2_score(y_r_bt, p_r_bt), 4)
        #print(f"On left out data - RMSE: {rmse_train}, R^2: {r_sq_train}")

    # print sorted feature importances
    if method == 'boost':
        feat_df=pd.DataFrame(zip(feat, model.feature_importances_), columns=['feature', 'importance'])
        feat_df.sort_values('importance', ascending=False, inplace=True)
        feat_df.reset_index(inplace=True)
        print(feat_df[['feature', 'importance']].to_string())
        
        n=25
        feat_df[['feature', 'importance']][:n].plot.bar(x='feature', y='importance', legend=None, title='Feature Importance')
        

if __name__ == "__main__":
    main()
