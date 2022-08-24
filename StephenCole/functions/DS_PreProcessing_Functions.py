# PreProcessing and Feature Selection Functions

import os
import time
import pandas as pd
import numpy as np
import pickle

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb


def save_obj(obj, name, file_path):
    """
    This function saves any object as a .pkl file so that it can be easily read in other notebooks
    
    Parameters
    -----
    obj : Object's variable name 
        The object that needs to be saved.
    name : String
        The name that you would like to save the object as.
    file_path: String
        Directory in which you are saving the object to.
    
    Returns
    -----
        A saved .pkl file in dir and name specified.
    """
    file_path = os.path.join(file_path, name)
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, file_path):
    """
    This function loads any .pkl file
    
    Parameters
    -----
    name : String
        The name of the file that needs to be loaded.
    file_path: String
        Directory in which you are loading the object from.
    
    Returns
    -----
        An object that was saved as a .pkl file in dir and name specified.
    """
    file_path = os.path.join(file_path, name)
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)


def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. 
    Note: Use after PreProcessing step   
    
    Parameters
    -----
    df : pandas.DataFrame
        DataFrame that needs memory optimisation
    
    Returns
    -----
    df : pandas.DataFrame  
      Memory efficient DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == object:
            df[col] = df[col].astype('category')
        else:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def find_column_type(df):
    """
    This function identify categorical, boolean and numerical values.
    
    Parameters
    ---------
    df : DataFrame
        Usually a DataFrame with training samples that will be used to fit a model.
    
    Returns
    -------
    categorical_cols : list
        Categorical features.
    bool_cols:
        Boolean features.
    numerical_cols:
        Numerical features.
    """
    
    all_cols=list(df.columns)
    numerical_cols_temp = df._get_numeric_data().columns
    categorical_cols = list(set(all_cols) - set(numerical_cols_temp))
    bool_cols = [col for col in all_cols if np.isin(df[col].dropna().unique(), [0, 1,0.0,1.0]).all()]
    numerical_cols = list(set(numerical_cols_temp) - set(bool_cols))
    
    return categorical_cols,bool_cols,numerical_cols


def vif_cal(selected_features, df):
    """
    Function calculates VIF values for selected_features
    
    Parameters
    -----
    selected_features : list
        Selected feature names for calculating VIF factors
    df: pandas DataFrame
        Dataframe containing features
        
    Returns
    -----
    vif : DataFrame
        VIF values for each feature
    """
    
    start_time = time.time()
    print('start_time', time.asctime( time.localtime(time.time()) ))

    
    data = [[variance_inflation_factor(df.iloc[:,:].values, i) for i in range(df.shape[1])], selected_features]
    vif = pd.DataFrame(data=data, index=['VIF Factor','Feature']).transpose()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('total_time to calculate VIF Factor', str(timedelta(seconds=total_time)))
    
    return vif

def recalc_vif(selected_features, df, vif):
    """
    Function Identifies the highest VIF Factor from the vif table and removes the feature from the list and DataFrame and
    recalculates the VIF Factors
    
    Parameters
    -----
    selected_features : list
        Selected feature names for calculating VIF factors
    df: pandas DataFrame
        Dataframe containing features
    vif : pandas DataFrame
        DataFrame containing the VIF Factor and Features
        
    Returns
    -----
    vif : pandas DataFrame
        VIF values for each feature
    df : pandas DataFrame
        DataFrame containing the modelling dataset
    selected_features : list
        list of features needed for RFECV
    """
    
    feature_to_drop = vif.loc[vif['VIF Factor']== vif['VIF Factor'].max(), 'Feature'].values[0]
    feature_max = vif['VIF Factor'].max()
    df.drop(feature_to_drop, axis=1, inplace=True)
    selected_features.remove(feature_to_drop)
    vif = vif_cal(selected_features, df)
    print("Feature that was dropped: {} ({:.3f})".format(feature_to_drop, feature_max))
    
    return selected_features, df, vif

def rfe_cv(X, y, step, n_splits, params):
    """
    Completes recursive feature elimination with k-fold cross validation. Step features are removed at each round 
    
    Parameters
    -----
    X : pandas DataFrame
        DataFrame containing model input features
    y : pandas Series
        Series containing target feature
    step : int
        Number of features removed at each round 
    n_splits : int
        Number of folds for cross validation
    params : dict
        Training model parameters
        
    Returns : dict
        Cross validation results for each step of features
    -----
    """
    count = 0 
    
    feature_names = list(X.columns)
    n_features = len(X.columns)
    
    results = {}
    
    start_time = time.time()

    while n_features >= 1:
        
        loop_time = time.time()
        
        kf = StratifiedKFold(n_splits=n_splits)
        X = X[feature_names]

        xgb_importance = []
        roc_auc_average = 0
        precision_recall_auc_average = 0
        log_loss_average = 0
        precision_average = 0

        for train_index, test_index in kf.split(X, y):

            X_train_i = X.iloc[train_index]
            y_train_i = y.iloc[train_index]
            dtrain = xgb.DMatrix(X_train_i, y_train_i)
            del X_train_i
            del y_train_i

            X_test_i = X.iloc[test_index]
            y_test_i = y.iloc[test_index]
            dtest = xgb.DMatrix(X_test_i, y_test_i)
            del X_test_i
            del y_test_i

            evallist = [(dtrain, 'train'), (dtest, 'test')]
            del dtrain
            del dtest
            
            xgb_model = xgb.train(params=params, dtrain=evallist[0][0], num_boost_round=100, evals=evallist, early_stopping_rounds=20, maximize=False, verbose_eval=False)
            
            predictions = xgb_model.predict(evallist[1][0])

            # Feature importances
            feature_importances_dict = xgb_model.get_score(importance_type='gain')

            total_importance = sum(feature_importances_dict.values())
            xgb_importance_dict_norm = {k:v/total_importance for k, v in feature_importances_dict.items()}
            xgb_importance.append(xgb_importance_dict_norm)

            # ROC AUC Metric
            roc_auc_average += roc_auc_score(evallist[1][0].get_label(), predictions)

            # Precision & Recall AUC Metric
            precision, recall, thresholds = precision_recall_curve(evallist[1][0].get_label(), predictions)
            precision_recall_auc_average += auc(recall, precision)

            # Log Loss Metric
            log_loss_average += float(xgb_model.attributes()['best_score'])

            # Precision Metric
            precision_average += average_precision_score(evallist[1][0].get_label(), predictions)
            
        xgb_importance_avg = pd.DataFrame(xgb_importance).fillna(0).mean(axis=0).sort_values(ascending=False)
        feature_names = list(xgb_importance_avg.index)

        results[n_features] = {'feature_importance': xgb_importance_avg,
                               'roc_auc_average': roc_auc_average/n_splits,
                               'precision_recall_auc_average': precision_recall_auc_average/n_splits,
                               'log_loss_average': log_loss_average/n_splits,
                               'precision_average': precision_average/n_splits
                              }
        
        print('----- Features: ', n_features, ' -----')
        print('roc_auc_average: ', roc_auc_average/n_splits)
        print('precision_recall_auc_average: ', precision_recall_auc_average/n_splits)
        print('log_loss_average:', log_loss_average/n_splits)
        print('precision_average:', precision_average/n_splits)
        print('step runtime:', time.time() - loop_time)
        print('\n')

        n_features = n_features - step
        feature_names = feature_names[0:n_features] 

    print('Runtime: ', time.time() - start_time)
    
    return results


def model_performance(X, y, params):
    """
    Prints model performance using stratified k-folds
    
    Parameters
    -----
    X : pandas DataFrame
        Input features
    params : dictionary
        Model parameters
    """
    start_time = time.time()

    kf = StratifiedKFold(n_splits=3)

    xgb_importance = []
    roc_auc_average = 0
    precision_recall_auc_average = 0
    log_loss_average = 0
    precision_average = 0

    for train_index, test_index in kf.split(X, y):

        X_train_i = X.iloc[train_index]
        y_train_i = y.iloc[train_index]
        dtrain = xgb.DMatrix(X_train_i, y_train_i)
        del X_train_i
        del y_train_i

        X_test_i = X.iloc[test_index]
        y_test_i = y.iloc[test_index]
        dtest = xgb.DMatrix(X_test_i, y_test_i)
        del X_test_i
        del y_test_i

        evallist = [(dtrain, 'train'), (dtest, 'test')]
        del dtrain
        del dtest

        xgb_model = xgb.train(params=params, dtrain=evallist[0][0], num_boost_round=100, evals=evallist, early_stopping_rounds=20, maximize=False, verbose_eval=False)

        predictions = xgb_model.predict(evallist[1][0])

        # Feature importances
        feature_importances_dict = xgb_model.get_score(importance_type='gain')

        total_importance = sum(feature_importances_dict.values())
        xgb_importance_dict_norm = {k:v/total_importance for k, v in feature_importances_dict.items()}
        xgb_importance.append(xgb_importance_dict_norm)

        # ROC AUC Metric
        roc_auc_average += roc_auc_score(evallist[1][0].get_label(), predictions)

        # Precision & Recall AUC Metric
        precision, recall, thresholds = precision_recall_curve(evallist[1][0].get_label(), predictions)
        precision_recall_auc_average += auc(recall, precision)

        # Log Loss Metric
        log_loss_average += float(xgb_model.attributes()['best_score'])

        # Precision Metric
        precision_average += average_precision_score(evallist[1][0].get_label(), predictions)

    xgb_importance_avg = pd.DataFrame(xgb_importance).fillna(0).mean(axis=0).sort_values(ascending=False)
    feature_names = list(xgb_importance_avg.index)


    print('----- Features: ', len(X.columns), ' -----')
    print('roc_auc_average: ', roc_auc_average/3)
    print('precision_recall_auc_average: ', precision_recall_auc_average/3)
    print('log_loss_average:', log_loss_average/3)
    print('precision_average:', precision_average/3)
    print('\n')

    print('Runtime: ', time.time() - start_time)

