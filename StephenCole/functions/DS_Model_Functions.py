from numpy import mean, std
import pandas as pd
import itertools
import seaborn as sns
import numpy as np
import time
import math
import tqdm
import os
import shap
import pickle
import yaml


from datetime import timedelta, datetime
from scipy import interp
from itertools import cycle
from tqdm import tqdm

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold ,StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import auc, roc_auc_score, roc_curve, make_scorer, f1_score, recall_score, precision_score, fbeta_score 
from sklearn.metrics import classification_report,average_precision_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.base import clone

from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize,scale

import xgboost as xgb
import lightgbm as lgb
import logging


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
    




def create_gridsearch(base_params, params):
    """
    Create gridsearch to be used in xgb or lgb hyperparameter tuning function
    
     Parameters
    -----
    base_params : Dictionary
        A dictonary of all the base parameters
    params : List of Dictionaries
        A grid which contains all the new parameters needed for testing
    
    Returns
    -----
    gridsearch_params : List of Dictionaries
        A grid with all the new parameters specified as well as the base parameters that were left alone
    """
    
    gridsearch_params = []
    
    for i in params:
        initial_params = base_params.copy()

        for key, value in i.items():
            initial_params[key] = value
            
        gridsearch_params.append(initial_params)
        
    return gridsearch_params
    
    
    
    
def overwrite_base_params(base_params, new_params):
    """
    This function overwrites the base parameters with the optimal (new) params
    
     Parameters
    -----
    base_params : Dictionary
        A dictonary of all the base parameters
    new_params : Dictionary
        A dictonary of the optimal parameters
    
    Returns
    -----
    best_params : List of Dictionaries
        A grid with all the new parameters specified as well as the base parameters that were left alone
    """
    
    best_params = base_params.copy()
    
    new_params_list = list(new_params.keys())
    
    for param in new_params_list:
        best_params[param] = new_params[param]
        
    return best_params
  
    
    
    

def param_tuning_xgboost(LOGGER, params_grid, dtrain, num_boost_round, nfold, stratified, metrics, early_stopping_rounds, seed):
    """
    Function completes hyperparameter tuning
    
    Parameters
    -----
    params_grid : Dictionary
        A grid which contains all the parameters needed for testing
    dtrain : XGBoost DMatrix
        A data structure the XGBoost developers created for memory efficiency and training speed
        with their machine learning library
    num_boost_round : Integer
        Number of boosting iterations
    nfold : Integer
        Number of folds for cross validation
    stratified : Boolean
        Perform stratified sampling
    metrics : Dictionary
        Evaluation metrics to be watched in CV
    early_stopping_rounds : Integer
        Early_stopping_rounds round(s) to continue training (needs to be tested separately after tuning the model)
    seed : Integer
        Seed used to generate the folds
    
    Returns
    -----
    max_roc_auc : Float64
        Max ROC AUC found for the model
    best_params : Dictionary
        The optimal dictionary of all specified parameters
    """
    
    start_time = time.time()
    print('start_time', time.asctime( time.localtime(time.time()) ))
    LOGGER.info("XGBoost Parameter Tuning started at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    best_params = {}
    max_roc_auc = 0
    
    for param in tqdm(params_grid):
        
        cv_results = xgb.cv(
            params=param,
            dtrain = dtrain,
            num_boost_round = num_boost_round,
            nfold = nfold,
            stratified = stratified,
            metrics = metrics,
            early_stopping_rounds = early_stopping_rounds,
            seed = seed
        )
        
        mean_roc_auc = cv_results['test-auc-mean'].max()
        
        if mean_roc_auc > max_roc_auc:
            max_roc_auc = mean_roc_auc
            best_params = param
        LOGGER.info("BATCH MAX ROC: {:.6f}".format(max_roc_auc))
        LOGGER.info("XGBoost Parameter Tuning Batch Finished at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    
    return max_roc_auc, best_params 







def param_tuning_lgb(LOGGER, params_grid, X_train, y_train, nfold, stratified, metrics):
    """
    Function completes hyperparameter tuning with LightGBM Cross Validation
    
    Parameters
    -----
    params_grid : Dictionary
        A grid which contains all the parameters needed for testing
    train_set : LightGBM DataFrame
        A data structure the LightGBM developers created for memory efficiency and training speed
        with their machine learning library
    num_boost_round : Integer
        Number of boosting iterations
    nfold : Integer
        Number of folds for cross validation
    stratified : Boolean
        Perform stratified sampling
    metrics : Dictionary
        Evaluation metrics to be watched in CV
    
    Returns
    -----
    max_roc_auc : Float64
        Max ROC AUC found for the model
    best_params : Dictionary
        The optimal dictionary of all specified parameters
    """
    
    start_time = time.time()
    print('start_time', time.asctime( time.localtime(time.time()) ))
    LOGGER.info("LightGBM Parameter Tuning started at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    best_params = {}
    max_roc_auc = 0
    
    for param in tqdm(params_grid):
        
        train_set = lgb.Dataset(X_train, label=y_train, params={'max_bins':param['max_bins']})
    
        light_results = lgb.cv(
                params=param,
                train_set = train_set,
                num_boost_round=1000,
                nfold = nfold,
                stratified = stratified,
                metrics = metrics
            )
        
        mean_roc_auc = max(light_results['auc-mean'])
        
        if mean_roc_auc > max_roc_auc:
            max_roc_auc = mean_roc_auc
            best_params = param
            
        LOGGER.info("BATCH MAX ROC: {:.6f}".format(max_roc_auc))
        LOGGER.info("LightGBM Parameter Tuning Batch Finished at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    
    return max_roc_auc, best_params






def xgb_hpt_tune(LOGGER, base_params, grid, dtrain, num_boost_round, nfold, stratified, metrics, early_stopping_rounds, seed):
    """
    Function extracts the best parameters from the specified grid based on mean AUC value of the CV splits for XGBoost
    
    Parameters
    -----
    base_params : Dictionary
        A dictonary of all the base parameters
    grid : List of Dictionaries
        A grid which contains all the new parameters needed for testing
    dtrain : XGBoost DMatrix
        A data structure the XGBoost developers created for memory efficiency and training speed
        with their machine learning library
    num_boost_round : Integer
        Number of boosting iterations
    nfold : Integer
        Number of folds for cross validation
    stratified : Boolean
        Perform stratified sampling
    metrics : Dictionary
        Evaluation metrics to be watched in CV
    early_stopping_rounds : Integer
        Early_stopping_rounds round(s) to continue training (needs to be tested separately after tuning the model)
    seed : Integer
        Seed used to generate the folds
    
    Returns
    -----
    best_params : Dictionary
        The optimal dictionary of parameters for XGBoost model
    """
    
    
    
    start_time = time.time()
    print('XGBoost HPT start_time', time.asctime( time.localtime(time.time()) ))
    LOGGER.info("xgb_hpt_tune started at {}.".format( time.asctime( time.localtime(time.time()) ) ))

    
    for params in grid:
        print('Hyperparameter space grid has {} points.\n'.format(len(params)))
        
        # Create gridsearch to be used in param_tuning function
        gridsearch_params = create_gridsearch(base_params=base_params, params=params)
        
        # Check performance of each grid
        max_roc_auc, new_params = param_tuning_xgboost(LOGGER=LOGGER,
                                                       params_grid=gridsearch_params, 
                                                       dtrain=dtrain, 
                                                       num_boost_round=num_boost_round, 
                                                       nfold=nfold, 
                                                       stratified=nfold,
                                                       metrics=metrics, 
                                                       early_stopping_rounds=early_stopping_rounds, 
                                                       seed=seed)
        
        # Overwrite base parameters with the optimal params
        best_params = overwrite_base_params(base_params=base_params, new_params=new_params)
        
        param_changes = [key for key in base_params if base_params[key] != best_params[key]]
        
        print("Parameters Changed ----------------------------------------------------------- ")
        for key in param_changes:
            print(key, ':', base_params[key], '->', best_params[key])
        print()
        # Re initialise for next loop
        base_params = best_params
        
    print()
    print("New set of parameters:", "\n")
    print()
    print(base_params, "\n")
    LOGGER.info("New set of parameters {}.".format(base_params))
    
    end_time = time.time()
    total_time = end_time - start_time
    print('XGBoost HPT total time', str(timedelta(seconds=total_time)))
    
    print()
    print()
    
    print('XGBoost HPT end_time', time.asctime( time.localtime(time.time()) ))
    
    LOGGER.info("xgb_hpt_tune finished at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    LOGGER.info("xgb_hpt_tune total time: {}.".format( str(timedelta(seconds=total_time)) ))
        
    return base_params


def lgb_hpt_tune(LOGGER, base_params, grid, X_train, y_train, nfold, stratified, metrics):
    """
    Function extracts the best parameters from the specified grid based on mean AUC value of the CV splits for 
    LightGBM Cross Validation
    
    Parameters
    -----
    base_params : Dictionary
        A dictonary of all the base parameters
    grid : List of Dictionaries
        A grid which contains all the new parameters needed for testing
    train_set : LightGBM DataFrame
        A data structure the LightGBM developers created for memory efficiency and training speed
        with their machine learning library
    num_boost_round : Integer
        Number of boosting iterations
    nfold : Integer
        Number of folds for cross validation
    stratified : Boolean
        Perform stratified sampling
    metrics : Dictionary
        Evaluation metrics to be watched in CV
    
    Returns
    -----
    best_params : Dictionary
        The optimal dictionary of parameters for LightGBM model
    """
    
    start_time = time.time()
    print('LightGBM HPT start_time', time.asctime( time.localtime(time.time()) ))
    LOGGER.info("lgb_hpt_tune started at {} seconds.".format( time.asctime( time.localtime(time.time()) ) ))

    
    for params in grid:
        print('Hyperparameter space grid has {} points.\n'.format(len(params)))
        
        # Create gridsearch to be used in param_tuning function
        gridsearch_params = create_gridsearch(base_params=base_params, params=params)
        
        # Check performance of each grid
        max_roc_auc, new_params = param_tuning_lgb(LOGGER=LOGGER,
                                                   params_grid=gridsearch_params, 
                                                   X_train=X_train,
                                                   y_train=y_train,
                                                   nfold=5,
                                                   stratified=True, 
                                                   metrics='auc')

        # Overwrite base parameters with the optimal params
        best_params = overwrite_base_params(base_params=base_params, new_params=new_params)
        
        param_changes = [key for key in base_params if base_params[key] != best_params[key]]
        
        print("Parameters Changed ----------------------------------------------------------- ")
        for key in param_changes:
            print(key, ':', base_params[key], '->', best_params[key])
        print()
        
        # Re initialise for next loop
        base_params = best_params
        
    print("New Parameters:", "\n")
    print(base_params, "\n")
    LOGGER.info("New set of parameters {}.".format(base_params))
    
    end_time = time.time()
    total_time = end_time - start_time
    print('LightGBM HPT total time', str(timedelta(seconds=total_time)))
    
    print()
    print()
    
    print('LightGBM HPT end_time', time.asctime( time.localtime(time.time()) ))
    
    LOGGER.info("lgb_hpt_tune finished at {}.".format( time.asctime( time.localtime(time.time()) ) ))
    LOGGER.info("lgb_hpt_tune total time: {}.".format( str(timedelta(seconds=total_time)) ))
        
    return base_params
