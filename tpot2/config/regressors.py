from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import RidgeCV


from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV

import numpy as np

from xgboost import XGBRegressor
from functools import partial




#TODO: fill in remaining
#TODO check for places were we could use log scaling

def params_RandomForestRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'n_estimators': 100,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.0, rng_=rng),
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False], rng_=rng),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21, rng_=rng),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21, rng_=rng),
        'random_state': rng_seed_
    }


# SGDRegressor parameters
def params_SGDRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'loss': trial.suggest_categorical(f'loss_{name}', ['huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive'], rng_=rng),
        'penalty': 'elasticnet',
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-5, 0.01, log=True, rng_=rng),
        'learning_rate': trial.suggest_categorical(f'learning_rate_{name}', ['invscaling', 'constant'], rng_=rng),
        'fit_intercept':True,
        'l1_ratio': trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0, rng_=rng),
        'eta0': trial.suggest_float(f'eta0_{name}', 0.01, 1.0, rng_=rng),
        'power_t': trial.suggest_float(f'power_t_{name}', 1e-5, 100.0, log=True, rng_=rng),
        'random_state': rng_seed_

    }
    return params

# Ridge parameters
def params_Ridge(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0, rng_=rng),
        'fit_intercept': True,


        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'solver': trial.suggest_categorical(f'solver_{name}', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], rng_=rng),
        'random_state': rng_seed_
    }
    return params


# Lasso parameters
def params_Lasso(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0, rng_=rng),
        'fit_intercept': True,
        # 'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
        'precompute': trial.suggest_categorical(f'precompute_{name}', [True, False, 'auto'], rng_=rng),

        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),

        'positive': trial.suggest_categorical(f'positive_{name}', [True, False], rng_=rng),
        'selection': trial.suggest_categorical(f'selection_{name}', ['cyclic', 'random'], rng_=rng),
        'random_state': rng_seed_
    }
    return params

# ElasticNet parameters
def params_ElasticNet(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'alpha': 1 - trial.suggest_float(f'alpha_{name}', 0.0, 1.0, log=True, rng_=rng),
        'l1_ratio': 1- trial.suggest_float(f'l1_ratio_{name}',0.0, 1.0, rng_=rng),
        'selection': trial.suggest_categorical(f'selection_{name}', ['cyclic', 'random'], rng_=rng), # if random, random_state will keep it deterministic
        'random_state': rng_seed_
        }
    return params

# Lars parameters
def params_Lars(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'fit_intercept': True,
        'verbose': trial.suggest_categorical(f'verbose_{name}', [True, False], rng_=rng),
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),

        # 'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto_{name}', True, False], rng_=rng),
        'n_nonzero_coefs': trial.suggest_int(f'n_nonzero_coefs_{name}', 1, 100, rng_=rng),
        'eps': trial.suggest_float(f'eps_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False], rng_=rng),
        'fit_path': trial.suggest_categorical(f'fit_path_{name}', [True, False], rng_=rng),
        # 'positive': trial.suggest_categorical(f'positive_{name}', [True, False], rng_=rng),
        'random_state': rng_seed_
    }
    return params

# OrthogonalMatchingPursuit parameters
def params_OrthogonalMatchingPursuit(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'n_nonzero_coefs': trial.suggest_int(f'n_nonzero_coefs_{name}', 1, 100, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
        'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto', True, False], rng_=rng),
    }
    return params

# BayesianRidge parameters
def params_BayesianRidge(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'n_iter': trial.suggest_int(f'n_iter_{name}', 100, 1000, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'alpha_1': trial.suggest_float(f'alpha_1_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'alpha_2': trial.suggest_float(f'alpha_2_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'lambda_1': trial.suggest_float(f'lambda_1_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'lambda_2': trial.suggest_float(f'lambda_2_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'compute_score': trial.suggest_categorical(f'compute_score_{name}', [True, False], rng_=rng),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False], rng_=rng),
    }
    return params

# LassoLars parameters
def params_LassoLars(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0, rng_=rng),
        # 'fit_intercept': True,
        # 'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
        # 'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto_{name}', True, False], rng_=rng),
        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000, rng_=rng),
        'eps': trial.suggest_float(f'eps_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'jitter': trial.suggest_float(f'jitter_{name}', 0, 1, log=False, rng_=rng),
        # 'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False], rng_=rng),
        # 'positive': trial.suggest_categorical(f'positive_{name}', [True, False], rng_=rng),
        'random_state': rng_seed_
    }
    return params

# LassoLars parameters
def params_LassoLarsCV(trial, rng_seed_, rng_, name=None): # TODO: cv splitter must be passed into cv for deterministic results
    rng = np.random.default_rng(rng_)

    params = {
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
    }
    return params

# BaggingRegressor parameters
def params_BaggingRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'n_estimators': trial.suggest_int(f'n_estimators_{name}', 10, 100, rng_=rng),
        'max_samples': trial.suggest_float(f'max_samples_{name}', 0.05, 1.00, rng_=rng),
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.00, rng_=rng),
        'bootstrap': trial.suggest_categorical(f'bootstrap_{name}', [True, False], rng_=rng),
        'bootstrap_features': trial.suggest_categorical(f'bootstrap_features_{name}', [True, False], rng_=rng),
        'random_state': rng_seed_
    }
    return params

# ARDRegression parameters
def params_ARDRegression(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'n_iter': trial.suggest_int(f'n_iter_{name}', 100, 1000, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'alpha_1': trial.suggest_float(f'alpha_1_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'alpha_2': trial.suggest_float(f'alpha_2_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'lambda_1': trial.suggest_float(f'lambda_1_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'lambda_2': trial.suggest_float(f'lambda_2_{name}', 1e-6, 1e-1, log=True, rng_=rng),
        'compute_score': trial.suggest_categorical(f'compute_score_{name}', [True, False], rng_=rng),
        'threshold_lambda': trial.suggest_int(f'threshold_lambda_{name}', 100, 1000, rng_=rng),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False], rng_=rng),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False], rng_=rng),
    }
    return params



# TheilSenRegressor parameters
def params_TheilSenRegressor(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'n_subsamples': trial.suggest_int(f'n_subsamples_{name}', 10, 100, rng_=rng),
        'max_subpopulation': trial.suggest_int(f'max_subpopulation_{name}', 100, 1000, rng_=rng),
        'fit_intercept': True,
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False], rng_=rng),
        'verbose': trial.suggest_categorical(f'verbose_{name}', [True, False], rng_=rng),
        'random_state': rng_seed_
    }
    return params


# SVR parameters
def params_SVR(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid'], rng_=rng),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True, rng_=rng),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4, rng_=rng),
        'max_iter': 3000,
        'tol': trial.suggest_float(f'tol_{name}', 0, 1, log=False, rng_=rng)
    }
    return params

# Perceptron parameters
def params_Perceptron(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'penalty': trial.suggest_categorical(f'penalty_{name}', [None, 'l2', 'l1', 'elasticnet'], rng_=rng),
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'l1_ratio': trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0, rng_=rng),
        'fit_intercept': True,
        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000, rng_=rng),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'shuffle': trial.suggest_categorical(f'shuffle_{name}', [True, False], rng_=rng),
        'verbose': trial.suggest_categorical(f'verbose_{name}', [0, 1, 2, 3, 4, 5], rng_=rng),
        'eta0': trial.suggest_float(f'eta0_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        'learning_rate': trial.suggest_categorical(f'learning_rate_{name}', ['constant', 'optimal', 'invscaling'], rng_=rng),
        'early_stopping': trial.suggest_categorical(f'early_stopping_{name}', [True, False], rng_=rng),
        'validation_fraction': trial.suggest_float(f'validation_fraction_{name}', 0.05, 1.00, rng_=rng),
        'n_iter_no_change': trial.suggest_int(f'n_iter_no_change_{name}', 1, 100, rng_=rng),
        'class_weight': trial.suggest_categorical(f'class_weight_{name}', [None, 'balanced'], rng_=rng),
        'warm_start': trial.suggest_categorical(f'warm_start_{name}', [True, False], rng_=rng),
        'average': trial.suggest_categorical(f'average_{name}', [True, False], rng_=rng),
        'random_state': rng_seed_
    }
    return params

def params_MLPRegressor(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-4, 1e-1, log=True, rng_=rng),
        'learning_rate_init': trial.suggest_float(f'learning_rate_init_{name}', 1e-3, 1., log=True, rng_=rng),
        'random_state': rng_seed_
    }

    return params


#GradientBoostingRegressor parameters
def params_GradientBoostingRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    loss = trial.suggest_categorical(f'loss_{name}', ['ls', 'lad', 'huber', 'quantile'])

    params = {

        'n_estimators': 100,
        'loss': loss,
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-4, 1, log=True, rng_=rng),
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11, rng_=rng),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21, rng_=rng),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21, rng_=rng),
        'subsample': 1-trial.suggest_float(f'subsample_{name}', 0.05, 1.00, log=True, rng_=rng),
        'max_features': 1-trial.suggest_float(f'max_features_{name}', 0.05, 1.00, log=True, rng_=rng),
        'random_state': rng_seed_

    }

    if loss == 'quantile' or loss == 'huber':
        alpha = trial.suggest_float(f'alpha_{name}', 0.05, 0.95)
        params['alpha'] = alpha

    return params



def params_DecisionTreeRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1,11, rng_=rng),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21, rng_=rng),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21, rng_=rng),
        # 'criterion': trial.suggest_categorical(f'criterion_{name}', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], rng_=rng),
        # 'splitter': trial.suggest_categorical(f'splitter_{name}', ['best', 'random'], rng_=rng),
        #'max_features': trial.suggest_categorical(f'max_features_{name}', [None, 'auto', 'sqrt', 'log2'], rng_=rng),
        #'ccp_alpha': trial.suggest_float(f'ccp_alpha_{name}',  1e-1, 10.0, rng_=rng),
        'random_state': rng_seed_

    }
    return params

def params_KNeighborsRegressor(trial, rng_seed_, rng_, name=None, n_samples=100):
    rng = np.random.default_rng(rng_)

    params = {
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, 100, rng_=rng),
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance'], rng_=rng),
        'p': trial.suggest_int(f'p_{name}', 1, 3, rng_=rng),
        'metric': trial.suggest_categorical(f'metric_{name}', ['minkowski', 'euclidean', 'manhattan'], rng_=rng),

        }
    return params

def params_LinearSVR(trial, rng_seed_, rng_, name=None): # TODO: not in use?
    rng = np.random.default_rng(rng_)

    params = {
        'epsilon': trial.suggest_float(f'epsilon_{name}', 1e-4, 1.0, log=True, rng_=rng),
        'C': trial.suggest_float(f'C_{name}', 1e-4,25.0, log=True, rng_=rng),
        'dual': trial.suggest_categorical(f'dual_{name}', [True,False], rng_=rng),
        'loss': trial.suggest_categorical(f'loss_{name}', ['epsilon_insensitive', 'squared_epsilon_insensitive'], rng_=rng),
        'random_state': rng_seed_

    }
    return params


# XGBRegressor parameters
def params_XGBRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True, rng_=rng),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.05, 1.0, rng_=rng),
        'min_child_weight': trial.suggest_int(f'min_child_weight_{name}', 1, 21, rng_=rng),
        #'booster': trial.suggest_categorical(name='booster_{name}', choices=['gbtree', 'dart'], rng_=rng),
        'n_estimators': 100,
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11, rng_=rng),
        'nthread': 1,
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'random_state': rng_seed_
    }


def params_AdaBoostRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'n_estimators': 100,
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1.0, log=True, rng_=rng),
        'loss': trial.suggest_categorical(f'loss_{name}', ['linear', 'square', 'exponential'], rng_=rng),
        'random_state': rng_seed_

    }
    return params

# ExtraTreesRegressor parameters
def params_ExtraTreesRegressor(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    params = {
        'n_estimators': 100,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.0, rng_=rng),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21, rng_=rng),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21, rng_=rng),
        'bootstrap': trial.suggest_categorical(f'bootstrap_{name}', [True, False], rng_=rng),

        #'criterion': trial.suggest_categorical(f'criterion_{name}', ['squared_error', 'poisson', 'absolute_error', 'friedman_mse'], rng_=rng),

        #'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 10, rng_=rng),

        #'min_weight_fraction_leaf': trial.suggest_float(f'min_weight_fraction_leaf_{name}', 0.0, 0.5, rng_=rng),
        # 'max_features': trial.suggest_categorical(f'max_features_{name}', [None, 'auto', 'sqrt', 'log2'], rng_=rng),
        #'max_leaf_nodes': trial.suggest_int(f'max_leaf_nodes_{name}', 2, 100, rng_=rng),
        #'min_impurity_decrease': trial.suggest_float(f'min_impurity_decrease_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        # 'min_impurity_split': trial.suggest_float(f'min_impurity_split_{name}', 1e-5, 1e-1, log=True, rng_=rng),

        #if bootstrap is True
        #'oob_score': trial.suggest_categorical(f'oob_score_{name}', [True, False], rng_=rng),

        #'ccp_alpha': trial.suggest_float(f'ccp_alpha_{name}', 1e-5, 1e-1, log=True, rng_=rng),
        # 'max_samples': trial.suggest_float(f'max_samples_{name}', 0.05, 1.00, rng_=rng),
        'random_state': rng_seed_
    }
    return params



def make_regressor_config_dictionary(n_samples=10):
    n_samples = min(n_samples,100) #TODO optimize this


    regressor_config_dictionary = {
        ElasticNet: params_ElasticNet, # added values to make deterministic
        # ElasticNetCV: { # TODO: the cv value must be a give cv splitter to keep it deterministic
        #                 'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
        #                 'cv': 5,
        #                 },
        ExtraTreesRegressor: params_ExtraTreesRegressor,
        GradientBoostingRegressor: params_GradientBoostingRegressor,
        AdaBoostRegressor: params_AdaBoostRegressor,
        DecisionTreeRegressor: params_DecisionTreeRegressor,
        KNeighborsRegressor: partial(params_KNeighborsRegressor,n_samples=n_samples),
        # LassoLarsCV: params_LassoLarsCV, # cv will make this undeterministic until cv splitter actually is passed
        LassoLars: params_LassoLars,
        SVR: params_SVR,
        RandomForestRegressor: params_RandomForestRegressor,
        # RidgeCV: {}, # cv will make this undeterministic until cv splitter actually is passed
        XGBRegressor: params_XGBRegressor,
        SGDRegressor: params_SGDRegressor,

    }

    return regressor_config_dictionary
