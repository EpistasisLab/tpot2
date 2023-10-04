from functools import partial
from tpot2.builtin_modules import ZeroCount, OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

import numpy as np


def params_sklearn_preprocessing_Binarizer(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'threshold': trial.suggest_float(f'threshold_{name}', 0.0, 1.0, log=False, rng_=rng)
    }

def params_sklearn_decomposition_FastICA(trial, rng_seed_, rng_, name=None, n_features=100):
    rng = np.random.default_rng(rng_)

    return {
        'algorithm': trial.suggest_categorical(f'algorithm_{name}', ['parallel', 'deflation'], rng_=rng),
        'whiten':'unit-variance',
        'random_state':rng_seed_
    }

def params_sklearn_cluster_FeatureAgglomeration(trial, rng_seed_, rng_, name=None, n_features=100):
    rng = np.random.default_rng(rng_)

    linkage = trial.suggest_categorical(f'linkage_{name}', ['ward', 'complete', 'average'], rng_=rng)
    if linkage == 'ward':
        metric = 'euclidean'
    else:
        metric = trial.suggest_categorical(f'metric_{name}', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], rng_=rng)
    return {
        'linkage': linkage,
        'metric': metric,
        'n_clusters': trial.suggest_int(f'n_clusters_{name}', 2, 4, rng_=rng), #TODO perhaps a percentage of n_features
    }

def params_sklearn_preprocessing_Normalizer(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'norm': trial.suggest_categorical(f'norm_{name}', ['l1', 'l2', 'max'], rng_=rng),
    }

def params_sklearn_kernel_approximation_Nystroem(trial, rng_seed_, rng_, name=None, n_features=100):
    rng = np.random.default_rng(rng_)

    return {
        'gamma': trial.suggest_float(f'gamma_{name}', 0.0, 1.0, log=False, rng_=rng),
        'kernel': trial.suggest_categorical(f'kernel_{name}', ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'], rng_=rng),
        'n_components': trial.suggest_int(f'n_components_{name}', 1, 20, rng_=rng), #TODO perhaps a percentage of n_features
        'random_state': rng_seed_
    }

def params_sklearn_decomposition_PCA(trial, rng_seed_, rng_, name=None, n_features=100):
    rng = np.random.default_rng(rng_)

    # keep the number of components required to explain 'variance_explained' of the variance
    # variance_explained = 1 - trial.suggest_float(f'n_components_{name}', 0.001, 0.5, rng_=rng, log=True) #values closer to 1 are more likely
    variance_explained = trial.suggest_float(f'n_components_{name}', 1, 5, rng_=rng, log=False)


    return {
        'n_components': variance_explained,
        'random_state':rng_seed_
    }

def params_sklearn_kernel_approximation_RBFSampler(trial, rng_seed_, rng_, name=None, n_features=100):
    rng = np.random.default_rng(rng_)

    return {
        'gamma': trial.suggest_float(f'gamma_{name}', 0.0, 1.0, log=False, rng_=rng),
        'random_state':rng_seed_
    }

def params_sklearn_kernel_preprocessing_PolynomialFeatures(trial, rng_seed_, rng_, name=None): # added to let evolution figure it out
    rng = np.random.default_rng(rng_)

    degree = trial.suggest_int(f'gamma_{name}', 1, 5, log=False, rng_=rng)

    return {
        'degree': (0,degree),
        'interaction_only': trial.suggest_categorical(f'kernel_{name}', [True,False], rng_=rng),
        'include_bias': trial.suggest_categorical(f'kernel_{name}', [True,False], rng_=rng)
    }


def params_tpot_builtins_ZeroCount(trial, rng_seed_, rng_, name=None):

    return {}


def params_tpot_builtins_OneHotEncoder(trial, rng_seed_, rng_, name=None):

    return {}





def make_transformer_config_dictionary(n_features=10):
    #n_features = min(n_features,100) #TODO optimize this
    return {
                Binarizer: params_sklearn_preprocessing_Binarizer,
                FastICA: partial(params_sklearn_decomposition_FastICA,n_features=n_features),
                FeatureAgglomeration: partial(params_sklearn_cluster_FeatureAgglomeration,n_features=n_features),
                MaxAbsScaler: {},
                MinMaxScaler: {},
                Normalizer: params_sklearn_preprocessing_Normalizer,
                Nystroem: partial(params_sklearn_kernel_approximation_Nystroem,n_features=n_features),
                PCA: partial(params_sklearn_decomposition_PCA,n_features=n_features),
                PolynomialFeatures: params_sklearn_kernel_preprocessing_PolynomialFeatures,
                RBFSampler: partial(params_sklearn_kernel_approximation_RBFSampler,n_features=n_features),
                RobustScaler: {},
                StandardScaler: {},
                ZeroCount: params_tpot_builtins_ZeroCount,
                OneHotEncoder: params_tpot_builtins_OneHotEncoder,
            }