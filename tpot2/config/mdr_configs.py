from mdr import MDR, ContinuousMDR
from skrebate import ReliefF, SURF, SURFstar, MultiSURF
from functools import partial
import numpy as np

#MDR
def params_MDR(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'tie_break': trial.suggest_categorical(name=f'tie_break_{name}', choices=[0,1], rng_=rng),
        'default_label': trial.suggest_categorical(name=f'default_label_{name}', choices=[0,1], rng_=rng),
    }

def params_ContinuousMDR(trial, rng_seed_, rng_, name=None):
    rng = np.random.default_rng(rng_)

    return {
        'tie_break': trial.suggest_categorical(name=f'tie_break_{name}', choices=[0,1], rng_=rng),
        'default_label': trial.suggest_categorical(name=f'default_label_{name}', choices=[0,1], rng_=rng),
    }


#skrebate
def params_skrebate_ReliefF(trial, rng_seed_, rng_, name=None, n_features=10):
    rng = np.random.default_rng(rng_)

    return {
        'n_features_to_select': trial.suggest_int(f'n_features_to_select_{name}', 1, n_features, rng_=rng, log=True),
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 2, 500, rng_=rng, log=True),
    }

def params_skrebate_SURF(trial, rng_seed_, rng_, name=None, n_features=10):
    rng = np.random.default_rng(rng_)

    return {
        'n_features_to_select': trial.suggest_int(f'n_features_to_select_{name}', 1, n_features, rng_=rng, log=True),
    }

def params_skrebate_SURFstar(trial, rng_seed_, rng_, name=None, n_features=10):
    rng = np.random.default_rng(rng_)

    return {
        'n_features_to_select': trial.suggest_int(f'n_features_to_select_{name}', 1, n_features, rng_=rng, log=True),
    }

def params_skrebate_MultiSURF(trial, rng_seed_, rng_, name=None, n_features=10):
    rng = np.random.default_rng(rng_)

    return {
        'n_features_to_select': trial.suggest_int(f'n_features_to_select_{name}', 1, n_features, rng_=rng, log=True),
    }



def make_skrebate_config_dictionary(n_features=10):
    return {
        ReliefF : partial(params_skrebate_ReliefF, n_features=n_features),
        SURF : partial(params_skrebate_SURF, n_features=n_features),
        SURFstar : partial(params_skrebate_SURFstar, n_features=n_features),
        MultiSURF: partial(params_skrebate_MultiSURF,n_features=n_features),
    }


def make_MDR_config_dictionary():
    return {
        MDR : params_MDR
    }

def make_ContinuousMDR_config_dictionary():
    return {
        ContinuousMDR : params_ContinuousMDR
    }
