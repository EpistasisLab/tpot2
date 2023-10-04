# import random
# from scipy.stats import loguniform, logser #TODO: remove this dependency?
import numpy as np #TODO: remove this dependency and use scipy instead?




#Replicating the API found in optuna: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
#copy-pasted some code
def suggest_categorical(name, choices, rng_):
    rng = np.random.default_rng(rng_)
    return rng.choice(choices)

def suggest_float(
    name: str,
    low: float,
    high: float,
    rng_,
    *,
    step = None,
    log = False,
    ):

    rng = np.random.default_rng(rng_)

    if log and step is not None:
        raise ValueError("The parameter `step` is not supported when `log` is true.")

    if low > high:
        raise ValueError(
            "The `low` value must be smaller than or equal to the `high` value "
            "(low={}, high={}).".format(low, high)
        )

    if log and low <= 0.0:
        raise ValueError(
            "The `low` value must be larger than 0 for a log distribution "
            "(low={}, high={}).".format(low, high)
        )

    if step is not None and step <= 0:
        raise ValueError(
            "The `step` value must be non-zero positive value, " "but step={}.".format(step)
        )

    #TODO check this produces correct output
    if log:
        value = rng.uniform(np.log(low),np.log(high))
        return np.e**value

    else:
        if step is not None:
            return rng.choice(np.arange(low,high,step))
        else:
            return rng.uniform(low,high)

def suggest_discrete_uniform(name, low, high, q, rng_):
    rng = np.random.default_rng(rng_)
    return suggest_float(name, low, high, step=q, rng_=rng)


def suggest_int(name, low, high, rng_, step=1, log=False):
    rng = np.random.default_rng(rng_)

    if low == high: #TODO check that this matches optuna's behaviour
        return low

    if log and step >1:
        raise ValueError("The parameter `step`>1 is not supported when `log` is true.")

    if low > high:
        raise ValueError(
            "The `low` value must be smaller than or equal to the `high` value "
            "(low={}, high={}).".format(low, high)
        )

    if log and low <= 0.0:
        raise ValueError(
            "The `low` value must be larger than 0 for a log distribution "
            "(low={}, high={}).".format(low, high)
        )

    if step is not None and step <= 0:
        raise ValueError(
            "The `step` value must be non-zero positive value, " "but step={}.".format(step)
        )

    if log:
        value = rng.uniform(np.log(low),np.log(high))
        return int(np.e**value)
    else:
        return rng.choice(list(range(low,high,step)))

def suggest_uniform(name, low, high, rng_):
    rng = np.random.default_rng(rng_)
    return suggest_float(name, low, high, rng_=rng)