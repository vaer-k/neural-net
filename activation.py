import numpy as np


def tanh(x, derivative=False):
    return np.tanh(x) if not derivative else 1.0 - np.tanh(x) ** 2


def logistic(x, derivative=False):
    sigma = 1/(1 + np.exp(-x))
    return sigma if not derivative else sigma * (1 - sigma(x))


def get(type_):
    activations = {
        'tanh': tanh,
        'logistic': logistic,
    }

    return activations[type_]
