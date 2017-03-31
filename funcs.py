import numpy as np


class Activation:
    def __init__(self):
        self.options = {
            "tanh": self.tanh,
            "logistic": self.logistic,
        }

    @classmethod
    def get(cls, type_):
        return cls.options[type_]

    @staticmethod
    def tanh(x, derivative=False):
        return np.tanh(x) if not derivative else 1.0 - np.tanh(x) ** 2

    @staticmethod
    def logistic(x, derivative=False):
        sigma = 1/(1 + np.exp(-x))
        return sigma if not derivative else sigma * (1 - sigma(x))


class Cost:
    def __init__(self):
        self.options = {
            "cross_entropy": self.cross_entropy,
            "mse": self.mse,
        }

    @classmethod
    def get(cls, type_):
        return cls.options[type_]

    @staticmethod
    def cross_entropy(y, yhat):
        return -1 * np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    @staticmethod
    def mse(y, yhat):
        return np.mean((yhat - y) ** 2) / 2
