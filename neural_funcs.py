import pdb
import numpy as np


class Base:
    def __init__(self):
        self.options = {}

    def get(cls, type_):
        return cls.options[type_]


class Activation(Base):
    def __init__(self):
        self.options = {
            "tanh": self.tanh,
            "logistic": self.logistic,
        }

    @staticmethod
    def tanh(x, derivative=False):
        return np.tanh(x) if not derivative else 1.0 - np.tanh(x) ** 2

    @staticmethod
    def logistic(x, derivative=False):
        sigma = 1/(1 + np.exp(-x))
        return sigma if not derivative else sigma * (1 - sigma)


class Cost(Base):
    def __init__(self):
        self.options = {
            "cross_entropy": self.cross_entropy,
            "mse": self.mse,
        }

    @staticmethod
    def cross_entropy(y, yhat, derivative=False):
        return -1 * np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    @staticmethod
    def mse(y, yhat, derivative=False):
        return np.mean((yhat - y) ** 2) / 2 if not derivative else (yhat - y)


class Weight(Base):
    def __init__(self):
        self.options = {
            "epsilon": self.epsilon,
            "gaussian": self.gaussian,
        }

    @staticmethod
    def gaussian(L_in, L_out):
        return np.random.randn(L_out, L_in)

    @staticmethod
    def epsilon(L_in, L_out):
        """
        Randomly initialize the weights of a layer with L_in incoming
        connections and L_out outgoing connections. Note the addition
        of space (the +1s) for the "bias" terms.
        """
        epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
        return (np.random.random((L_out, L_in)) * 2 * epsilon) - epsilon
