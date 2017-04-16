import pdb
import numpy as np
from collections import defaultdict


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
        return np.tanh(x) if not derivative else 1.0 - np.square(np.tanh(x))

    @staticmethod
    def logistic(x, derivative=False):
        sigma = 1/(1 + np.exp(-x))

        if derivative:
            return sigma * (1 - sigma)

        return sigma


class Cost(Base):
    def __init__(self):
        self.options = {
            "cross_entropy": self.cross_entropy,
            "mse": self.mse,
        }

    @staticmethod
    def cross_entropy(y, yhat, derivative=False, **kwargs):
        if derivative:
            return yhat - y

        return np.mean(np.nan_to_num(-y * np.log(yhat) - (1 - y) * np.log(1 - yhat)))

    @staticmethod
    def mse(y, yhat, derivative=False, activation_deriv=None):
        if derivative:
            return (yhat - y) * activation_deriv

        return np.mean((yhat - y) ** 2) / 2


class Evaluation(Base):
    def __init__(self):
        self.options = {
            "accuracy": self.accuracy,
            "f1": self.f1
        }

    @staticmethod
    def accuracy(test_result):
        return round(sum(int(x == y) for x, y in test_result) / float(len(test_result)), 3)

    @staticmethod
    def f1(test_result):
        metrics = {class_: defaultdict(float) for class_ in xrange(0, 10)}
        for class_, counts in metrics.iteritems():
            for result in test_result:
                if result[0] != class_ and result[1] != class_:
                    continue

                counts["tru_pos"] += int(result[0] == result[1])
                counts["pred_pos"] += int(result[0] == class_)
                counts["cond_pos"] += int(result[1] == class_)

            try:
                precision = counts["tru_pos"] / counts["pred_pos"]
                recall = counts["tru_pos"] / counts["cond_pos"]
                counts["f1"] = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                counts["f1"] = 0.0

        return round(np.mean([counts["f1"] for _, counts in metrics.iteritems()]), 3)


class Weight(Base):
    def __init__(self):
        self.options = {
            "epsilon": self.epsilon,
            "gaussian": self.gaussian,
        }

    @staticmethod
    def normalize(weights):
        min = np.min(weights)
        max = np.max(weights)
        normalize_x = lambda x: (x - min) / (max - min)
        for i, row in enumerate(weights):
            for j, col in enumerate(row):
                weights[i][j] = normalize_x(col)

        return weights

    @staticmethod
    def gaussian(L_in, L_out, normalize=False):
        # return self.normalize(np.random.randn(L_out, 1 + L_in)) if normalize else np.random.randn(L_out, 1 + L_in)
        return np.random.randn(L_out, 1 + L_in) / np.sqrt(L_in)

    @staticmethod
    def epsilon(L_in, L_out):
        """
        Randomly initialize the weights of a layer with L_in incoming
        connections and L_out outgoing connections. Note the addition
        of space (the +1s) for the "bias" terms.
        """
        epsilon = np.sqrt(6) / np.sqrt(L_in + L_out)
        return (np.random.random((L_out, 1 + L_in)) * 2 * epsilon) - epsilon
