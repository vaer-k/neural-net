import pdb
import neural_funcs
import numpy as np
import pandas as pd

TEST = "./raw_data/test.csv"
TRAIN = "./raw_data/train.csv"


class DigitClassifier:
    """
    :param layers: A list containing the number of units in each layer.
    The last integer element is considered the output layer
    :param activation_: The activation function to be used. Can be
    "logistic" or "tanh"
    """
    def __init__(self,
                 activation_="logistic",
                 cost="mse",
                 alpha=0.5,
                 lamda=0.5,
                 epochs=10,
                 layers=None,
                 batch_size=100,
                 weight_init="epsilon"):

        layers = layers or [5, 10]  # Default to one hidden layer with 5 units and one 10 unit output layer

        if len(layers) < 2:
            raise TypeError('The layers arg should be a list containing at least two integers')

        self.weights = [neural_funcs.Weight().get(weight_init)(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self._params = {
            "activation": neural_funcs.Activation().get(activation_),
            "cost": neural_funcs.Cost().get(cost),
            "alpha": alpha,
            "lamda": lamda,
            "epochs": epochs,
            "num_layers": len(layers),
            "hidden_layers": layers[:-1],
            "output_layer": layers[-1],
            "batch_size": batch_size,
            "weight_init": neural_funcs.Weight().get(weight_init)
        }

    def fit(self, X=None, y=None):
        if not X:
            print("Fitting with default (MNIST) training data and labels...")
            X = pd.read_csv(TRAIN)
            y = np.array([X["label"].as_matrix()])
            X = X.drop("label", axis=1).as_matrix()

        self.weights = [self.params["weight_init"](X.shape[1], self.params["hidden_layers"][0])] + self.weights
        self._sgd(X, y)

    def predict(self, x):
        _, output = self._feedforward(x)
        return np.argmax(output[-1])

    def evaluate(self, test=None):
        test_results = [(self.predict(row[1:]), row[0]) for row in test]
        return sum(int(x == y) for (x, y) in test_results) / len(test_results)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, key_value):
        key, value = key_value
        self._params[key] = value
        return self._params

    def _feedforward(self, x):
        weighted_sums = []
        activations = [x]
        a = x  # init the first activation layer to the input X
        for theta in self.weights:
            a = np.insert(a, 0, 1)  # add bias term
            z = np.dot(theta, a)
            weighted_sums.append(z)
            a = self.params["activation"](z)
            activations.append(a)

        return weighted_sums, activations

    def _backprop(self, x, y):
        weighted_sums, activations = self._feedforward(x)
        nabla = [np.zeros(w.shape) for w in self.weights]

        # Use the output layer activations and weights to initialize error delta
        delta = self.params["cost"](y, activations[-1], derivative=True) * \
                self.params["activation"](weighted_sums[-1], derivative=True)

        # Backpropagate error
        for l in xrange(2, self.params["num_layers"]):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * \
                    self.params["activation"](weighted_sums[-l], derivative=True)
            nabla[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla

    def _sgd(self, X, y):
        data = np.concatenate((y.T, X), axis=1)
        n = len(data)
        for i in xrange(self.params["epochs"]):
            data = np.random.permutation(data)
            batches = [data[m:m + self.params["batch_size"]]
                       for m in xrange(0, n, self.params["batch_size"])]

            for batch in batches:
                y = batch[:, 0]
                X = batch[:, 1:]
                self._update_model(X, y)

            print(self.evaluate(data))

    def _update_model(self, X, y):
        nabla = [np.zeros(w.shape) for w in self.weights]
        for i in xrange(len(X)):
            delta_nabla = self._backprop(X[i], y[i])
            nabla = [n + dn for n, dn in zip(nabla, delta_nabla)]

        # Update model weights
        self.weights = [w - (self.params["alpha"] / len(X)) * n
                        for w, n in zip(self.weights, nabla)]
