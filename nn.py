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
                 alpha=3.0,
                 lamda=0.5,
                 epochs=30,
                 layers=None,
                 batch_size=10,
                 weight_init="gaussian"):

        layers = layers or [25, 10]  # Default to one hidden layer with 25 units and one 10 unit output layer

        if len(layers) < 2:
            raise TypeError('The layers arg should be a list containing at least two integers')

        self.curr_cost = 0
        self.biases = [np.random.randn(y, 1) for y in layers]
        self.weights = [neural_funcs.Weight().get(weight_init)(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self._params = {
            "activation": neural_funcs.Activation().get(activation_),
            "cost": neural_funcs.Cost().get(cost),
            "alpha": alpha,
            "lamda": lamda,
            "epochs": epochs,
            "num_layers": len(layers) + 1,
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
        output = self._feedforward(x)
        return np.argmax(output[-1])

    def evaluate(self, test=None):
        test_results = [(self.predict(row[1:]), row[0]) for row in test]
        print('Cost: {0}'.format(self.curr_cost))
        return sum(int(x == y) for (x, y) in test_results) / float(len(test_results))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, key_value):
        key, value = key_value
        self._params[key] = value
        return self._params

    def _feedforward(self, x):
        a = x
        for b, w in zip(self.biases, self.weights):
            a = self.params["activation"](np.dot(w, a)+b)
        return a

    def _backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = np.array([x]).T
        activations = [activation]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.params["activation"](z)
            activations.append(activation)

        # backward pass
        delta = self.params["cost"](y, activations[-1], derivative=True) * \
            self.params["activation"](zs[-1], derivative=True)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.params["num_layers"]):
            z = zs[-l]
            sp = self.params["activation"](z, derivative=True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def _label(self, y):
        label = np.zeros(self.params["output_layer"])
        label[y] = 1
        return label

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

            print('Percent correct: {0} at epoch {1}\n'.format(self.evaluate(data), i + 1))

    def _update_model(self, X, y):
        # nabla = [np.zeros(w.shape) for w in self.weights]
        # m = len(X)
        # for i in xrange(m):
        #     delta_nabla = self._backprop(X[i], y[i])
        #     nabla = [n + dn for n, dn in zip(nabla, delta_nabla)]
        #
        # Update model weights
        # self.weights = [w - (self.params["alpha"] / m) * n for w, n in zip(self.weights, nabla)]

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m = len(X)
        for i in xrange(m):
            delta_nabla_b, delta_nabla_w = self._backprop(X[i], y[i])
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(self.params["alpha"]/m)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.params["alpha"]/m)*nb
                       for b, nb in zip(self.biases, nabla_b)]