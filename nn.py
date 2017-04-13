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
                 alpha=0.05,
                 lamda=0.005,
                 epochs=30,
                 layers=None,
                 batch_size=10,
                 weight_init="epsilon"):

        layers = layers or [25, 10]  # Default to one hidden layer with 25 units and one 10 unit output layer
        # layers = layers or [256, 64, 10]

        if len(layers) < 2:
            raise TypeError('The layers arg should be a list containing at least two integers')

        self.curr_cost = 0
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
        _, output = self._feedforward(x)
        return np.argmax(output[-1])

    def evaluate(self, test=None):
        test_results = [(self.predict(row[1:]), row[0]) for row in test]
        return round(sum(int(x == y) for (x, y) in test_results) / float(len(test_results)), 3)

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
        activ_deriv = self.params["activation"](weighted_sums[-1], derivative=True)
        delta = self.params["cost"](self._label(y), activations[-1], derivative=True, activation_deriv=activ_deriv)

        delta = np.array([delta]).T
        a = np.array([np.insert(activations[-2], 0, 1)]).T
        nabla[-1] = np.dot(delta, a.T)

        delta = np.insert(delta, 0, 1)
        delta = np.array([delta]).T

        # Backpropagate error
        for l in xrange(2, self.params["num_layers"]):
            delta = np.dot(self.weights[-l + 1].T, delta[1:]) \
                    * np.array([np.insert(self.params["activation"](weighted_sums[-l], derivative=True), 0, 1)]).T

            a = np.array([np.insert(activations[-l - 1], 0, 1)]).T
            nabla[-l] = np.dot(delta[1:], a.T)

        return nabla

    def _label(self, y):
        label = np.zeros(self.params["output_layer"])
        label[y] = 1
        return label

    def _sgd(self, X, y):
        data = np.concatenate((y.T, X), axis=1)
        training_length = int(len(data) * .9)
        train = data[:training_length]
        test = data[training_length:]

        n = len(train)
        for i in xrange(self.params["epochs"]):
            train = np.random.permutation(train)
            batches = [train[m:m + self.params["batch_size"]]
                       for m in xrange(0, n, self.params["batch_size"])]

            print('Updating model with alpha {0}'.format(round(self.params["alpha"], 3)))
            for batch in batches:
                y = batch[:, 0]
                X = batch[:, 1:]
                self._update_model(X, y)

            if self.params["alpha"] < 0.01:
                self.params["alpha"] *= 0.95
            else:
                self.params["alpha"] *= .90

            self.curr_cost = self._compute_cost(train)

            if not i % 5:
                print('Current cost: {0}\n'.format(round(self.curr_cost, 3)))

            print('Epoch #{0} results:'.format(i + 1))
            print('\tTrain set accuracy: {0}%'.format(self.evaluate(train) * 100))
            print('\tTest set accuracy: {0}%\n'.format(self.evaluate(test) * 100))

    def _update_model(self, X, y):
        nabla = [np.zeros(w.shape) for w in self.weights]
        m = len(X)
        for i in xrange(m):
            delta_nabla = self._backprop(X[i], y[i])
            nabla = [n + dn for n, dn in zip(nabla, delta_nabla)]

        # Update model weights
        alpha = self.params["alpha"]
        lamda = self.params["lamda"]
        self.weights = [w - (alpha / m) * n for w, n in zip(self.weights, nabla)]

        for theta in self.weights:
            theta[:, 1:] = [w - ((alpha * lamda) / m) * w for w in theta[:, 1:]]

    def _compute_cost(self, data):
        cost = 0
        m = len(data)
        for row in data:
            _, activations = self._feedforward(row[1:])
            cost += self.params["cost"](self._label(row[0]), activations[-1]) / m

        cost += np.sum([(self.params["lamda"] / (2 * m)) * np.sum(np.square(theta[:, 1:])) for theta in self.weights])
        return cost
