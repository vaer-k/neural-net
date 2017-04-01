import pdb
import random
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

        self.weights = [neural_funcs.Weight.get(weight_init)(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self._params = {
            "activation": neural_funcs.Activation.get(activation_),
            "cost": neural_funcs.Cost.get(cost),
            "alpha": alpha,
            "lamda": lamda,
            "epochs": epochs,
            "num_layers": len(layers),
            "hidden_layers": layers[:-1],
            "output_layer": layers[-1],
            "batch_size": batch_size,
            "weight_init": neural_funcs.get(weight_init)
        }

    def fit(self, X=None, y=None):
        if not X:
            print("Fitting with default (MNIST) training data and labels...")
            X = pd.read_csv(TRAIN)
            y = np.array([X["label"].as_matrix()])
            X = X.drop("label", axis=1).as_matrix()

        self.weights = [self.params["weight_init"](len(X), self.params["hidden_layers"][0])] + self.weights
        self._sgd(X, y)

    def predict(self, X=None):
        X = X or pd.read_csv(TEST)
        _, output = self._feedforward(X)

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
        for i in xrange(self.params["epochs"]):
            data = np.concatenate((y.T, X), axis=1)
            data = np.random.permutation(data)
            n = len(data)
            batches = [data[m:m + self.params["batch_size"]]
                       for m in xrange(0, n, self.params["batch_size"])]

            for batch in batches:
                y = data[:, 0]
                X = data[:, 1:]
                self._update_model(X, y)

    def _update_model(self, X, y):
        nabla = [np.zeros(w.shape) for w in self.weights]
        for x in X:
            delta_nabla = self.backprop(x, y)
            nabla = [nw+dnw for nw, dnw in zip(nabla, delta_nabla)]

        self.weights = [w-(self.params["alpha"]/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla)]


########################################################################################################################

def nnCostFunction(self, nn_params, X, y, lambda_):
    """
    Computes the cost and gradient of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    """

    ## Initialize

    theta1_size = self.hidden_layer_size * (self.input_layer_size + 1)
    Theta1 = np.reshape(nn_params[0:theta1_size],
                           (self.hidden_layer_size, self.input_layer_size + 1))
    Theta2 = np.reshape(nn_params[theta1_size + 1:],
                           (self.num_labels, self.hidden_layer_size + 1))

    m = np.shape(X)[0]

    J = 0
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

    ## Feed forward from initial theta

    # Add bias terms
    X = np.concatenate((np.ones([m, 1]), X), axis=1)

    # Compute activation of every node in hidden layer for every example
    a2 = self.activation(np.dot(X, Theta1.T))






    grad = np.concatenate(np.ravel(Theta1_grad), np.ravel(Theta2_grad))
    return grad