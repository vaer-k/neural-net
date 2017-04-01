import pdb
import neural_funcs
import numpy as np
import pandas as pd

TEST = "./raw_data/test.csv"
TRAIN = "./raw_data/train.csv"


class DigitClassifier:
    def __init__(self, activation_="logistic", cost="mse", alpha=0.5, lamda=0.5, layers=None):
        """
        :param layers: A list containing the number of units in each layer.
        The last integer element is considered the output layer
        :param activation_: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        layers = layers or [5, 10]  # Default to one hidden layer with 5 units and one 10 unit output layer

        if len(layers) < 2:
            raise TypeError('The layers arg should be a list containing at least two integers')

        self.epoch = 0
        self._params = {
            "activation": neural_funcs.Activation.get(activation_),
            "cost": neural_funcs.Cost.get(cost),
            "alpha": alpha,
            "lamda": lamda,
            "hidden_layers": layers[:-1],
            "output_layer": layers[-1],
        }

    def fit(self, X=None, y=None):
        if not X:
            print("Fitting with default (MNIST) training data and labels...")
            X = pd.read_csv(TRAIN)
            y = X["label"]
            X = X.drop("label", axis=1)

        # TODO gradient descent

    def predict(self, X=None):
        X = self._add_bias(X or pd.read_csv(TEST))

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param, value):
        self.params[param] = value
        return self._params

    def _feedforward(self, X):
        weighted_sums = []
        activations = [X]
        a = X  # init the first activation layer to the input X
        for theta in self.weights:
            a = self._add_bias(a)  # add bias term
            z = np.dot(theta, a)
            weighted_sums.append(z)
            a = self.params["activation"](z)
            activations.append(a)

        return weighted_sums, activations

    def _backprop(self, X, y):
        weighted_sums, activations = self._feedforward(X)
        gradient = [np.zeros(w.shape) for w in self.weights]

        # Use the output layer activations and weights to compute output error delta
        delta_out = self.params["cost"](y, activations[-1], derivative=True) * \
                    self.params["activation"](weighted_sums[-1], derivative=True)

        # TODO
        # backpropagate error
        # Compute gradient

    @staticmethod
    def _add_bias(self, X):
        """Add a bias feature to each vector"""
        bias_terms = pd.DataFrame.from_items([('bias', [1] * X.shape[0])])
        X.insert(0, 'bias', bias_terms['bias'])
        return X


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