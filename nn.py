import numpy as np
import pandas as pd
import math

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, test='./input/test.csv', train='./input/train.csv', num_labels=10, activation='logistic'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        
        self.test = pd.read_csv(test)
        self.train = pd.read_csv(train)
        
        self.input_layer_size = self.train.shape[1] - 1
        self.hidden_layer_size = 25
        self.num_labels = num_labels
        
        
        if activation == 'logistic':
            self.activation = logistic
            self.activation_derivative = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative

    def randInitializeWeights(self, L_in, L_out):
        """
        Randomly initialize the weights of a layer with L_in incoming
        connections and L_out outgoing connections. Note the addition
        of space (the +1s) for the "bias" terms.
        """
        
        epsilon_init = math.sqrt(6) / math.sqrt(L_in + L_out)
        return (np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init) - epsilon_init

    def nnCostFunction(self, nn_params, X, y, lambda_):
        """
        Computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
        """

        # Initialize
        
        train_size = self.hidden_layer_size * (self.input_layer_size + 1)
        Theta1 = np.reshape(nn_params[0:train_size],
                               (self.hidden_layer_size, self.input_layer_size + 1))
        Theta2 = np.reshape(nn_params[train_size + 1:],
                               (self.num_labels, self.hidden_layer_size + 1))

        m = np.shape(X)[0]
#
        J = 0
        Theta1_grad = np.zeros(np.shape(Theta1))
        Theta2_grad = np.zeros(np.shape(Theta2))
#
#        # Feed forward from initial theta
#        X = np.concatenate((np.ones([m, 1]), X), axis=1)
#
#
#
#        return grad