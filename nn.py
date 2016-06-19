import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

    def randInitializeWeights(L_in, L_out):
        """
        Randomly initialize the weights of a layer with L_in incoming
        connections and L_out outgoing connections. Note that W is set
        to a matrix of size(L_out, 1 + L_in) because the column row of
        W handles the "bias" terms.
        """
        epsilon_init = 0.12
        return (np.random.random(L_out, 1 + L_in) * 2 * epsilon_init) - epsilon_init

    def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, C):
        """
        Computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
        """

        # Initialize

        Theta1 = numpy.reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)],
                               (hidden_layer_size, (input_layer_size + 1)))
        Theta2 = numpy.reshape(nn_params[1 + (hidden_layer_size * (input_layer_size + 1)):],
                               (num_labels, (hidden_layer_size + 1)))

        m = numpy.shape(X)[0]

        J = 0
        Theta1_grad = np.zeros(np.shape(Theta1))
        Theta2_grad = np.zeros(np.shape(Theta2))

        # Feed forward from initial theta
        X = np.concatenate((np.ones([m, 1]), X), axis=1)



        return grad