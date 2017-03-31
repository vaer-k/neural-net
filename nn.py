import pdb
import funcs
import numpy as np
import pandas as pd

TEST = "./raw_data/test.csv"
TRAIN = "./raw_data/train.csv"


class DigitClassifier:
    def __init__(self, test=TEST, train=TRAIN, activation_="logistic", alpha=0.5, layers=None):
        """
        :param layers: A list containing the number of units in each layer.
        The last layer is considered the output layer
        :param activation_: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        layers = layers or [5, 10]  # Default to one hidden layer with 5 units and one 10 unit output layer

        if len(layers) < 2:
            raise TypeError('The layers arg should be a tuple containing at least two integers')

        self.test = pd.read_csv(test)
        self.train = pd.read_csv(train)
        self.activation = funcs.Activation.get(activation_)
        self.alpha = alpha
        self.hidden_layers = layers[:-1]
        self.output_layer = layers[-1]

    def init_weights(self, L_in, L_out):
        """
        Randomly initialize the weights of a layer with L_in incoming
        connections and L_out outgoing connections. Note the addition
        of space (the +1s) for the "bias" terms.
        """
        
        epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)
        return (np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init) - epsilon_init

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