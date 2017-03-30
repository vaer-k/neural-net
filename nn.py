import activation
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, test='./raw_data/test.csv', train='./raw_data/train.csv', num_labels=10, activation_='logistic'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation_: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        train = pd.read_csv(train)
        self.labels = train['label']
        self.train = train.drop('label', axis=1)

        self.test = pd.read_csv(test)

        self.input_layer_size = self.train.shape[1] - 1
        self.hidden_layer_size = 25
        self.num_labels = num_labels
        
        self.activation = activation.get(activation_)

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