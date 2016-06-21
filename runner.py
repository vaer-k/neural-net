# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:48:46 2016

@author: vrupp
"""

import nn


input_layer_size = 784 # 28x28 Input Images of Digits
hidden_layer_size = 25 # 25 hidden units
num_labels = 10        # 10 labels, from 1 to 10

net = nn.NeuralNetwork()

initial_Theta1 = net.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = net.randInitializeWeights(hidden_layer_size, num_labels)

#initial_nn_params = 