# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:48:46 2016

@author: vrupp
"""

import nn

net = nn.NeuralNetwork()

initial_Theta1 = net.randInitializeWeights(net.input_layer_size, net.hidden_layer_size)
initial_Theta2 = net.randInitializeWeights(net.hidden_layer_size, net.num_labels)

initial_nn_params = [item for sublist in initial_Theta1 for item in sublist] + [item for sublist in initial_Theta2 for item in sublist]