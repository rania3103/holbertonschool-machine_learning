#!/usr/bin/env python3
"""Write a class NeuralNetwork that defines a neural
network with one hidden layer performing binary classification"""
import numpy as np


class NeuralNetwork:
    """defines a neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, nodes):
        """nx is the number of input features,
        W1: The weights vector for the hidden layern,
        b1: The bias for the hidden layer,
        A1: The activated output for the hidden layer,
        W2: The weights vector for the output neuron,
        b2: The bias for the output neuron,
        A2: The activated output of the output neuron
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.W1, self.W2 = np.random.randn(
            nodes, nx), np.random.randn(
            1, nodes)
        self.b1, self.A1 = np.zeros((nodes, 1)), 0
        self.b2, self.A2 = 0, 0
