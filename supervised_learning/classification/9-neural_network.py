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
        self.__W1, self.__W2 = np.random.randn(
            nodes, nx), np.random.randn(
            1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__b2, self.__A2 = 0, 0

    @property
    def W1(self):
        """getter function of W1"""
        return self.__W1

    @property
    def W2(self):
        """getter function of W2"""
        return self.__W2

    @property
    def b1(self):
        """getter function of b1"""
        return self.__b1

    @property
    def b2(self):
        """getter function of b2"""
        return self.__b2

    @property
    def A1(self):
        """getter function of A1"""
        return self.__A1

    @property
    def A2(self):
        """getter function of A2"""
        return self.__A2
