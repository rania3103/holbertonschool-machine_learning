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

    def forward_prop(self, X):
        """Calculates the forward propagation
        of the neural network"""
        sum_w_hidden_layer = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-sum_w_hidden_layer))
        sum_w_output_layer = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-sum_w_output_layer))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        numb_examples = len(Y[0])
        return - np.sum(Y * np.log(A) + (1 - Y) *
                        np.log(1.0000001 - A)) / numb_examples

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent
        on the neural network"""
        self.__W2 -= alpha * np.matmul((A2 - Y), (A1.T)) / len(Y[0])
        self.__b2 -= alpha * np.sum(A2 - Y).reshape(1, 1) / len(Y[0])
        mul = np.matmul(self.__W2.T, (A2 - Y)) * A1 * (1 - A1)
        self.__W1 -= alpha * np.matmul(mul, X.T) / len(Y[0])
        self.__b1 -= alpha * np.sum(mul).reshape(1, 1) / len(Y[0])
