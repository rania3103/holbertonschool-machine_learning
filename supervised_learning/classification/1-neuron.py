#!/usr/bin/env python3
"""Write a class Neuron that defines a single
neuron performing binary classification"""
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron,
        W: The weights vector for the neuron,
        b: The bias for the neuron,
        A: The activated output of the neuron"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b, self.__A = 0, 0

    @property
    def W(self):
        """returns The weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """returns The bias for the neuron."""
        return self.__b

    @property
    def A(self):
        """returns The activated output of the neuron."""
        return self.__A
