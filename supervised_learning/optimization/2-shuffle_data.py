#!/usr/bin/env python3
import numpy as np
"""Write a function that shuffles the data points in two matrices the same way"""


def shuffle_data(X, Y):
    """Returns: the shuffled X and Y matrices"""
    return np.random.permutation(X), np.random.permutation(Y)
