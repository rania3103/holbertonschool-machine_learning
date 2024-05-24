#!/usr/bin/env python3
"""write a function that updates the weights and biases
of a neural network using gradient descent
with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Returns: the updated weights and
    biases of a neural network"""
    m = Y.shape[1]
    for lay in range(1, L + 1):
        a = cache['A' + str(lay)] - Y
        prev_a = cache['A' + str(lay - 1)]
        d_w = np.dot(a, prev_a.T) / m
        w = weights['W' + str(lay)]
        reg_term = lambtha / (2 * m)
        d_w += reg_term * w
        d_b = np.sum(a, axis=1) / m
        weights['b' + str(lay)] -= alpha * d_b
        weights['W' + str(lay)] -= alpha * d_w
    return weights
