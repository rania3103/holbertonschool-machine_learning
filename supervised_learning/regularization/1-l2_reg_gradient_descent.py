#!/usr/bin/env python3
"""write a function that updates the weights and biases
of a neural network using gradient descent
with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Returns: the cost of the network
    accounting for L2 regularization"""
    m = Y.shape[1]
    for l in range(1, L + 1):
        a = cache['A' + str(l)] - Y
        prev_a = cache['A' + str(l - 1)]
        d_w = np.dot(a, prev_a.T) / m
        w = weights['W' + str(l)]
        reg_term = lambtha / (2 * m)
        d_w += reg_term * w
        d_b = np.sum(a, axis=1) / m
        weights['b' + str(l)] -= alpha * d_b
        weights['W' + str(l)] -= alpha * d_w
    return weights
