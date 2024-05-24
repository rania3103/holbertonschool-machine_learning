#!/usr/bin/env python3
"""write a function that calculates the cost
of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Returns: the cost of the network
    accounting for L2 regularization"""
    sum_sq_w = 0
    for k in weights:
        if k[0] == 'W':
            sum_sq_w += np.sum((weights[k] ** 2))
    return cost + ((sum_sq_w * lambtha) / (2 * m))
