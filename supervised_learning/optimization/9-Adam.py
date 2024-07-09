#!/usr/bin/env python3
"""Write a function  that updates a variable in place
using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Returns: the updated variable, the new first moment
    and the new second moment, respectively"""
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v = beta1 * v + (1 - beta1) * grad
    corr_v = v / (1 - beta1 ** t)
    corr_s = s / (1 - beta2 ** t)
    var = var - alpha * corr_v / (np.sqrt(corr_s) + epsilon)
    return var, v, s
