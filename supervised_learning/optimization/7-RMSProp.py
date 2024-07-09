#!/usr/bin/env python3
"""Write a function  that updates a variable
using the RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Returns: the updated variable and the new moment, respectively"""
    s *= beta2 + (1 - beta2) * np.square(grad)
    var -= alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
