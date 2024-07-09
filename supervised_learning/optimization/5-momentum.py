#!/usr/bin/env python3
"""Write a function  that updates a variable using
the gradient descent with momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Returns: the updated variable and the new moment, respectively"""
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
