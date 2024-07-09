#!/usr/bin/env python3
"""Write a function  that normalizes an unactivated
output of a neural network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Returns: the normalized Z matrix"""
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    norm_Z = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * norm_Z + beta
