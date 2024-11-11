#!/usr/bin/env python3
"""a function that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """returns the computed policy"""
    z = np.dot(matrix, weight)
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=1, keepdims=True)
