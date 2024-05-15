#!/usr/bin/env python3
import numpy as np
"""Write a function that normalizes (standardizes) a matrix:"""


def normalization_constants(X):
    """Returns: The normalized X matrix"""
    return np.mean(X, axis=0), np.mean(X, axis=0)
