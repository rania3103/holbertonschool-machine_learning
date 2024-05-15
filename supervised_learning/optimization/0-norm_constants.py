#!/usr/bin/env python3
"""Write a function that normalizes (standardizes) a matrix:"""
import numpy as np


def normalization_constants(X):
    """Returns: The normalized X matrix"""
    return np.mean(X, axis=0), np.std(X, axis=0)
