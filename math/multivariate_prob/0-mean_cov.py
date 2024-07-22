#!/usr/bin/env python3
"""a function that calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """Returns: mean, cov"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0)
    cov = np.dot((X - mean).T, X - mean) / (n - 1)
    return mean, cov
