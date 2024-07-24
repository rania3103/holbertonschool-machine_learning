#!/usr/bin/env python3
"""a function that calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """Returns a numpy.ndarray of shape (d, d)
    containing the correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if C.shape[0] != C.shape[1] or C.ndim != 2:
        raise ValueError('C must be a 2D square matrix')
    std_dev = np.sqrt(np.diag(C))
    d = C.shape[0]
    corr_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr_matrix[i, j] = C[i, j] / (std_dev[i] * std_dev[j])
    return corr_matrix
