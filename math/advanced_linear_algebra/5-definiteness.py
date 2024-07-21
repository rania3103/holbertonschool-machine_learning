#!/usr/bin/env python3
"""a function that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Return: the string Positive definite,
    Positive semi-definite, Negative semi-definite,
    Negative definite,
    or Indefinite if the matrix is positive definite,
    positive semi-definite, negative semi-definite,
    negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1] or \
            not np.allclose(matrix.T, matrix):
        return None
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return 'Positive definite'
    elif np.all(eigenvalues >= 0):
        return 'Positive semi-definite'
    elif np.all(eigenvalues < 0):
        return 'Negative definite'
    elif np.all(eigenvalues <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
