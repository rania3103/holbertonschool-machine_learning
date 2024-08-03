#!/usr/bin/env python3
"""a function that calculates the probability density
function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Returns: P(is a numpy.ndarray of shape (n,)
    containing the PDF values for each data point),
    or None on failure"""
    n, d = X.shape
    if not isinstance(
            X,
            np.ndarray) or not isinstance(
            m,
            np.ndarray) or not isinstance(
                S,
            np.ndarray) or m.shape[0] != d or S.shape[0] != d\
            or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    norm_fact = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    diff_X_m = X - m
    inv = np.linalg.inv(S)
    exp_fact = np.exp(-0.5 * np.sum(np.dot(diff_X_m, inv) * diff_X_m, axis=1))
    P = norm_fact * exp_fact
    return np.maximum(P, 1e-300)
