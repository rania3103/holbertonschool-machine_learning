#!/usr/bin/env python3
"""a function that calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """var (is the total variance), or None on failure"""
    try:
        n, d = X.shape
        k, d = C.shape
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clusters = np.argmin(dist, axis=1)
        return np.sum((X - C[clusters]) ** 2)
    except Exception:
        return None
