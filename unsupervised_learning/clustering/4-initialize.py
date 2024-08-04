#!/usr/bin/env python3
"""a function that initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Returns: pi, m, S, or None, None, None on failure"""
    try:
        n, d = X.shape
        pi = np.ones(k) / k
        C, clss = kmeans(X, k)
        m = C
        S = np.array([np.eye(d)] * k)
        return pi, m, S
    except Exception:
        return None, None, None
