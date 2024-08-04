#!/usr/bin/env python3
"""a function that calculates the maximization step
in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Returns: pi, m, S, or None, None, None on failure"""
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        if not isinstance(g, np.ndarray) or g.ndim != 2 or g.shape[1] != n:
            return None, None, None
        n, d = X.shape
        k, n = g.shape
        pi = np.sum(g, axis=1) / n
        m = np.dot(g, X) / np.sum(g, axis=1)[:, None]
        S = np.zeros((k, d, d))
        for i in range(k):
            diff_X_m = X - m[i]
            S[i] = np.dot(g[i] * diff_X_m.T, diff_X_m) / np.sum(g[i])
        return pi, m, S

    except Exception:
        return None, None, None
