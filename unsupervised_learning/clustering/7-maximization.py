#!/usr/bin/env python3
"""a function that calculates the maximization step
in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Returns: pi, m, S, or None, None, None on failure"""
    try:
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
