#!/usr/bin/env python3
"""a function that performs forward propagation
for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Returns: H, Y"""
    t, m, i = X.shape
    h = h_0.shape[1]
    forward = np.zeros((t, m, h))
    backward = np.zeros((t, m, h))
    h_prev_f = h_0
    for i in range(t):
        h_prev_f = bi_cell.forward(h_prev_f, X[i])
        forward[i] = h_prev_f

    h_prev_b = h_t
    for i in reversed(range(t)):
        h_prev_b = bi_cell.backward(h_prev_b, X[i])
        backward[i] = h_prev_b
    H = np.concatenate((forward, backward), axis=2)
    Y = bi_cell.output(H)
    return H, Y
