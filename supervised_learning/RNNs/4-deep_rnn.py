#!/usr/bin/env python3
"""a function that performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Returns: H, Y"""
    t, m, i = X.shape
    n_layers = len(rnn_cells)
    l, m, h = h_0.shape
    H = np.zeros((t + 1, n_layers, m, h))
    H[0] = h_0
    Y = []
    for t_step in range(t):
        x_step = X[t_step]
        for lay in range(n_layers):
            if lay == 0:
                h_prev, y = rnn_cells[lay].forward(H[t_step, lay], x_step)
            else:
                h_prev, y = rnn_cells[lay].forward(H[t_step, lay], h_prev)
            H[t_step + 1, lay] = h_prev
        Y.append(y)
    return H, np.array(Y)
