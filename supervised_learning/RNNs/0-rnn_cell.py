#!/usr/bin/env python3
"""Create a class RNNCell that represents a cell of a simple RNN"""
import numpy as np


class RNNCell:
    """RNNCell class"""

    def __init__(self, i, h, o):
        """constructor"""
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        x = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y = e_x / e_x.sum(axis=1, keepdims=True)
        return h_next, y
