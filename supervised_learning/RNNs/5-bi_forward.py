#!/usr/bin/env python3
"""Create a class BidirectionalCell that represents
a bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell:
    """BidirectionalCell class"""

    def __init__(self, i, h, o):
        """constructor"""
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in the forward
        direction for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next
