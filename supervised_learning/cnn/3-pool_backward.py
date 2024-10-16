#!/usr/bin/env python3
"""a function that performs back propagation
over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Returns: the partial derivatives with
    respect to the previous layer (dA_prev)"""
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_prev, w_prev, c = A_prev.shape[1:]
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw
                    if mode == 'max':
                        a_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, h_start:h_end, w_start:w_end,
                                c] += mask * dA[i, h, w, c]
                    if mode == "avg":
                        dA_prev[i, h_start:h_end, w_start:w_end,
                                c] += ((dA[i, h, w, c] / (kh * kw)) *
                                       np.ones((kh, kw)))
    return dA_prev
