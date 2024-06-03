#!/usr/bin/env python3
"""a function that performs forward propagation
over a pooling layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Returns: the output of the pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h_prev - kh) // sh + 1
    out_w = (w_prev - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c_prev))

    for h in range(out_h):
        for w in range(out_w):
            if mode == 'max':
                output[:, h, w, :] = np.max(
                    A_prev[:, h * sh:h * sh + kh, w * sw:w * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                output[:, h, w, :] = np.mean(
                    A_prev[:, h * sh:h * sh + kh, w * sw:w * sw + kw, :],
                    axis=(1, 2))
    return output
