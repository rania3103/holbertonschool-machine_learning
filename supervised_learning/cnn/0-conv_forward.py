#!/usr/bin/env python3
"""a function that performs forward propagation
over a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Returns: the output of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    out_h = ((h_prev + 2 * ph - kh) // sh) + 1
    out_w = ((w_prev + 2 * pw - kw) // sw) + 1

    pad_inp = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out = np.zeros((m, out_h, out_w, c_new))

    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(c_new):
                    out[i, h, w, c] = np.sum(
                        pad_inp[i, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                        * W[:, :, :, c]) + b[:, :, :, c]
    return activation(out)
