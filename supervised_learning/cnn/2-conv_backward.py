#!/usr/bin/env python3
"""a function that performs back propagation
over a convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_new) // 2
        pw = ((w_prev - 1) * sw + kw - w_new) // 2
    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1
    pad_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev = np.zeros(pad_A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    db[0, 0, 0, c] += dZ[i, h, w, c]
                    dW[:, :, :, c] += (pad_A_prev[i,
                                                  h * sh:h * sh + kh,
                                                  w * sw:w * sw + kw,
                                                  :] * dZ[i, h, w, c])
                    dA_prev[i, h * sh:h * sh + kh, w * sw:w * sw + kw, :] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
    if padding == 'same':
        dA_prev = dA_prev[:, ph:h_prev + ph, pw:w_prev + pw, :]
    return dA_prev, dW, db
