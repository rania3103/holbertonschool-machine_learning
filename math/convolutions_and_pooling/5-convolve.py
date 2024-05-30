#!/usr/bin/env python3
"""a function that performs a convolution
on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (((h - 1) * sh + kh - h) // 2) + 1
        pw = (((w - 1) * sw + kw - w) // 2) + 1
    elif isinstance(padding, tuple):
        ph, pw = padding

    pad_img = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, nc))
    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    pad_img[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                    * kernels[:, :, :, k], axis=(1, 2, 3))
    return output
