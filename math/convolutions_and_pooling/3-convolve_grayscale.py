#!/usr/bin/env python3
"""a function that performs that performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        ph, pw = 0, 0
        out_h = h + 1 - kh
        out_w = w + 1 - kw
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
        out_h = ((h + 2 * ph - kh) // sh) + 1
        out_w = ((w + 2 * pw - kw) // sw) + 1

    pad_img = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)))

    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            if padding == 'valid':
                output[:, i, j] = np.sum(
                    images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
            else:
                output[:, i, j] = np.sum(
                    pad_img[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    * kernel, axis=(1, 2))
    return output
