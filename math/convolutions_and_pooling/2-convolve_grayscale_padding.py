#!/usr/bin/env python3
"""a function that performs a same convolution on
grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding
    pad_img = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)))

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                pad_img[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return output
