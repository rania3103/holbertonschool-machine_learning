#!/usr/bin/env python3
"""a function that performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h + 1 - kh
    out_w = w + 1 - kw
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return output
