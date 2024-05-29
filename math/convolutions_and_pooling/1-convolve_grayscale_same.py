#!/usr/bin/env python3
"""a function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    padding_h = (kh - 1) // 2
    padding_w = (kw - 1) // 2
    pad_img = np.pad(
        images, ((0, 0), (padding_h, padding_h), (padding_w, padding_w)))
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                pad_img[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return output
