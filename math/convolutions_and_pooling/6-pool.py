#!/usr/bin/env python3
"""a function that performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Returns: a numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))
    return output
