#!/usr/bin/env python3
"""a function that rotates an image by 90 degrees counter-clockwise"""
import tensorflow as tf


def rotate_image(image, size):
    """Returns the rotated image"""
    return tf.image.rot90(image, k=1)
