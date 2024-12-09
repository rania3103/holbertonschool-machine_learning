#!/usr/bin/env python3
"""a function that randomly adjusts the contrast of an image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Returns the contrast-adjusted image"""
    return tf.image.random_contrast(image, lower=lower, upper=upper)
