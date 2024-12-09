#!/usr/bin/env python3
"""a function that randomly changes the brightness of an image."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Returns the altered image"""
    return tf.image.random_brightness(image, max_delta)
