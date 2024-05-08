#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""Write a function that returns
two placeholders, x and y, for the neural network"""


def create_placeholders(nx, classes):
    """Returns: placeholders named x and y"""
    x = tf.placeholder(dtype="float", shape=(None, nx), name="x")
    y = tf.placeholder(dtype="float", shape=(None, classes), name="y")
    return x, y
