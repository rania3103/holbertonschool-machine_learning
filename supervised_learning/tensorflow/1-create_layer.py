#!/usr/bin/env python3
"""
prev is the tensor output of the previous layer
n is the number of nodes in the layer to create
activation is the activation function that the layer should use
use tf.keras.initializers.VarianceScaling(mode='fan_avg')
to implementHe et. al initialization for the layer weights
*each layer should be given the name layer
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        dtype="float",
        name="layer",
        activation=activation)
    return layer(prev)
