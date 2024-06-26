#!/usr/bin/env python3
"""write a function that creates a neural network
layer in tensorFlow that includes L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Returns: the output of the new layer"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    reg = tf.keras.regularizers.l2(lambtha)
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=reg,
        kernel_initializer=initializer)(prev)
