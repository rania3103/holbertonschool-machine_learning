#!/usr/bin/env python3
"""Write a function  that updates a variable in place
using the Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Returns: optimizer"""
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        epsilon=epsilon,
        beta_1=beta1,
        beta_2=beta2)
