#!/usr/bin/env python3
"""Write a function  that sets up the RMSProp
optimization algorithm in TensorFlow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Returns: optimizer"""
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
