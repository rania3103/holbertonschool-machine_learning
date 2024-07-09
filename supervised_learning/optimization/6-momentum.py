#!/usr/bin/env python3
"""Write a function  that sets up the gradient descent
with momentum optimization algorithm in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Returns: optimizer"""
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
