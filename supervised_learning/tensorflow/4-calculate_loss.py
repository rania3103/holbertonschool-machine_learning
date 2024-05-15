#!/usr/bin/env python3
"""Write a function that calculates the softmax
cross-entropy loss of a prediction"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Returns: a tensor containing the loss of the prediction"""
    return tf.losses.softmax_cross_entropy(onehot_labels=y_pred, logits=y)
