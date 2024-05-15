#!/usr/bin/env python3
"""Write a function that calculates the accuracy of a prediction"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Returns: a tensor containing the decimal accuracy of the prediction"""
    acc = tf.metrics.accuracy(
        tf.argmax(
            y, axis=1), tf.argmax(
            y_pred, axis=1), tf.shape(y_pred)[1])[0]
    return tf.reduce_mean(acc)
