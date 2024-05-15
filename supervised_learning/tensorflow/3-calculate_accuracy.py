#!/usr/bin/env python3
"""Write a function that calculates the accuracy of a prediction"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Returns: a tensor containing the decimal accuracy of the prediction"""
    correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
