#!/usr/bin/env python3
"""Write a function that creates the
training operation for the network"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ an operation that trains the network using gradient descent"""
    opt = tf.train.GradientDescentOptimizer(alpha, name="GradientDescent")
    return opt.minimize(loss)
