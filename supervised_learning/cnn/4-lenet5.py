#!/usr/bin/env python3
"""a function that builds a modified version
of the LeNet-5 architecture using tensorflow"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Returns:
    *a tensor for the softmax activated output
    *a training operation that utilizes Adam
    optimization (with default hyperparameters)
    *a tensor for the loss of the netowrk
    *a tensor for the accuracy of the network
    """
    conv_lay = tf.layers.conv2d(
        inputs=x, filters=6, kernel_size=(5, 5), padding='same',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0), activation=tf.nn.relu)

    max_pool_lay = tf.layers.max_pooling2d(
        inputs=conv_lay, pool_size=(2, 2), strides=(2, 2))

    conv_lay2 = tf.layers.conv2d(
        inputs=max_pool_lay, filters=16, kernel_size=(5, 5), padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0), activation=tf.nn.relu)

    max_pool_lay2 = tf.layers.max_pooling2d(
        inputs=conv_lay2, pool_size=(2, 2), strides=(2, 2))

    flat = tf.layers.flatten(inputs=max_pool_lay2)

    fully_con1 = tf.layers.dense(
        inputs=flat,
        units=120,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        activation=tf.nn.relu)

    fully_con2 = tf.layers.dense(
        inputs=fully_con1,
        units=84,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        activation=tf.nn.relu)

    output = tf.layers.dense(
        inputs=fully_con2,
        units=10,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        activation=tf.nn.softmax)

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=output))
    grad_desc = tf.train.AdamOptimizer().minimize(loss)
    return output, grad_desc, loss, acc
