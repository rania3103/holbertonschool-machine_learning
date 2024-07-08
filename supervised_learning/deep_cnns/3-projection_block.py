#!/usr/bin/env python3
"""a function that builds a projection block
as described in Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Returns: the activated output of the projection block"""
    initializer = K.initializers.he_normal(seed=0)
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(
        s, s), padding='same', kernel_initializer=initializer)(A_prev)

    bn1 = K.layers.BatchNormalization(axis=3)(conv1)

    relu1 = K.layers.Activation('relu')(bn1)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), strides=(
            1, 1), padding='same', kernel_initializer=initializer)(relu1)

    bn2 = K.layers.BatchNormalization(axis=3)(conv2)

    relu2 = K.layers.Activation('relu')(bn2)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            1, 1), padding='same', kernel_initializer=initializer)(relu2)

    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    con = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=initializer)(A_prev)

    bn_con = K.layers.BatchNormalization(axis=3)(con)

    final = K.layers.Add()([bn3, bn_con])
    return K.layers.Activation('relu')(final)
