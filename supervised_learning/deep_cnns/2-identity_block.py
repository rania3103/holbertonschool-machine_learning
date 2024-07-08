#!/usr/bin/env python3
"""a function that builds an identity block
as described in Deep Residual Learning for Image Recognition (2015)"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Returns: the activated output of the identity block"""
    initializer = K.initializers.he_normal(seed=0)
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(
        1, 1), padding='same', kernel_initializer=initializer)(A_prev)

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
    relu3 = K.layers.Activation('relu')(bn3)

    final = K.layers.Add()([relu3, A_prev])
    return K.layers.Activation('relu')(final)
