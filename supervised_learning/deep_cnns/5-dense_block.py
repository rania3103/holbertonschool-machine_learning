#!/usr/bin/env python3
"""a function that builds a dense block
as described in Densely Connected Convolutional Networks"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Returns:The concatenated output of each layer within
    the Dense Block and the number of filters within
    the concatenated outputs, respectively"""
    initializer = K.initializers.he_normal(seed=0)
    for i in range(layers):
        bn1 = K.layers.BatchNormalization(axis=-1)(X)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1), padding='same',
                                kernel_initializer=initializer)(relu1)

        bn2 = K.layers.BatchNormalization(axis=-1)(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3),
            padding='same', kernel_initializer=initializer)(relu2)

        X = K.layers.Concatenate(axis=-1)([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters
