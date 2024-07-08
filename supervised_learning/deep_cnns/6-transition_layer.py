#!/usr/bin/env python3
"""a function that builds a transition layer
as described in Densely Connected Convolutional Networks"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Returns:The output of the transition layer
    and the number of filters within the output, respectively"""
    initializer = K.initializers.he_normal(seed=0)
    nb_filters = int(compression * nb_filters)

    bn1 = K.layers.BatchNormalization(axis=-1)(X)
    relu1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=(1, 1), padding='same',
                            kernel_initializer=initializer)(relu1)
    avg_pool = K.layers.AveragePooling2D(
        (2, 2), strides=(2, 2), padding='same')(conv1)
    return avg_pool, nb_filters
