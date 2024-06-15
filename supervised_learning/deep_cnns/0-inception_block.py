#!/usr/bin/env python3
"""a function that builds an inception block
as described in Going Deeper with Convolutions (2014)"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Returns: the concatenated output of the inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv_1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu')(A_prev)

    conv_3_red = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                             activation='relu')(conv_3_red)

    conv_5_red = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                             activation='relu')(conv_5_red)

    max_pool = K.layers.MaxPooling2D((3, 3), strides=(
        1, 1), padding='same')(A_prev)
    max_pool_conv = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu')(max_pool)

    output = K.layers.Concatenate(
        axis=-1)([conv_1, conv_3, conv_5, max_pool_conv])
    return output
