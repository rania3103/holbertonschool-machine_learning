#!/usr/bin/env python3
"""a function that builds the inception network as
described in Going Deeper with Convolutions (2014)"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Returns: the keras model"""
    initializer = K.initializers.HeNormal()
    input_lay = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, (7, 7), strides=2,
                            activation='relu', padding='same',
                            kernel_initializer=initializer)(input_lay)

    pool1 = K.layers.MaxPool2D(
        (3, 3), strides=2, padding='same')(conv1)

    conv2 = K.layers.Conv2D(64, (1, 1), strides=1,
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(pool1)

    conv3 = K.layers.Conv2D(192, (3, 3), strides=1,
                            activation='relu', padding='same',
                            kernel_initializer=initializer)(conv2)

    pool2 = K.layers.MaxPool2D(
        (3, 3), strides=2, padding='same')(conv3)

    incep1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    incep2 = inception_block(incep1, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPool2D(
        (3, 3), strides=2, padding='same')(incep2)
    incep3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    incep4 = inception_block(incep3, [160, 112, 224, 24, 64, 64])
    incep5 = inception_block(incep4, [128, 128, 256, 24, 64, 64])
    incep6 = inception_block(incep5, [112, 144, 288, 32, 64, 64])
    incep7 = inception_block(incep6, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPool2D(
        (3, 3), strides=2, padding='same')(incep7)
    incep8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    incep9 = inception_block(incep8, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D((7, 7), strides=1)(incep9)
    drop = K.layers.Dropout(0.4)(avg_pool)
    output_lay = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer)(drop)
    return K.Model(inputs=input_lay, outputs=output_lay)
