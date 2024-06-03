#!/usr/bin/env python3
"""a function that builds a modified version
of the LeNet-5 architecture using keras"""
from tensorflow import keras as K


def lenet5(X):
    """Returns: a K.Model compiled to use Adam
    optimization (with default hyperparameters)
    and accuracy metrics"""
    conv_lay = K.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        kernel_initializer=K.initializers.VarianceScaling(
            scale=2.0), activation='relu')(X)

    max_pool_lay = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv_lay)

    conv_lay2 = K.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        kernel_initializer=K.initializers.VarianceScaling(
            scale=2.0), activation='relu')(max_pool_lay)

    max_pool_lay2 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv_lay2)

    flat = K.layers.Flatten()(max_pool_lay2)

    fully_con1 = K.layers.Dense(
        units=120,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0),
        activation='relu')(flat)

    fully_con2 = K.layers.Dense(
        units=84,
        kernel_initializer=K.initializers.VarianceScaling(scale=2.0),
        activation='relu')(fully_con1)

    fully_con3 = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=K.initializers.VarianceScaling(
            scale=2.0))(fully_con2)
    model = K.Model(inputs=X, outputs=fully_con3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss=K.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
