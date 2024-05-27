#!/usr/bin/env python3
"""a function that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Returns: the keras model"""
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    input_lay = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=reg,
        input_shape=(
            nx,
        ))
    model.add(input_lay)
    for i in range(1, len(layers)):
        if keep_prob < 1:
            dropout_layer = K.layers.Dropout(1 - keep_prob)
            model.add(dropout_layer)
        hidd_layer = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=reg)
        model.add(hidd_layer)
    return model
