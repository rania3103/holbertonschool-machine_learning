#!/usr/bin/env python3
"""a function that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Returns: the keras model"""
    input_lay = K.layers.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    curr_lay = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=reg,
        input_shape=(
            nx,
        ))(input_lay)
    for i in range(1, len(layers)):
        if keep_prob < 1:
            curr_lay = K.layers.Dropout(1 - keep_prob)(curr_lay)
        curr_lay = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=reg)(curr_lay)

    model = K.models.Model(inputs=input_lay, outputs=curr_lay)
    return model
