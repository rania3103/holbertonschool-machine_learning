#!/usr/bin/env python3
"""a function that sets up Adam optimization for a keras model
with categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Returns: None"""
    opt = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
