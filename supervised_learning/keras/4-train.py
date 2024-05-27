#!/usr/bin/env python3
"""a function that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Returns: the History object generated after training the model"""
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
