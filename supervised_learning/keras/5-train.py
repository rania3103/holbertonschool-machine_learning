#!/usr/bin/env python3
"""a function that trains a model using mini-batch gradient descent
and to also analyze validaiton data"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """validation_data is the data to validate the model with, if not None"""
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)
