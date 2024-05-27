#!/usr/bin/env python3
"""a function that trains a model using mini-batch gradient descent
and also analyzes validaiton data and trains the model using early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """early_stopping is a boolean that indicates whether
    early stopping should be used
    patience is the patience used for early stopping
    """
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=patience)
        return network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[callback],
            verbose=verbose,
            shuffle=shuffle)
    else:
        return network.fit(data, labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose, shuffle=shuffle)
