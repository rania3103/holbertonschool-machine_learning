#!/usr/bin/env python3
"""a function that trains a model using mini-batch gradient descent
and to also analyze validaiton data"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """validation_data is the data to validate the model with, if not None"""
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
