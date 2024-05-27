#!/usr/bin/env python3
"""a function that trains a model using mini-batch gradient descent
and also analyzes validaiton data and trains the model using early stopping
and also trains the model with learning rate decay:"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """learning_rate_decay is a boolean that indicates
    whether learning rate decay should be used
    alpha is the initial learning rate
    decay_rate is the decay rate
    """
    callbacks = []

    if early_stopping and validation_data:
        early_stop_callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=patience)
        callbacks.append(early_stop_callback)

    if validation_data and learning_rate_decay:

        def scheduler(epoch):
            """calculates learning rate using inverse time decay"""
            new_lr = alpha / (1 + decay_rate * epoch)
            return new_lr

        lr_schedule_callback = K.callbacks.LearningRateScheduler(
            scheduler, verbose=1)
        callbacks.append(lr_schedule_callback)

        return network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle)
