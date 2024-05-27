#!/usr/bin/env python3
"""a function that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Returns: the loss and accuracy of the model
    with the testing data, respectively"""
    test_loss, test_acc = network.evaluate(data, labels, verbose=verbose)
    return test_loss, test_acc
