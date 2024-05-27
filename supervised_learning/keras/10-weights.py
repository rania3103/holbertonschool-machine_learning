#!/usr/bin/env python3
"""save_weights:saves a model’s weights
Returns: None
load_weights:loads a model’s weights
Returns: None"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """saves a model’s weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """loads a model’s weights"""
    network.load_weights(filename)
