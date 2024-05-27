#!/usr/bin/env python3
"""save_model:saves an entire model
Returns: None
load_model:loads an entire model
Returns: the loaded model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model"""
    network.save(filename)


def load_model(filename):
    """loads an entire model"""
    return K.models.load_model(filename)
