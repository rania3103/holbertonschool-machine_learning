#!/usr/bin/env python3
"""Write a function that creates the forward
propagation graph for the neural network"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Returns: the prediction of the network in tensor form"""
    neuralnetwork_archi = x
    for i in range(len(layer_sizes)):
        neuralnetwork_archi = create_layer(
            neuralnetwork_archi, layer_sizes[i], activations[i])
    return neuralnetwork_archi
