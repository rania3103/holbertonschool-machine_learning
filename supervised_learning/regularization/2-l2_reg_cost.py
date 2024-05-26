#!/usr/bin/env python3
"""write a function that calculates the
cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Returns: a tensor containing the total
    cost for each layer of the network,
    accounting for L2 regularization"""
    reg_cost = []
    for layer in model.layers:
        reg_cost.append(layer + cost)
    return reg_cost
