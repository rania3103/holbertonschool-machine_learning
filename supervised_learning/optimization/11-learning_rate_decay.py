#!/usr/bin/env python3
"""Write a function  that updates the learning rate
using inverse time decay in numpy"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Returns: the updated value for alpha"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
