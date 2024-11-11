#!/usr/bin/env python3
"""a function that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """returns the computed policy"""
    z = np.dot(matrix, weight)
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient based on
    a state and a weight matrix."""
    probs = policy(state, weight)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    action = np.random.choice(len(probs[0]), p=probs[0])
    gradient = np.zeros_like(weight)
    gradient[:, action] = state * (1 - probs[0, action])
    for i in range(len(probs[0])):
        if i != action:
            gradient[:, i] = - state * probs[0, action] * probs[0, i]
    return action, gradient
