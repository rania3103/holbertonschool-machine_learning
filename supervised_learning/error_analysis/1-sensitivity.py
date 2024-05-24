#!/usr/bin/env python3
""" a function that calculates the sensitivity
for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Returns: a numpy.ndarray of shape (classes,)
    containing the sensitivity of each class"""
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    return true_pos / (false_neg + true_pos)
