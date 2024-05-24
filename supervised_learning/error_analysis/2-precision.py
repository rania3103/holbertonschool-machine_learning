#!/usr/bin/env python3
""" a function that calculates the precision
for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """Returns: a numpy.ndarray of shape (classes,)
    containing the precision of each class"""
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    return true_pos / (false_pos + true_pos)
