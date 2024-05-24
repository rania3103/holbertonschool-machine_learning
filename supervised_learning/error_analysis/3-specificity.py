#!/usr/bin/env python3
""" a function that calculates the specificity
for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """Returns: a numpy.ndarray of shape (classes,)
    containing the specificity of each class"""
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
    return true_neg / (false_pos + true_neg)
