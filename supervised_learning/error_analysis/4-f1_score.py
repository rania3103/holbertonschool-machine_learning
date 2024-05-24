#!/usr/bin/env python3
""" a function that calculates the F1 score of a confusion matrix"""
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """Returns: a numpy.ndarray of shape (classes,)
    containing the  F1 score of each class"""
    prec = precision(confusion)
    recall = sensitivity(confusion)
    return 2 * prec * recall / (prec + recall)
