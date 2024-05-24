#!/usr/bin/env python3
""" a function that creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Returns: a confusion numpy.ndarray of shape (classes, classes)
    with row indices representing the correct labels
    and column indices representing the predicted labels"""
    classes = labels.shape[1]
    con_mat = np.zeros((classes, classes), dtype=int)
    max_ind_list_labels = np.argmax(labels, axis=1)
    max_ind_list_logits = np.argmax(logits, axis=1)
    for i, j in zip(max_ind_list_labels, max_ind_list_logits):
        con_mat[i, j] += 1
    return con_mat
