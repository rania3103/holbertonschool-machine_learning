#!/usr/bin/env python3
"""  a function that performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Returns: clss, a numpy.ndarray of shape (n,) containing the cluster indices for each data point """
    ward_link_matrix = scipy.cluster.hierarchy.linkage(X, method="ward")
    clss = scipy.cluster.hierarchy.fcluster(
        ward_link_matrix, dist, criterion="distance")
    scipy.cluster.hierarchy.dendrogram(ward_link_matrix, color_threshold=dist)
    plt.show()
    return clss
