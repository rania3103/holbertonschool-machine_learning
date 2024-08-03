#!/usr/bin/env python3
"""a function that  performs K-means on a dataset"""
import sklearn.cluster


def kmeans(X, k):
    """C( of shape (k, d) containing the centroid
    means for each cluster), clss(of shape (n,)
    containing the index of the cluster in C that
    each data point belongs to)"""
    kmeans_ = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_.fit(X)
    C = kmeans_.cluster_centers_
    clss = kmeans_.labels_
    return C, clss
