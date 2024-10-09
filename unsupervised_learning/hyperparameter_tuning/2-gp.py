#!/usr/bin/env python3
"""Create a class GaussianProcess that represents
a noiseless 1D Gaussian process"""
import numpy as np


class GaussianProcess:
    """GaussianProcess class"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices"""
        squared_euc_dist = np.sum(X1 ** 2, 1).reshape(-1, 1) - 2 * \
            np.dot(X1, X2.T) + np.sum(X2 ** 2, 1)
        exp_term = np.exp(squared_euc_dist * -0.5 / (self.l ** 2))
        k = (self.sigma_f ** 2) * exp_term
        return k

    def predict(self, X_s):
        """predicts the mean and standard deviation
        of points in a Gaussian process"""
        cov1 = self.kernel(self.X, X_s)
        cov2 = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(self.K)
        mu = np.transpose(cov1) @ k_inv @ self.Y.flatten()
        sigma = cov2 - np.transpose(cov1) @ k_inv @ cov1
        return mu, np.diagonal(sigma)

    def update(self, X_new, Y_new):
        """updates a Gaussian Process"""
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
