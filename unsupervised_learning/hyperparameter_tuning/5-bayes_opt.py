#!/usr/bin/env python3
"""Create a class BayesianOptimization that
performs Bayesian optimization on
a noiseless 1D Gaussian process"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """BayesianOptimization class"""

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location"""
        if self.minimize:
            Y_best = np.min(self.gp.Y)
        else:
            Y_best = np.max(self.gp.Y)
        mu, sigma = self.gp.predict(self.X_s)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (Y_best - mu - self.xsi) / sigma
        EI = (Y_best - mu - self.xsi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """optimizes the black-box function"""
        for i in range(iterations):
            X_next, EI = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            ind_opt = np.argmin(self.gp.Y)
        else:
            ind_opt = np.argmax(self.gp.Y)
        X_opt, Y_opt = self.gp.X[ind_opt], self.gp.Y[ind_opt]
        return X_opt, Y_opt
