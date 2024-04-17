#!/usr/bin/env python3
"""code to plot x ↦ y as a scatter plot"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """data is plotted as magenta points"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.scatter(x, y, c='magenta')
    plt.show()
