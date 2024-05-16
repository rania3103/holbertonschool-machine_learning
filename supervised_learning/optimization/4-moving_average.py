#!/usr/bin/env python3
"""Write a function that calculates
the weighted moving average of a data set"""


def moving_average(data, beta):
    """Returns: a list containing the moving averages of data"""
    MVA = []
    avg = 0
    for i in range(len(data)):
        avg = avg * beta + data[i] * (1 - beta)
        MVA.append(avg / (1 - beta ** (i + 1)))
    return MVA
