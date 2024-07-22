#!/usr/bin/env python3
"""a class Binomial that represents a Binomial distribution"""


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """constructor"""
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError('n must be a positive value')
            if not (0 < p < 1):
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            var = sum([(numb - mean) ** 2 for numb in data]) / len(data)
            p = 1 - (var / mean)
            n = round(mean / p)
            self.n = n
            self.p = float(mean / n)
            self.data = data
