#!/usr/bin/env python3
"""a class Normal that represents a Normal distribution"""


class Normal:
    """Normal class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            var = sum((numb - self.mean) ** 2 for numb in data) / len(data)
            self.stddev = var ** 0.5
            self.data = data

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return self.mean + (self.stddev * z)
