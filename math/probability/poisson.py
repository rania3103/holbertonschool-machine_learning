#!/usr/bin/env python3
"""a class Poisson that represents a poisson distribution"""


class Poisson:
    """Poisson class"""

    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))
            self.data = data

    def pmf(self, k):
        """Calculates the value of the PMF for
        a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            calc = 1
            for i in range(1, k + 1):
                calc *= i
            return float((self.lambtha ** k) * (e ** (- self.lambtha)) / calc)

    def cdf(self, k):
        """Calculates the value of the CDF for
        a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf += self.pmf(i)
            return cdf
