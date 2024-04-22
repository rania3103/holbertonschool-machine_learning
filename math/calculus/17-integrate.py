#!/usr/bin/env python3
"""a function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Returns a new list of coefficients
    representing the integral of the polynomial"""
    if not isinstance(poly, list) or not poly or not isinstance(C, int):
        return None
    else:
        integral_coef = [C]
        integral_coef.append(poly[0])
        for i in range(1, len(poly)):
            if poly[i] == 0:
                integral_coef.append(0)
            else:
                integral_coef.append(poly[i] / (i + 1))
        return integral_coef
