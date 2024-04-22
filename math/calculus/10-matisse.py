#!/usr/bin/env python3
"""a function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Returns a new list of coefficients representing
    the derivative of the polynomial"""
    if not isinstance(poly, list) or not poly:
        return None
    else:
        derv_coef = poly[1:]
        for i in range(1, len(derv_coef)):
            if derv_coef[i] != 0:
                derv_coef[i] *= i + 1
            else:
                return [0]
        return derv_coef
