#!/usr/bin/env python3
"""a function that performs element-wise addition,
subtraction, multiplication, and division"""
import numpy as np


def np_elementwise(mat1, mat2):
    """returns a tuple containing the element-wise sum,
    difference, product, and quotient, respectively"""
    return (
        np.add(
            mat1, mat2), np.subtract(
            mat1, mat2), np.multiply(
                mat1, mat2), np.divide(
                    mat1, mat2))
