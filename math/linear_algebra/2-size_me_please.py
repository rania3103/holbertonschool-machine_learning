#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """returns a list of integers"""
    if matrix:
        dim = []
        while isinstance(matrix, list):
            dim += [len(matrix)]
            if matrix:
                matrix = matrix[0]
            else:
                break
        return dim
    else:
        return []
