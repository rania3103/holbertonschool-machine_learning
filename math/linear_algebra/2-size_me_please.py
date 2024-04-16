#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """returns a list of integers"""
    if matrix:
        numb_rows = len(matrix)
        if not isinstance(matrix[0], list):
            return [numb_rows]
        elif isinstance(matrix[0], list) and not isinstance(matrix[0][0], list):
            return [numb_rows, len(matrix[0])]
        else:
            while isinstance(matrix[0][0], list):
                matrix = matrix[0]
            numb_cols = len(matrix)
            return [numb_rows, numb_cols, len(matrix[0])]
    else:
        return []
