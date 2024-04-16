#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """returns a list of integers"""
    if matrix:
        numb_rows = len(matrix)
        if isinstance(matrix[0], list):
            numb_cols = len(matrix[0])
            if isinstance(matrix[0][0], list):
                numb_items = len(matrix[0][0])
                return [numb_rows, numb_cols, numb_items]
            else:
                return [numb_rows, numb_cols]
        else:
            return [numb_rows]
    else:
        return []
