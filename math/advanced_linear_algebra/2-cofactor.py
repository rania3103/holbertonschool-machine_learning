#!/usr/bin/env python3
"""a function that calculates the cofactor matrix of a matrix"""

minor = __import__('1-minor').minor


def cofactor(matrix):
    """Returns: the cofactor matrix of matrix"""

    if not isinstance(matrix, list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == []:
        raise TypeError('matrix must be a list of lists')
    n_rows = len(matrix)
    if any(len(row) != n_rows for row in matrix) or n_rows == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    minor_mat = minor(matrix)
    n_r = len(minor_mat)
    for row in range(n_r):
        for col in range(n_r):
            minor_mat[row][col] *= (-1) ** (col + row)
    return minor_mat
