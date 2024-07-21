#!/usr/bin/env python3
"""a function that calculates the adjugate matrix of a matrix"""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Returns: the adjugate matrix of matrix"""

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
    cof_matrix = cofactor(matrix)
    n_rows = len(cof_matrix)
    adj_matrix = []
    for col in range(n_rows):
        row_list = []
        for row in range(n_rows):
            row_list.append(cof_matrix[row][col])
        adj_matrix.append(row_list)
    return adj_matrix
