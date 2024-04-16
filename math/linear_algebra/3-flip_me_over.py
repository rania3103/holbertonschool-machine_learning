#!/usr/bin/env python3
"""a function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """return a new matrix"""
    trans = []
    for col in range(len(matrix[0])):
        trans_row = []
        row = 0
        while (row < len(matrix)):
            trans_row.append(matrix[row][col])
            row += 1
        trans.append(trans_row)

    return trans
