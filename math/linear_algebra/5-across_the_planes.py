#!/usr/bin/env python3
"""a function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """returns a new matrix (assuming mat1 and mat2 are 2D matrices)"""
    if len(mat1[0]) != len(mat2[0]) or len(mat1) != len(mat2):
        return None
    else:
        new_mat = []
        for row in range(2):
            new_row = []
            for col in range(2):
                new_row.append(mat1[row][col] + mat2[row][col])
            new_mat.append(new_row)
        return new_mat
