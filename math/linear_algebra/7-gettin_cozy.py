#!/usr/bin/env python3
"""a function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """returns a new matrix if possible otherwise none"""
    if not mat1 or not mat2:
        return None
    if len(mat1) != len(mat2):
        return None
    new_mat = []
    if axis == 0:
        new_mat += mat1
        new_mat += mat2
        return new_mat
    elif axis == 1:
        new_row = []
        for row in range(len(mat1)):
            temp = mat1[row] + mat2[row]
            new_row.append(temp)
        new_mat += new_row
        return new_mat
