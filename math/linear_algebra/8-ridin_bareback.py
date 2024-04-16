#!/usr/bin/env python3
""" a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """returns a new matrix and f the two matrices
    cannot be multiplied, return None"""
    if len(mat1[0]) != len(mat2) or not mat1 or not mat2:
        return None

    def mat_with_zero(mat1, mat2):
        """create a new matrix filled with zero"""
        mat_zero = []
        for row in range(len(mat1)):
            new_row = []
            for col in range(len(mat2[0])):
                new_row.append(0)
            mat_zero.append(new_row)
        return mat_zero
    mat_zero = mat_with_zero(mat1, mat2)
    for row_mat1 in range(len(mat1)):
        for col_mat2 in range(len(mat2[0])):
            for row_mat2 in range(len(mat2)):
                mat_zero[row_mat1][col_mat2] += mat1[row_mat1][row_mat2] * \
                    mat2[row_mat2][col_mat2]
    return mat_zero
