#!/usr/bin/env python3
"""a function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """returns a new list"""
    if len(arr1) != len(arr2):
        return None
    else:
        new_arr = []
        mat = []
        mat += [arr1]
        mat += [arr2]
        for col in range(len(arr1)):
            sum_col = 0
            for row in range(2):
                sum_col += mat[row][col]
            new_arr.append(sum_col)
        return new_arr
