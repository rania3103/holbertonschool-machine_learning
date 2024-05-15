#!/usr/bin/env python3
"""Write a function that normalizes (standardizes) a matrix:"""


def normalize(X, m, s):
    """Returns: The normalized X matrix"""
    return (X - m) / s
