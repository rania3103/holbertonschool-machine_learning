#!/usr/bin/env python3
"""Write a function that creates mini-batches to be
used for training a neural network using
mini-batch gradient descent"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Returns: list of mini-batches containing
    tuples (X_batch, Y_batch)"""
    shuf_X, shuf_Y = shuffle_data(X, Y)
    num_batches = len(X) // batch_size
    batches = []
    for i in range(num_batches):
        start_ind = i * batch_size
        end_ind = start_ind + batch_size
        batch_X = shuf_X[start_ind:end_ind]
        batch_Y = shuf_Y[start_ind:end_ind]
        batches.append((batch_X, batch_Y))
    return batches
