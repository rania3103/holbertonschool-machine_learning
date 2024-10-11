#!/usr/bin/env python3
"""a function that creates a bag of words embedding matrix"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Returns: embeddings, features"""
    tokens = [sentence.lower().split() for sentence in sentences]
    if vocab is None:
        vocab = []
        for sentence in tokens:
            for word in sentence:
                vocab.append(word)
        vocab = sorted(set(vocab))
    embeddings = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(tokens):
        for word in sentence:
            if word in set(vocab):
                embeddings[i][vocab.index(word)] += 1
    return embeddings, vocab
