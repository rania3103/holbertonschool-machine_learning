#!/usr/bin/env python3
"""a function that creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Returns: embeddings, features"""
    tokens = [sentence.lower().split() for sentence in sentences]
    tokens = [[re.sub(r"[^\w]", "", word) for word in sentence if re.match(
        r"^\w+$", word)] for sentence in tokens]
    if vocab is None:
        vocab = set()
        for sentence in tokens:
            for word in sentence:
                vocab.add(word)
        vocab = sorted(vocab)
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(tokens):
        for word in sentence:
            if word in vocab:
                embeddings[i][vocab.index(word)] += 1
    return embeddings, vocab
