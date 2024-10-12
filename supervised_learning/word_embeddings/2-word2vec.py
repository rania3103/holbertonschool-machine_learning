#!/usr/bin/env python3
"""a function that creates , builds and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Returns: the trained model"""
    if cbow:
        sg = 1
    else:
        sg = 0
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers)
    return model
