#!/usr/bin/env python3
"""a class Dataset that loads and preps a dataset for machine translation"""
import transformers
import tensorflow_datasets as tfds


class Dataset:
    """Dataset class"""

    def __init__(self):
        """constructor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split="train",
            as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split="validation",
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for the dataset"""
        tokenizer_pt = transformers.BertTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased")
        vocab_size = 2**13
        tokenizer_pt = min(tokenizer_pt.vocab_size, vocab_size)
        tokenizer_en = min(tokenizer_en.vocab_size, vocab_size)
        return tokenizer_pt, tokenizer_en
