#!/usr/bin/env python3
"""a class Dataset that loads and preps a dataset for machine translation"""
import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


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
            split="validate",
            as_supervised=True)
        self.tokenizer_pt = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for the dataset"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        pt_tokenized = []
        en_tokenized = []
        for pt, en in data:
            pt_tokenized.append(
                tokenizer_pt.encode(
                    pt.numpy().decode('utf-8')))
            en_tokenized.append(
                tokenizer_en.encode(
                    en.numpy().decode('utf-8')))
        pt_tensor = tf.ragged.constant(pt_tokenized, dtype=tf.int32)
        en_tensor = tf.ragged.constant(en_tokenized, dtype=tf.int32)
        return pt_tensor, en_tensor
