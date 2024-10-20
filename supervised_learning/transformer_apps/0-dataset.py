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
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        pt_sentences = []
        en_sentences = []
        for pt, en in data:
            pt_sentences.append(pt.numpy().decode('utf-8'))
            en_sentences.append(en.numpy().decode('utf-8'))
        tokenizer_pt.train_new_from_iterator(pt_sentences, vocab_size=2**13)
        tokenizer_en.train_new_from_iterator(en_sentences, vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
