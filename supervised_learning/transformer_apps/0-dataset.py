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
        self.tokenizer_pt = self.tokenize_dataset(self.data_train)
        self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for the dataset"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        pt_sentences = [pt.numpy().decode('utf-8') for pt, en in data]
        en_sentences = [en.numpy().decode('utf-8') for pt, en in data]
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2**13)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2**13)
        return tokenizer_pt, tokenizer_en
