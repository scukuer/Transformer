# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 23 Mar, 2021
Author : chenlangscu@163.com
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

src_path = os.path.abspath("..")
sys.path.append(src_path)

from utils import conf

print(tf.__version__)

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


def filter_max_length(x, y, max_length=conf.MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


def gene_dataset():
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)

    # 将数据集缓存到内存中以加快读取速度
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(conf.BUFFER_SIZE).padded_batch(conf.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(conf.BATCH_SIZE)
    return train_dataset, val_dataset


def debug():
    train_dataset, val_dataset = gene_dataset()
    pt_batch, en_batch = next(iter(val_dataset))
    print(pt_batch, en_batch)


if __name__ == "__main__":
    print(tf.__version__)
    debug()