# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 24 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from utils.position_embedding import positional_encoding
from layers.encoder_layer import EncoderLayer
from utils import conf


class EncoderModel(tf.keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        # position_embedding.shape: (1, max_length, d_model)
        self.position_embedding = positional_encoding(max_length, self.d_model)

        self.dropout = tf.keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(input_seq_len, self.max_length,
                                       "input_seq_len should be less or equal to self.max_length")

        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :input_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        # x.shape: (batch_size, input_seq_len, d_model)
        return x


def debug():
    sample_encoder_model = EncoderModel(2, 8500, conf.MAX_LENGTH, 512, 8, 2048)
    sample_encoder_model_input = tf.random.uniform((64, 37))
    sample_encoder_model_output = sample_encoder_model(
        sample_encoder_model_input, False, encoder_padding_mask=None)
    print(sample_encoder_model_output.shape)  # (64, 37, 512)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
