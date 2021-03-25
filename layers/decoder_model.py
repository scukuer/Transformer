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
from layers.decoder_layer import DecoderLayer
from utils import conf

from layers.encoder_model import EncoderModel


class DecoderModel(tf.keras.layers.Layer):
    def __init__(self, num_layers, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        self.max_length = max_length
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.position_embedding = positional_encoding(max_length,
                                                      d_model)

        self.dropout = tf.keras.layers.Dropout(rate)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask):
        # x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(output_seq_len, self.max_length,
                                       "output_seq_len should be less or equal to self.max_length")

        attention_weights = {}

        # x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn1, attn2 = self.decoder_layers[i](x, encoding_outputs, training,
                                                     decoder_mask, encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i + 1)] = attn1
            attention_weights['decoder_layer{}_att2'.format(i + 1)] = attn2

        # x.shape: (batch_size, output_seq_len, d_model)
        return x, attention_weights


def debug():
    sample_decoder_model = DecoderModel(2, 8000, conf.MAX_LENGTH, 512, 8, 2048)

    sample_decoder_model_input = tf.random.uniform((64, 35))

    sample_encoder_model = EncoderModel(2, 8500, conf.MAX_LENGTH, 512, 8, 2048)
    sample_encoder_model_input = tf.random.uniform((64, 37))
    # (64, 37, 512)
    sample_encoder_model_output = sample_encoder_model(sample_encoder_model_input, False, encoder_padding_mask=None)

    sample_decoder_model_output, sample_decoder_model_att = sample_decoder_model(sample_decoder_model_input,
                                                                                 sample_encoder_model_output,
                                                                                 training=False, decoder_mask=None,
                                                                                 encoder_decoder_padding_mask=None)

    print(sample_decoder_model_output.shape)  # (64, 35, 512)
    for key in sample_decoder_model_att:
        # (64, 8, 35, 35)
        # (64, 8, 35, 37)
        # (64, 8, 35, 35)
        # (64, 8, 35, 37)
        print(sample_decoder_model_att[key].shape)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
