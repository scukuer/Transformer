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

from utils.conf import *
from layers.encoder_model import EncoderModel
from layers.decoder_model import DecoderModel


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size, max_length, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_model = EncoderModel(num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate)

        self.decoder_model = DecoderModel(num_layers, target_vocab_size, max_length, d_model, num_heads, dff, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(inp, training, encoder_padding_mask)

        # decoding_outputs.shape: (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weight = self.decoder_model(tar, encoding_outputs, training,
                                                                decoder_mask, encoder_decoder_padding_mask)

        # prediction.shape: (batch_size, output_seq_len, target_vocab_size)
        prediction = self.final_layer(decoding_outputs)

        return prediction, attention_weight


def debug():
    sample_transformer = Transformer(2, 8500, 8000, MAX_LENGTH, 512, 8, 2048, rate=0.1)
    temp_input = tf.random.uniform((64, 26))
    temp_target = tf.random.uniform((64, 31))

    predictions, attention_weights = sample_transformer(temp_input, temp_target, training=False,
                                                        encoder_padding_mask=None, decoder_mask=None,
                                                        encoder_decoder_padding_mask=None)

    print(predictions.shape)  # (64, 31, 8000)
    for key in attention_weights:

        # decoder_layer1_att1  shape:  (64, 8, 31, 31)
        # decoder_layer1_att2  shape:  (64, 8, 31, 26)
        # decoder_layer2_att1  shape:  (64, 8, 31, 31)
        # decoder_layer2_att2  shape:  (64, 8, 31, 26)
        print(key, ' shape: ', attention_weights[key].shape)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
