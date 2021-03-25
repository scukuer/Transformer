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

from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward_network import feed_forward_network

from layers.encoder_layer import EncoderLayer


class DecoderLayer(tf.keras.layers.Layer):
    """
        x -> self attention -> add & normalize & dropout -> out1
        out1 , encoding_outputs -> attention -> add & normalize & dropout -> out2
        out2 -> ffn -> add & normalize & dropout -> out3
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, decoder_mask, encoder_decoder_padding_mask):
        # decoder_mask: 由look_ahead_mask和decoder_padding_mask合并而来
        # x.shape: (batch_size, target_seq_len, d_model)
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)

        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # (batch_size, target_seq_len, d_model)
        # q: out1
        # k: enc_output
        # v: enc_output
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, encoder_decoder_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)                         # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)                                  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)                    # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


def debug():
    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    # (64, 43, 512)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)),
                                                             sample_encoder_layer_output, False, None, None)

    # (batch_size, target_seq_len, d_model)
    print("sample_decoder_layer_output shape: ", sample_decoder_layer_output.shape)  # (64, 50, 512)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
