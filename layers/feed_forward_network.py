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


def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def debug():
    sample_ffn = feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)   # (64, 50, 512)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
