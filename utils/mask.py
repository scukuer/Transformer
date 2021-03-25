# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 24 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)


# padding mask
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到注意力对数（logits）
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# look ahead mask
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def debug():
    # padding mask: 0的位置为1，非0的为0
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    padding_mask = create_padding_mask(x)
    print("padding mask: ", padding_mask)                 # (3, 1, 1, 5)

    # look ahead mask:右上角为1
    x = tf.random.uniform((1, 3))
    look_ahead_mask = create_look_ahead_mask(x.shape[1])
    print("look ahead mask: ", look_ahead_mask)           # (3, 3)


if __name__ == "__main__":
    print(tf.__version__)
    debug()
