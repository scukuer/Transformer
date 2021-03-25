# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 23 Mar, 2021
Author : chenlangscu@163.com
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

src_path = os.path.abspath("..")
sys.path.append(src_path)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates   # 广播


#  位置编码
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def debug():
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)  # (1, 50, 512)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    print(tf.__version__)
    debug()
