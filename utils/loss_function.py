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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# 将 pad 对应的 loss 不参与计算
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 0的变为0 非0的变为1
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


if __name__ == "__main__":
    print(tf.__version__)
