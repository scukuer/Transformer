# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 24 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from utils.conf import *


# lrate = (d_model ** -0.5) * min(step_num ** (-0.5), step_num * warm_up_steps **(-1.5))


class CustomizedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))

        arg3 = tf.math.rsqrt(self.d_model)

        return arg3 * tf.math.minimum(arg1, arg2)


learning_rate = CustomizedSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def debug():
    temp_learning_rate_schedule = CustomizedSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Leraning rate")
    plt.xlabel("Train step")


if __name__ == "__main__":
    print(tf.__version__)
    debug()
