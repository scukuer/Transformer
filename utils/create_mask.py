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

from utils.mask import *


def create_masks(inp, tar):
    """
    Encoder:
      - encoder_padding_mask (self attention of EncoderLayer)
    Decoder:
      - look_ahead_mask (self attention of DecoderLayer)
      - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
      - decoder_padding_mask (self attention of DecoderLayer)
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask


if __name__ == "__main__":
    print(tf.__version__)
