# !/usr/bin/python
# -*- coding:utf-8 -*-

"""
Created on 24 Mar, 2021
Author : chenlangscu@163.com
"""

import sys
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from utils.conf import *
from utils.learning_rate import optimizer
from utils.create_mask import create_masks
from utils.input_pipeline import tokenizer_en, tokenizer_pt
from models.transformer import Transformer

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

transformer = Transformer(num_layers, input_vocab_size, target_vocab_size,
                          MAX_LENGTH, d_model, num_heads, dff, dropout_rate)

checkpoint_path = os.path.join(checkpoint_dir, "train")

ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
print('Latest checkpoint restored!!')


def evaluate(inp_sentence):
    """
    eg: A B C D -> E F G H.
    Train: A B C D, E F G -> F G H
    Eval:  A B C D -> E
           A B C D, E -> F
           A B C D, E F -> G
           A B C D, E F G -> H
    """
    input_id_sentence = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(inp_sentence) + [tokenizer_pt.vocab_size + 1]
    # encoder_input.shape: (1, input_sentence_length)
    encoder_input = tf.expand_dims(input_id_sentence, 0)

    # decoder_input.shape: (1, 1)
    decoder_input = tf.expand_dims([tokenizer_en.vocab_size], 0)

    for i in range(MAX_LENGTH):
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(encoder_input, decoder_input)
        # predictions.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(
            encoder_input,
            decoder_input,
            False,
            encoder_padding_mask,
            decoder_mask,
            encoder_decoder_padding_mask)
        # predictions.shape: (batch_size, target_vocab_size)
        predictions = predictions[:, -1, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, tokenizer_en.vocab_size + 1):
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predicted_id]], axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights


def plot_encoder_decoder_attention(attention, input_sentence,
                                   result, layer_name):
    fig = plt.figure(figsize=(16, 8))

    input_id_sentence = tokenizer_pt.encode(input_sentence)

    # attention.shape: (num_heads, tar_len, input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        ax.matshow(attention[head][:-1, :])

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(['<start>'] + [tokenizer_pt.decode([i]) for i in input_id_sentence] + ['<end>'],
                           fontdict=fontdict, rotation=90)
        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


def translate(input_sentence, layer_name=''):
    result, attention_weights = evaluate(input_sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])

    print("Input: {}".format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))

    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence, result, layer_name)


def debug():
    translate('está muito frio aqui.')
    translate('isto é minha vida')
    translate('você ainda está em casa?')
    translate('este é o primeiro livro que eu já li')
    # translate('este é o primeiro livro que eu já li', layer_name='decoder_layer4_att2')


if __name__ == "__main__":
    print(tf.__version__)
    debug()
