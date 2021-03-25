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
import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)

from utils.conf import *
from utils.input_pipeline import tokenizer_en, tokenizer_pt, gene_dataset
from utils.create_mask import create_masks
from utils.loss_function import loss_function
from utils.learning_rate import optimizer
from models.transformer import Transformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

transformer = Transformer(num_layers, input_vocab_size, target_vocab_size,
                          MAX_LENGTH, d_model, num_heads, dff, dropout_rate)

# checkpoint_dir = './checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                  model=transformer)
# if not os.path.exists(checkpoint_dir):
#     os.mkdir(checkpoint_dir)
#
# if tf.train.latest_checkpoint(checkpoint_dir):
#     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# 创建检查点的路径和检查点管理器（manager）,这将用于在每 n 个周期（epochs）保存检查点
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, "train")
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_dataset, val_dataset = gene_dataset()

temp_inp, temp_tar = iter(train_dataset.take(1)).next()

print(temp_inp.shape)  # (64, 34)
print(temp_tar.shape)  # (64, 37)
create_masks(temp_inp, temp_tar)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# @tf.function
#
# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。该函数专用于参数张量的精确形状
# 为了避免由于可变序列长度或可变批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定更多的通用形状

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, encoder_padding_mask, decoder_mask,
                                     encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(),
                                                                         train_accuracy.result()))
            # checkpoint.save(file_prefix=checkpoint_prefix)   # 和上面注释的代码对应

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time take for 1 epoch: {} secs\n'.format(time.time() - start))


from eval.predict import translate

translate('está muito frio aqui')
translate('isto é minha vida')
translate('você ainda está em casa?')
translate('este é o primeiro livro que eu já li')




