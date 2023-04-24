"""
Implements original transformer model.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Union


@dataclass
class ModelHp:
    d_model: int
    num_attention_heads: int
    dropout_rate: Union[float, None]
    num_units: int
    activation: str
    vocab_size: int
    num_layers: int


def create_padding_mask(inputs: tf.Tensor):
    mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(inputs: tf.Tensor):
    seq_len = tf.shape(inputs)[1]
    mask = tf.ones((seq_len, seq_len), dtype=tf.float32)
    mask = 1 - tf.linalg.band_part(mask, -1, 0)
    padding_mask = create_padding_mask(inputs)
    return tf.maximum(mask, padding_mask)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = tf.cast(embedding_dim, dtype=tf.float32)
        self.embedding_matrix = self.build_embedding_matrix()

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"vocab_size": self.vocab_size, "embedding_dim": self.embedding_dim})
        return config

    def build_embedding_matrix(self):
        positions = tf.cast(tf.range(self.vocab_size)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(self.embedding_dim)[tf.newaxis, :], dtype=tf.float32)
        angle_rads = positions * (1 / tf.pow(10000, (2 * (i // 2)) / self.embedding_dim))

        sines = tf.math.sin(angle_rads[:, :2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return pos_encoding

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.embedding_dim[:, seq_len, :]



