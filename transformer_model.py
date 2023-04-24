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
    dropout_rate: float
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


def scaled_dot_product_attention(query, key, value, mask=None):
    """Calculate the attention weights."""
    depth = tf.cast(tf.shape(key)[-1], dtype=tf.float32)

    qk_matmul = tf.matmul(query, key, transpose_b=True)
    logits = qk_matmul / tf.math.sqrt(depth)

    if mask is not None:
        logits += mask * -1e9

    attention_weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.matmul(attention_weights, value)
    return outputs


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, model_dim: int, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_heads

        self.query_dense = tf.keras.layers.Dense(self.model_dim)
        self.key_dense = tf.keras.layers.Dense(self.model_dim)
        self.value_dense = tf.keras.layers.Dense(self.model_dim)
        self.dense = tf.keras.layers.Dense(self.model_dim)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "model_dim":self.model_dim})
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(
            lambda x: tf.reshape(
                x, shape=(batch_size, -1, self.num_heads, self.depth))
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs, mask):
        query, key, value, mask = (
            inputs['query'],
            inputs['key'],
            inputs['value'],
            inputs['mask'],
        )
        batch_size = tf.shape(query)[0]

        # linear layer
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3])
        )(scaled_attention)

        # concatenation of heads
        concat_attention = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=(batch_size, -1, self.model_dim))
        )(scaled_attention)

        outputs = self.dense(concat_attention)   # final linear layer
        return outputs


