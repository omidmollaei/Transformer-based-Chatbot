"""
Implements original transformer model.
"""

import tensorflow as tf
from dataclasses import dataclass


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


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = tf.cast(embedding_dim, dtype=tf.float32)
        self.embedding_matrix = self.build_embedding_matrix()

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"vocab_size": self.vocab_size, "embedding_dim": self.embedding_dim})
        return config

    def build_embedding_matrix(self):
        positions = tf.cast(tf.range(self.vocab_size)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(self.embedding_dim)[tf.newaxis, :], dtype=tf.float32)
        angle_rads = positions * (1 / tf.pow(10000, (2 * (i // 2)) / self.embedding_dim))

        sines = tf.math.sin(angle_rads[:, ::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return pos_encoding

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.embedding_matrix[:, seq_len, :]


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
        config.update({"num_heads": self.num_heads, "model_dim": self.model_dim})
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.keras.layers.Lambda(
            lambda x: tf.reshape(
                x, shape=(batch_size, -1, self.num_heads, self.depth))
        )(inputs)
        return tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs):
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


def encoder_layer(params: ModelHp, name: str = "encoder_layer"):
    inputs = tf.keras.layers.Input(shape=(None, params.d_model), name="inputs")
    padding_mask = tf.keras.layers.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttentionLayer(
        num_heads=params.num_attention_heads, model_dim=params.d_model, name="attention"
    )({
        "query": inputs,
        "key": inputs,
        "value": inputs,
        "mask": padding_mask
    })
    attention = tf.keras.layers.Dropout(params.dropout_rate)(attention)
    attention += tf.cast(inputs, dtype=tf.float32)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

    outputs = tf.keras.layers.Dense(params.num_units, activation=params.activation)(attention)
    outputs = tf.keras.layers.Dense(params.d_model)(outputs)
    outputs = tf.keras.layers.Dropout(params.dropout_rate)(outputs)
    outputs += attention
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(params: ModelHp, name: str = "encoder"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    padding_masks = tf.keras.layers.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(params.vocab_size, params.d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.d_model, dtype=tf.float32))
    embeddings = PositionalEncoding(params.vocab_size, params.d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(params.dropout_rate)(embeddings)
    for i in range(params.num_layers):
        outputs = encoder_layer(
            params, name=f"encoder_layer_{i}"
        )([outputs, padding_masks])

    return tf.keras.Model(inputs=[inputs, padding_masks], outputs=outputs, name=name)


def decoder_layer(params: ModelHp, name: str = "decoder_layer"):
    inputs = tf.keras.Input(shape=(None, params.d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, params.d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttentionLayer(
        num_heads=params.num_attention_heads, model_dim=params.d_model, name="attention_1"
    )(
        inputs={
            "query": inputs,
            "key": inputs,
            "value": inputs,
            "mask": look_ahead_mask,
        }
    )

    attention1 += tf.cast(inputs, dtype=tf.float32)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1)

    attention2 = MultiHeadAttentionLayer(
        num_heads=params.num_attention_heads, model_dim=params.d_model, name="attention_2"
    )(
        inputs={
            "query": attention1,
            "key": enc_outputs,
            "value": enc_outputs,
            "mask": padding_mask,
        }
    )
    attention2 = tf.keras.layers.Dropout(params.dropout_rate)(attention2)
    attention2 += attention1
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        attention2 + attention1
    )

    outputs = tf.keras.layers.Dense(params.num_units, activation=params.activation)(
        attention2
    )
    outputs = tf.keras.layers.Dense(params.d_model)(outputs)
    outputs = tf.keras.layers.Dropout(params.dropout_rate)(outputs)
    outputs += attention2
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )


def decoder(params: ModelHp, name: str = "decoder"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    enc_outputs = tf.keras.layers.Input(shape=(None, params.d_model), name="enc_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(params.vocab_size, params.d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(params.d_model, dtype=tf.float32))
    embeddings = PositionalEncoding(params.vocab_size, params.d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(params.dropout_rate)(embeddings)

    for i in range(params.num_layers):
        outputs = decoder_layer(
            params,
            name=f"decoder_layer_{i}",
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name,
    )


def encoder_decoder_transformer(params: ModelHp, name):
    """Build an encoder-decoder transformer model (i.e. original transformer model) with
     specified hyperparameters. """

    inputs = tf.keras.layers.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.layers.Input(shape=(None,), name="dec_inputs")

    padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="enc_padding_mask")(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask")(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name="dec_padding_mask")(inputs)

    encoder_output = encoder(params)(inputs=[inputs, padding_mask])
    decoder_outputs = decoder(params)(
        inputs=[dec_inputs, encoder_output, look_ahead_mask, dec_padding_mask]
    )

    outputs = tf.keras.layers.Dense(params.vocab_size, name="outputs")(decoder_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

