from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="nckh")
class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention.
    query: (B, Hq)
    values: (B, T, Hv)
    returns: context (B, Hv), weights (B, T)
    """

    def __init__(self, attn_units: int, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = int(attn_units)
        self.Wq = layers.Dense(self.attn_units, use_bias=False)
        self.Wv = layers.Dense(self.attn_units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)

    def call(self, query, values):
        # query -> (B, 1, U)
        q = tf.expand_dims(self.Wq(query), axis=1)
        # values -> (B, T, U)
        v = self.Wv(values)
        score = self.V(tf.nn.tanh(q + v))  # (B, T, 1)
        weights = tf.nn.softmax(score, axis=1)  # (B, T, 1)
        context = tf.reduce_sum(weights * values, axis=1)  # (B, Hv)
        return context, tf.squeeze(weights, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"attn_units": self.attn_units})
        return cfg


def build_dual_cba(
    window_size: int = 11,
    cnn_dropout: float = 0.0,
    lstm_units: int = 96,
    attn_units: int = 64,
    dense_units_1: int = 256,
    dense_units_2: int = 512,
) -> keras.Model:
    """
    Dual-stream CNN-BiLSTM with Additive Attention (Dual-CBA).

    CNN branch:
      Conv1D filters: 32, 64, 96, 128; kernel=3; relu
      Last two conv layers use padding='same'
      Flatten()

    BiLSTM + Attention branch:
      Bidirectional LSTM returning full sequence + last states
      Query = concat(last forward h, last backward h)
      Values = full sequence output
      Bahdanau attention -> context vector

    Dense head:
      concat(CNN_flat, context) -> Dense relu -> Dense relu -> Dense(1) linear
    """
    inp = keras.Input(shape=(window_size, 1), name="window")

    # --- CNN branch ---
    c = layers.Conv1D(32, 3, activation="relu", padding="causal", name="cnn_conv1")(inp)
    c = layers.Conv1D(64, 3, activation="relu", padding="causal", name="cnn_conv2")(c)
    c = layers.Conv1D(96, 3, activation="relu", padding="same", name="cnn_conv3")(c)
    c = layers.Conv1D(128, 3, activation="relu", padding="same", name="cnn_conv4")(c)
    if cnn_dropout and cnn_dropout > 0:
        c = layers.Dropout(cnn_dropout)(c)
    c = layers.Flatten(name="cnn_flatten")(c)

    # --- BiLSTM branch ---
    lstm = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, return_state=True),
        name="bilstm",
    )
    seq, h_f, c_f, h_b, c_b = lstm(inp)
    query = layers.Concatenate(name="attn_query_concat")([h_f, h_b])  # (B, 2*lstm_units)
    context, weights = BahdanauAttention(attn_units, name="bahdanau_attention")(query, seq)

    # --- Dense head ---
    x = layers.Concatenate(name="fusion_concat")([c, context])
    x = layers.Dense(dense_units_1, activation="relu", name="dense_1")(x)
    x = layers.Dense(dense_units_2, activation="relu", name="dense_2")(x)
    out = layers.Dense(1, activation="linear", name="y")(x)

    model = keras.Model(inputs=inp, outputs=out, name="DualCBA")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def rolling_forecast(
    model: keras.Model,
    seed_window_scaled: tf.Tensor | list | tuple,
    horizon: int = 10,
) -> tf.Tensor:
    """
    Rolling 1-step forecast repeated `horizon` times.
    seed_window_scaled: shape (window,)
    returns: shape (horizon,)
    """
    w = tf.convert_to_tensor(seed_window_scaled, dtype=tf.float32)
    w = tf.reshape(w, [-1])
    preds = []
    window_size = int(w.shape[0])
    for _ in range(horizon):
        x = tf.reshape(w[-window_size:], [1, window_size, 1])
        yhat = tf.reshape(model(x, training=False), [-1])[0]
        preds.append(yhat)
        w = tf.concat([w, tf.reshape(yhat, [1])], axis=0)
    return tf.stack(preds, axis=0)

