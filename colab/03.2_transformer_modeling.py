import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
except ValueError:
    strategy = tf.distribute.get_strategy()

class FeatureEmbedding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        self.projection = layers.Dense(self.d_model)
    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        return self.projection(x)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ffn_dim, activation='relu'), layers.Dense(d_model)])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    def call(self, x, training=False):
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x

def create_transformer_model(num_features):
    inputs = layers.Input(shape=(num_features,))
    x = FeatureEmbedding(128)(inputs)
    pos_encoding = tf.Variable(tf.random.normal([1, num_features, 128], stddev=0.02), trainable=True)
    x = x + pos_encoding
    for _ in range(2):
        x = TransformerEncoderBlock(128, 4, 256, 0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

eras = train["era"].unique()
fold_size = len(eras) // K_FOLDS
for fold in range(K_FOLDS):
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size
    val_eras = eras[val_start:val_end]
    train_eras = [e for i, e in enumerate(eras) if i < (val_start - 4) or i >= (val_end + 4)]
    X_train = train[train["era"].isin(train_eras)][feature_cols].values.astype(np.float32)
    y_train = train[train["era"].isin(train_eras)][MAIN_TARGET].values.astype(np.float32)
    X_val = train[train["era"].isin(val_eras)][feature_cols].values.astype(np.float32)
    y_val = train[train["era"].isin(val_eras)][MAIN_TARGET].values.astype(np.float32)
    with strategy.scope():
        transformer = create_transformer_model(len(feature_cols))
    transformer.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=256, verbose=1)

print("\nTraining final transformer model...")
with strategy.scope():
    transformer_model = create_transformer_model(len(feature_cols))
transformer_model.fit(train[feature_cols].values.astype(np.float32), train[MAIN_TARGET].values.astype(np.float32), epochs=20, batch_size=256, verbose=1)
