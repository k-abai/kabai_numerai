import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental_set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

class FeatureEmbedding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        self.projection = layers.Dense(self.d_model)
    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        return self.projection(x)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

class LinformerAttention(layers.Layer):
    def __init__(self, d_model, num_heads, k, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.k = d_model, num_heads, k
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    
    def build(self, input_shape):
        seq_len = input_shape[1]
        self.E = self.add_weight(shape=(seq_len, self.k), initializer='glorot_uniform', trainable=True, name='E')
        self.F = self.add_weight(shape=(seq_len, self.k), initializer='glorot_uniform', trainable=True, name='F')

    def call(self, x, training=False):
        k_proj = tf.einsum('bsd,sk->bkd', x, self.E)
        v_proj = tf.einsum('bsd,sk->bkd', x, self.F)
        return self.mha(x, v_proj, k_proj, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "k": self.k})
        return config

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate, k=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ffn_dim, self.dropout_rate, self.k = d_model, num_heads, ffn_dim, dropout_rate, k
        if k:
            self.attn = LinformerAttention(d_model, num_heads, k)
        else:
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ffn_dim, activation='relu'), layers.Dense(d_model)])
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.dropout1, self.dropout2 = layers.Dropout(dropout_rate), layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        if self.k:
            attn_output = self.attn(x, training=training)
        else:
            attn_output = self.attn(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_dim": self.ffn_dim, "dropout_rate": self.dropout_rate, "k": self.k})
        return config

def create_transformer_model(num_features):
    inputs = layers.Input(shape=(num_features,))
    x = FeatureEmbedding(32)(inputs)
    pos_encoding = tf.Variable(tf.random.normal([1, num_features, 32], stddev=0.02), trainable=True)
    x = x + pos_encoding
    
    # ALBERT-style: Shared parameters across layers
    # Linformer-style: k=32 bottleneck for attention
    shared_encoder = TransformerEncoderBlock(32, 2, 64, 0.1, k=32)
    for _ in range(4): # Increased depth but shared parameters
        x = shared_encoder(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Numerai Transformer Training (Local CUDA)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--memory", choices=["low", "medium", "high"], default="high", help="Memory usage level for validation (low, medium, high). Higher memory usage will be slower but more accurate.")
    args = parser.parse_args()
    setup_gpu()
    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]
    
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + ["target"])
    
    # Downsample to eras to reduce memory usage and speedup evaluation 
    # Default is slower and higher memory usage, but more accurate evaluation.
    if args.memory == "low":
        train = train[train["era"].isin(train["era"].unique()[::4])]
    if args.memory == "medium":
        train = train[train["era"].isin(train["era"].unique()[::2])]

    
    model = create_transformer_model(len(feature_cols))
    #model.fit(train[feature_cols].values.astype(np.float32), train["target"].values.astype(np.float32), epochs=20, batch_size=256, verbose=1)
    x_train = train[feature_cols].values.astype(np.float32)
    y_train = train["target"].values.astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, epochs=5, verbose=1)

    if not os.path.exists("local/models"): os.makedirs("local/models")
    model.save("local/models/transformer_model.keras")
    print("Final Transformer model saved to local/models/transformer_model.keras")

if __name__ == "__main__":
    main()
