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

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ffn_dim, self.dropout_rate = d_model, num_heads, ffn_dim, dropout_rate
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ffn_dim, activation='relu'), layers.Dense(d_model)])
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.dropout1, self.dropout2 = layers.Dropout(dropout_rate), layers.Dropout(dropout_rate)
    def call(self, x, training=False):
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_dim": self.ffn_dim, "dropout_rate": self.dropout_rate})
        return config

def create_transformer_model(num_features):
    inputs = layers.Input(shape=(num_features,))
    x = FeatureEmbedding(128)(inputs)
    pos_encoding = tf.Variable(tf.random.normal([1, num_features, 128], stddev=0.02), trainable=True)
    x = x + pos_encoding
    for _ in range(2): x = TransformerEncoderBlock(128, 4, 256, 0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Numerai Transformer Training (Local CUDA)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()
    setup_gpu()
    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]
    
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + ["target"])
    model = create_transformer_model(len(feature_cols))
    #model.fit(train[feature_cols].values.astype(np.float32), train["target"].values.astype(np.float32), epochs=20, batch_size=256, verbose=1)
    x_train = train[feature_cols].values.astype(np.float32)
    y_train = train["target"].values.astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(256)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, epochs=20, verbose=1)

    if not os.path.exists("local/models"): os.makedirs("local/models")
    model.save("local/models/transformer_model.keras")
    print("Final Transformer model saved to local/models/transformer_model.keras")

if __name__ == "__main__":
    main()
