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
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e: print(e)

def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Numerai NN Training (Local CUDA)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    setup_gpu()
    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]
    
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + ["target"])

    print("Training final model on all data...")
    final_model = create_model(len(feature_cols))
    final_model.fit(train[feature_cols], train["target"], epochs=args.epochs, batch_size=256, verbose=1)
    
    if not os.path.exists("local/models"): os.makedirs("local/models")
    final_model.save("local/models/nn_model.h5")
    print("Final NN model saved to local/models/nn_model.h5")

if __name__ == "__main__":
    main()
