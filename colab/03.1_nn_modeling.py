import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EMBARGO = 4 
K_FOLDS = 5

def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model

eras = train["era"].unique()
fold_size = len(eras) // K_FOLDS
all_histories = []

for fold in range(K_FOLDS):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size
    val_eras = eras[val_start:val_end]
    train_eras = [e for i, e in enumerate(eras) if i < (val_start - EMBARGO) or i >= (val_end + EMBARGO)]
    
    X_train = train[train["era"].isin(train_eras)][feature_cols]
    y_train = train[train["era"].isin(train_eras)]["target"]
    X_val = train[train["era"].isin(val_eras)][feature_cols]
    y_val = train[train["era"].isin(val_eras)]["target"]
    
    nn_model = create_model(X_train.shape[1])
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    all_histories.append(history.history)

print("\nTraining final model on all data...")
final_model = create_model(len(feature_cols))
final_model.fit(train[feature_cols], train["target"], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
