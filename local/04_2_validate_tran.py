import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from numerai_tools.scoring import numerai_corr, correlation_contribution, neutralize
from numerapi import NumerAPI
from scripts.plot_summarize import plot_and_summarize_validation
from model_defs.transformer_layers import FeatureEmbedding, TransformerEncoderBlock
from tensorflow.keras import layers, models

# (Include custom layers FeatureEmbedding, TransformerEncoderBlock here)
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
    # ... loading and evaluating
    parser = argparse.ArgumentParser(description="Numerai transformer Evaluation (Local CLI)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--memory", choices=["low", "medium", "high"], default="high", help="Memory usage level for validation (low, medium, high). Higher memory usage will be slower but more accurate.")
    args = parser.parse_args()

    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]

    validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era", "data_type", "target"] + feature_cols)
    validation = validation[validation["data_type"] == "validation"]
    del validation["data_type"]
    # Download and join in the meta_model for the validation eras
    napi = NumerAPI()
    napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
    validation["meta_model"] = pd.read_parquet(f"v4.3/meta_model.parquet")["numerai_meta_model"]

    # Downsample to eras to reduce memory usage and speedup evaluation 
    # Default is slower and higher memory usage, but more accurate evaluation.
    if args.memory == "low":
        validation = validation[validation["era"].isin(validation["era"].unique()[::4])]
    if args.memory == "medium":
        validation = validation[validation["era"].isin(validation["era"].unique()[::2])]

    # Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
    # so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet",columns=["era"]) # Load training eras for embargo calculation
    last_train_era = int(train["era"].unique()[-1])
    eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
    validation = validation[~validation["era"].isin(eras_to_embargo)]


    print("Loading lgbm model...")
    with open("local/models/lgbm_models.pkl", "rb") as f: lgbm_models = pickle.load(f)
    print("Loading NN model...")
    nn_model = tf.keras.models.load_model("local/models/nn_model.keras", compile=False)
    print("Loading transformer model...")

    final_model = create_transformer_model(len(feature_cols))
    final_model.load_weights("local/models/transformer_model.h5")

    print("Transformer model loaded.")
    print("Models loaded. Generating predictions and evaluating...")

    # LGBM target-level predictions
    for target, model in lgbm_models.items():
        n_iter = getattr(model, "best_iteration_", None) or model.n_estimators
        validation[f"prediction_lgbm_{target}"] = model.predict(
            validation[feature_cols],
            num_iteration=n_iter
        )

    # LGBM ensemble
    lgbm_cols = [f"prediction_lgbm_{t}" for t in lgbm_models.keys()]
    validation["prediction_lgbm_ensemble"] = (
        validation.groupby("era")[lgbm_cols]
        .rank(pct=True)
        .mean(axis=1)
    )

    # NN prediction
    validation["prediction_nn"] = nn_model.predict(
        validation[feature_cols].values.astype(np.float32),
        batch_size=1024,
        verbose=1
    ).reshape(-1)

    # Rank NN per era
    validation["prediction_nn"] = validation.groupby("era")["prediction_nn"].rank(pct=True)

    # Transformer prediction
    validation["prediction_transformer"] = final_model.predict(
        validation[feature_cols].values.astype(np.float32),
        batch_size=1024,
        verbose=1
    ).reshape(-1)

    # Rank Transformer per era
    validation["prediction_transformer"] = validation.groupby("era")["prediction_transformer"].rank(pct=True)

    # Rank predictions per era before ensembling to put them on the same scale
    ranked = validation.groupby("era")[[
        "prediction_lgbm_ensemble",
        "prediction_nn",
        "prediction_transformer"
    ]].rank(pct=True)

    # Final combined ensemble
    validation["prediction"] = (
        0.25 * ranked["prediction_lgbm_ensemble"] +
        0.25 * ranked["prediction_nn"] +
        0.5 * ranked["prediction_transformer"]
    )
    
    # Rank final prediction per era
    validation["prediction"] = validation.groupby("era")["prediction"].rank(pct=True)

    # Per-era CORR
    per_era_corr = (
        validation[["era", "prediction", "target"]]
        .groupby("era")
        .apply(lambda x: numerai_corr(x[["prediction"]], x["target"]).iloc[0])
    )

    # Per-era MMC
    per_era_mmc = (
        validation[["era", "prediction", "meta_model", "target"]]
        .dropna()
        .groupby("era")
        .apply(lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]).iloc[0])
    )

    print("per_era_corr length:", len(per_era_corr))
    print(per_era_corr.head())

    print("per_era_mmc length:", len(per_era_mmc))
    print(per_era_mmc.head())

    metrics = plot_and_summarize_validation(per_era_corr, per_era_mmc)
    print("\nValidation Metrics")
    print(metrics.round(6).to_string())
    
    pass

if __name__ == "__main__":
    main()
