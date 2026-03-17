"""
Generate predictions from saved models and submit as CSV to Numerai.

Usage:
    python local/06_predict_submit.py

Credentials are read from env vars NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY.
Optional args:
    --size      Feature set: small | medium | all  (default: medium)
    --target    Comma-separated LGBM targets       (default: target_ender_20)
    --weights   Three floats: lgbm nn tran         (default: 0.5 0.25 0.25)
    --model_id  Numerai model UUID                 (default: legomax)
    --out       Path to save predictions CSV       (default: local/models/legomax_predictions.csv)
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from numerapi import NumerAPI

from model_defs.transformer_layers import FeatureEmbedding, TransformerEncoderBlock  # noqa: F401 — required for custom_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "all"], default="medium")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.5, 0.25, 0.25],
                        help="Weights for [lgbm, nn, transformer]. Must sum to 1.")
    parser.add_argument("--model_id", default="1cf82f5c-9803-43db-b2bc-b32b9647d505")
    parser.add_argument("--out", default="local/models/legomax_predictions.csv")
    args = parser.parse_args()

    public_id = os.environ.get("NUMERAI_PUBLIC_ID", "")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY", "")

    with open("local/config.json") as f:
        config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]

    with open(Path(DATA_VERSION) / "features.json") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]

    # Load models
    print("Loading LGBM models...")
    with open("local/models/lgbm_models.pkl", "rb") as f:
        lgbm_models = pickle.load(f)

    print("Loading NN model...")
    nn_model = tf.keras.models.load_model("local/models/nn_model.keras", compile=False)

    print("Loading transformer model...")
    transformer_model = tf.keras.models.load_model(
        "local/models/transformer_model.keras",
        compile=False,
        custom_objects={"FeatureEmbedding": FeatureEmbedding, "TransformerEncoderBlock": TransformerEncoderBlock},
    )

    lgbm_weight, nn_weight, tran_weight = args.weights

    requested_targets = [t.strip() for t in args.target.split(",") if t.strip()]
    available_targets = list(lgbm_models.keys())
    favorite_targets = []
    for t in requested_targets:
        if t in available_targets:
            favorite_targets.append(t)
        elif "target" in available_targets:
            print(f"Warning: target '{t}' not found, falling back to 'target'.")
            if "target" not in favorite_targets:
                favorite_targets.append("target")
    if not favorite_targets:
        favorite_targets = available_targets

    print(f"LGBM targets: {favorite_targets}")
    print(f"Weights — LGBM: {lgbm_weight}, NN: {nn_weight}, Transformer: {tran_weight}")

    # Download live data
    napi = NumerAPI(public_id or None, secret_key or None)
    live_path = f"{DATA_VERSION}/live.parquet"
    print("Downloading live data...")
    napi.download_dataset(f"{DATA_VERSION}/live.parquet", dest_path=live_path)

    live_features = pd.read_parquet(live_path)
    print(f"Live data shape: {live_features.shape}")
    X = live_features[feature_cols]

    # LGBM predictions
    lgbm_preds = pd.DataFrame(index=live_features.index)
    for target in favorite_targets:
        model = lgbm_models[target]
        n_iter = getattr(model, "best_iteration_", None) or model.n_estimators
        lgbm_preds[target] = model.predict(X, num_iteration=n_iter)
    lgbm_rank = lgbm_preds.rank(pct=True).mean(axis=1).rank(pct=True)

    # NN predictions
    nn_pred = nn_model.predict(X.values.astype(np.float32), batch_size=1024, verbose=0).reshape(-1)
    nn_rank = pd.Series(nn_pred, index=live_features.index).rank(pct=True)

    # Transformer predictions
    trans_pred = transformer_model.predict(X.values.astype(np.float32), batch_size=1024, verbose=0).reshape(-1)
    trans_rank = pd.Series(trans_pred, index=live_features.index).rank(pct=True)

    # Weighted blend + final rank
    ensemble = (lgbm_weight * lgbm_rank) + (nn_weight * nn_rank) + (tran_weight * trans_rank)
    submission = ensemble.rank(pct=True, method="first").to_frame("prediction")

    print(f"Predictions shape: {submission.shape}")
    print(submission.describe())

    submission.to_csv(args.out)
    print(f"Saved predictions to {args.out}")

    if not public_id or not secret_key:
        print("No credentials — skipping submission. Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY env vars.")
        return

    print(f"Submitting to model {args.model_id}...")
    submission_id = napi.upload_predictions(args.out, model_id=args.model_id)
    print(f"Submitted! submission_id={submission_id}")


if __name__ == "__main__":
    main()
