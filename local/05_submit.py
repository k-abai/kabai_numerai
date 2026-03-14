import argparse
import cloudpickle
import json
import tensorflow as tf
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from numerapi import NumerAPI


# local/05_submit.py

def main():
    parser = argparse.ArgumentParser(description="Numerai model submission (Local CLI)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="medium", help="Feature set size to use")
    parser.add_argument("--target", type=str, default="target_ender_20", help="Main target(s) for LGBM (comma separated if multiple)")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.75, 0.25], help="Weights for LGBM ensemble and NN in final blend")
    parser.add_argument("--name", type=str, default="numerai_upload", help="Name for the output pickle file")
    args = parser.parse_args()

    # Load configuration
    with open("local/config.json") as f:
        config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]

    # Load feature metadata
    with open(Path(DATA_VERSION) / "features.json", "r") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]

    # Load Models
    print("Loading lgbm models...")
    with open("local/models/lgbm_models.pkl", "rb") as f:
        lgbm_models = pickle.load(f)

    print("Loading NN model...")
    nn_model = tf.keras.models.load_model("local/models/nn_model.keras", compile=False)

    # Commented out Transformer model for now
    # print("Loading transformer model...")
    # transformer_model = tf.keras.models.load_model("local/models/transformer_model.h5", compile=False)

    lgbm_weight = args.weights[0]
    nn_weight = args.weights[1]
    tran_weight = 1 - lgbm_weight - nn_weight
    # Split targets and filter out empty strings
    requested_targets = [t.strip() for t in args.target.split(",") if t.strip()]

    # Check which targets are actually available in lgbm_models
    available_targets = list(lgbm_models.keys())
    favorite_targets = []
    for t in requested_targets:
        if t in available_targets:
            favorite_targets.append(t)
        elif "target" in available_targets and t != "target":
            # Fallback to the generic "target" if requested specific isn't found
            print(f"Warning: Requested target '{t}' not found in lgbm_models. Falling back to 'target'.")
            if "target" not in favorite_targets:
                favorite_targets.append("target")
    
    if not favorite_targets and available_targets:
        print(f"Warning: No requested targets found. Using all available targets: {available_targets}")
        favorite_targets = available_targets

    print(f"Models loaded. LGBM targets to ensemble: {favorite_targets}")
    print(f"Weights: LGBM={lgbm_weight}, NN={nn_weight}")

    def predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame,
    ) -> pd.DataFrame:
        # Select only required features
        X = live_features[feature_cols]

        # 1. LGBM Predictions
        lgbm_preds = pd.DataFrame(index=live_features.index)
        for target in favorite_targets:
            model = lgbm_models[target]
            n_iter = getattr(model, "best_iteration_", None) or model.n_estimators
            lgbm_preds[target] = model.predict(X, num_iteration=n_iter)
        
        # Rank average LGBM targets
        if not lgbm_preds.empty:
            lgbm_ensemble_rank = lgbm_preds.rank(pct=True).mean(axis=1).rank(pct=True)
        else:
            lgbm_ensemble_rank = pd.Series(0.5, index=live_features.index)

        # 2. NN Predictions
        nn_pred = nn_model.predict(
            X.values.astype(np.float32),
            batch_size=1024,
            verbose=0
        ).reshape(-1)
        
        # Rank NN predictions
        nn_rank = pd.Series(nn_pred, index=live_features.index).rank(pct=True)

        # 3. Optional Transformer (Commented Out)
        # trans_pred = transformer_model.predict(X.values.astype(np.float32), batch_size=1024, verbose=0).reshape(-1)
        # trans_rank = pd.Series(trans_pred, index=live_features.index).rank(pct=True)

        # Final Weighted Blend
        ensemble = (lgbm_weight * lgbm_ensemble_rank) + (nn_weight * nn_rank) # + (tran_weight * trans_rank)
        
        # Final ranked submission
        submission = ensemble.rank(pct=True, method="first")
        return submission.to_frame("prediction")


    # Local smoke test with live data
    napi = NumerAPI()
    print("Downloading live data for smoke test...")
    napi.download_dataset(f"{DATA_VERSION}/live.parquet", dest_path=f"{DATA_VERSION}/live.parquet")
    
    sample_live = pd.read_parquet(
        f"{DATA_VERSION}/live.parquet",
        columns=feature_cols
    ).head(100) # Small sample for quick test

    print("Running prediction smoke test...")
    dummy_bench = pd.DataFrame(index=sample_live.index)
    preds = predict(sample_live, dummy_bench)
    print("Smoke test predictions (head):")
    print(preds.head())

    # Save for upload
    out_path = f"local/models/{args.name}.pkl"
    print(f"Saving upload pickle to: {out_path}")
    with open(out_path, "wb") as f:
        f.write(cloudpickle.dumps(predict))

    print("Done!")


if __name__ == "__main__":
    main()
