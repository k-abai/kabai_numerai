import argparse
import json
import pandas as pd
import lightgbm as lgb
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description="Numerai LightGBM Training (Local CUDA)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--target", default="target")
    args = parser.parse_args()

    with open("local/config.json") as f:
        config = json.load(f)
    
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]
    TARGET_CANDIDATES = [args.target, "target_victor_20", "target_xerxes_20", "target_teager2b_20", "target_ender_20"]

    print(f"Loading data... (size: {args.size})")
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + TARGET_CANDIDATES)

    models = {}
    for target in TARGET_CANDIDATES:
        print(f"Training LightGBM model for {target} on CUDA...")
        model = lgb.LGBMRegressor(
            n_estimators=10000, 
            learning_rate=0.01,
            max_depth=5,
            num_leaves=2**5,
            colsample_bytree=0.1,
            device="cuda", 
            verbosity=-1
        )
        model.fit(train[feature_cols], train[target])
        models[target] = model

    if not os.path.exists("local/models"): os.makedirs("local/models")
    with open("local/models/lgbm_models.pkl", "wb") as f: pickle.dump(models, f)
    print("All LightGBM models saved to local/models/lgbm_models.pkl")

if __name__ == "__main__":
    main()
