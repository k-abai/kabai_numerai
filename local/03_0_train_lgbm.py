import argparse
import json
import pandas as pd
import lightgbm as lgb
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description="Numerai LightGBM Training (Local CUDA/CPU)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--target", default="target", help="Target column to train on (default: target_ender_20)")
    args = parser.parse_args()

    with open("local/config.json") as f:
        config = json.load(f)
    
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]
    TARGET_CANDIDATES = [args.target, 
        #"target_victor_20",  
        #"target_xerxes_20",
        #"target_jasper_20",
        #"target_teager2b_20",
        #"target_jeremy_20",
        #"target_ralph_20",
        #"target_claudia_20",
        ]
    print(f"Loading data... (size: {args.size})")
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + TARGET_CANDIDATES)

    # split eras into train and validation sets (we'll use the validation set for early stopping)
    eras = sorted(train["era"].unique())
    cut = int(len(eras) * 0.8)

    train_eras = eras[:cut]
    valid_eras = eras[cut:]

    train_mask = train["era"].isin(train_eras)
    valid_mask = train["era"].isin(valid_eras)

    models = {}
    for target in TARGET_CANDIDATES:
        print(f"Training LightGBM model for {target} on CUDA/CPU...")
        model = lgb.LGBMRegressor(
            n_estimators=2000, 
            learning_rate=0.01,
            max_depth=5,
            num_leaves=2**5-1,
            colsample_bytree=0.1,
            device="cpu", #Configure list: "GPU", "CUDA", "CPU"
            verbosity=-1
        )
        # We've found the following "deep" parameters perform much better, but they require much more CPU and RAM
# model = lgb.LGBMRegressor(
#     n_estimators=30_000,
#     learning_rate=0.001,
#     max_depth=10,
#     num_leaves=2**10,
#     colsample_bytree=0.1
#     min_data_in_leaf=10000,
# )
    model.fit(
        train.loc[train_mask, feature_cols],
        train.loc[train_mask, target],
        eval_set=[(
            train.loc[valid_mask, feature_cols],
            train.loc[valid_mask, target]
        )],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )

    models[target] = model        

    if not os.path.exists("local/models"): os.makedirs("local/models")
    with open("local/models/lgbm_models.pkl", "wb") as f: pickle.dump(models, f)
    print("All LightGBM models saved to local/models/lgbm_models.pkl")

if __name__ == "__main__":
    main()
