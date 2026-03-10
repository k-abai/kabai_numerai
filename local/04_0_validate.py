import argparse
import json
import pandas as pd
import pickle
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr

def main():
    parser = argparse.ArgumentParser(description="Numerai Validation (Local CLI)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()

    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    napi = NumerAPI()
    napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]

    validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era", "data_type", "target"] + feature_cols)
    validation = validation[validation["data_type"] == "validation"]
    
    with open("local/models/lgbm_models.pkl", "rb") as f: models = pickle.load(f)
    for target in models.keys(): validation[f"prediction_{target}"] = models[target].predict(validation[feature_cols])

    validation["prediction"] = validation.groupby("era")[[f"prediction_{t}" for t in models.keys()]].rank(pct=True).mean(axis=1)
    per_era_corr = validation.groupby("era").apply(lambda x: numerai_corr(x[["prediction"]], x["target"]))
    print(f"Validation CORR Mean: {per_era_corr.mean():.6f}")

if __name__ == "__main__":
    main()
