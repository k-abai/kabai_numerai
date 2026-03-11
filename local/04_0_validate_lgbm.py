import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution, neutralize
from scripts.plot_summarize import plot_and_summarize_validation

def main():
    parser = argparse.ArgumentParser(description="Numerai Validation (Local CLI)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all", help="Feature set size to use for validation (small, medium, all)")
    parser.add_argument("--memory", choices=["low", "medium", "high"], default="high", help="Memory usage level for validation (low, medium, high). Higher memory usage will be slower but more accurate.")
    args = parser.parse_args()

    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    napi = NumerAPI()
    napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    if args.size == "small":
        feature_cols = feature_metadata["feature_sets"]["small"]
    elif args.size == "medium":
        feature_cols = feature_metadata["feature_sets"]["medium"]
    else:
        feature_cols = feature_metadata["feature_sets"]["all"] # Use all features for validation

    validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era", "data_type", "target"] + feature_cols)
    validation = validation[validation["data_type"] == "validation"]
    del validation["data_type"]
    # Download and join in the meta_model for the validation eras
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

    with open("local/models/lgbm_models.pkl", "rb") as f: models = pickle.load(f)
    for target, model in models.items():
        n_iter = getattr(model, "best_iteration_", None) or model.n_estimators

        validation[f"prediction_{target}"] = model.predict(
            validation[feature_cols],
            num_iteration=n_iter
        )
    

    validation["prediction"] = validation.groupby("era")[[f"prediction_{t}" for t in models.keys()]].rank(pct=True).mean(axis=1)
    validation["prediction"] = neutralize(
        validation[["prediction"]], validation[feature_cols], proportion=0.01)["prediction"]
    
    validation["prediction"] = validation.groupby("era")["prediction"].rank(pct=True)

    per_era_corr = validation.groupby("era").apply(lambda x: numerai_corr(x[["prediction"]], x["target"]).iloc[0])
        #print(f"Validation CORR Mean: {per_era_corr.mean():.6f}")
    
    # Compute the per-era mmc between our predictions, the meta model, and the target values
    per_era_mmc = validation.dropna().groupby("era").apply(lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]).iloc[0])
        #print(f"Validation MMC Mean: {per_era_mmc.mean():.6f}")
    
    print("per_era_corr length:", len(per_era_corr))
    print(per_era_corr.head())

    print("per_era_mmc length:", len(per_era_mmc))
    print(per_era_mmc.head())

    metrics = plot_and_summarize_validation(per_era_corr, per_era_mmc)
    print("\nValidation Metrics")
    print(metrics.round(6).to_string())


if __name__ == "__main__":
    main()
