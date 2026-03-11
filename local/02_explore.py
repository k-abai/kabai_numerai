import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from numerapi import NumerAPI


def main():
    parser = argparse.ArgumentParser(description="Numerai Data Exploration")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    parser.add_argument("--main", default="target_ender_20", help="Main target column to analyze (default: target)")
    args = parser.parse_args()

    with open("local/config.json") as f:
        config = json.load(f)
    
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)
    
    feature_cols = feature_metadata["feature_sets"][args.size]
    target_cols = config["target_cols"]

    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + target_cols
        )

    # Downsample to every 4th era to reduce memory usage and speedup model training (suggested for Colab free tier)
    # Comment out the line below to use all the data (higher memory usage, slower model training, potentially better performance)
    train = train[train["era"].isin(train["era"].unique()[::4])]

    # Drop `target` column
    MAIN_TARGET = args.main
    assert train["target"].equals(train[MAIN_TARGET])
    targets_df = train[["era"] + target_cols]

    (
    targets_df[target_cols]
        .corrwith(targets_df[MAIN_TARGET])
        .sort_values(ascending=False)
        .to_frame(args.main)
        .to_csv(f"local/reports/target_correlations_{args.main}.csv")
    )

    print(f"Loading data for size {args.size}...")
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + target_cols)
    print(f"Dataset shape: {train.shape}")
    
    train.groupby("era").size().plot(title="Number of rows per era")
    plt.savefig("local/reports/era_counts.png")

if __name__ == "__main__":
    main()
