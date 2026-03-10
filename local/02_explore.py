import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Numerai Data Exploration")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()

    with open("local/config.json") as f:
        config = json.load(f)
    
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)
    
    feature_cols = feature_metadata["feature_sets"][args.size]
    target_cols = config["target_cols"]

    print(f"Loading data for size {args.size}...")
    train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"] + feature_cols + target_cols)
    print(f"Dataset shape: {train.shape}")
    
    train.groupby("era").size().plot(title="Number of rows per era")
    plt.savefig("local/era_counts.png")
    train["target"].hist(bins=50)
    plt.savefig("local/target_distribution.png")
    print("Plots saved in local/")

if __name__ == "__main__":
    main()
