import argparse
import json
import pandas as pd
from numerapi import NumerAPI

def main():
    parser = argparse.ArgumentParser(description="Numerai Data Download")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all", help="Data size to use (small, medium, all)")
    args = parser.parse_args()

    napi = NumerAPI()
    DATA_VERSION = "v5.2"

    print(f"Downloading features metadata for version {DATA_VERSION}...")
    napi.download_dataset(f"{DATA_VERSION}/features.json")

    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)

    feature_cols = feature_metadata["feature_sets"][args.size]
    target_cols = feature_metadata["targets"]

    print(f"Data size selected: {args.size} ({len(feature_cols)} features)")
    
    config = {
        "DATA_VERSION": DATA_VERSION,
        "feature_size": args.size,
        "num_features": len(feature_cols),
        "target_cols": target_cols
    }
    with open("local/config.json", "w") as f:
        json.dump(config, f)

    print(f"Downloading training data for version {DATA_VERSION}...")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")
    print("Download complete.")

if __name__ == "__main__":
    main()
