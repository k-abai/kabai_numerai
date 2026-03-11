import argparse
import json
import pandas as pd
from numerapi import NumerAPI

def main():
    parser = argparse.ArgumentParser(description="Numerai Data Download")
    parser.add_argument("--version", choices=["v5.2", "v5.1", "v5.0"], default="v5.2", help="Data version (v5.2, v5.1, etc.)")
    args = parser.parse_args()

    napi = NumerAPI()
    DATA_VERSION = args.version

    print(f"Downloading features metadata for version {DATA_VERSION}...")
    napi.download_dataset(f"{DATA_VERSION}/features.json")

    with open(f"{DATA_VERSION}/features.json") as f:
        feature_metadata = json.load(f)

    feature_cols = feature_metadata["feature_sets"]
    target_cols = feature_metadata["targets"]

    print(f"Data: {DATA_VERSION} ({len(feature_cols)} features)")
    
    config = {
        "DATA_VERSION": DATA_VERSION,
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
