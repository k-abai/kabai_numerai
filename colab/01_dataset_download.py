from numerapi import NumerAPI
napi = NumerAPI()

all_datasets = napi.list_datasets()
dataset_versions = list(set(d.split('/')[0] for d in all_datasets))
print("Available versions:\n", dataset_versions)

DATA_VERSION = "v5.2"

current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]
print("Available", DATA_VERSION, "files:\n", current_version_files)

import json

napi.download_dataset(f"{DATA_VERSION}/features.json")

feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
feature_sets = feature_metadata["feature_sets"]
feature_cols = feature_sets["all"]
target_cols = feature_metadata["targets"]

# --------------------------------------------------------------------------
# EDIT THESE TO CHANGE YOUR ENSEMBLE TARGETS
# --------------------------------------------------------------------------
MAIN_TARGET = "target" 
TARGET_CANDIDATES = [
  MAIN_TARGET,
  "target_victor_20",
  "target_xerxes_20",
  "target_teager2b_20",
  "target_ender_20"
]
# --------------------------------------------------------------------------

import pandas as pd
napi.download_dataset(f"{DATA_VERSION}/train.parquet")

train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era"] + feature_cols + target_cols
)
train["target"] = train[MAIN_TARGET]
train = train[train["era"].isin(train["era"].unique()[::4])]
