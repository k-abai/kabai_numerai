import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from numerai_tools.scoring import numerai_corr

def main():
    parser = argparse.ArgumentParser(description="Numerai NN Evaluation (Local CLI)")
    parser.add_argument("--size", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()

    with open("local/config.json") as f: config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]
    with open(f"{DATA_VERSION}/features.json") as f: feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"][args.size]

    validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era", "data_type", "target"] + feature_cols)
    validation = validation[validation["data_type"] == "validation"]

    with open("local/models/lgbm_models.pkl", "rb") as f: lgbm_models = pickle.load(f)
    final_model = tf.keras.models.load_model("local/models/nn_model.h5")

    for target in lgbm_models.keys(): validation[f"prediction_lgbm_{target}"] = lgbm_models[target].predict(validation[feature_cols])
    validation["prediction_lgbm_ensemble"] = validation.groupby("era")[[f"prediction_lgbm_{t}" for t in lgbm_models.keys()]].rank(pct=True).mean(axis=1)
    validation["prediction_nn"] = final_model.predict(validation[feature_cols].values.astype(np.float32)).flatten()

    cols = ["prediction_lgbm_ensemble", "prediction_nn"]
    corrs = validation.groupby("era").apply(lambda x: numerai_corr(x[cols], x["target"]))
    print(corrs.mean())

if __name__ == "__main__":
    main()
