"""
Targeted research: 2 experiments to improve CORR Sharpe.

Experiment 1 — Neutralization sweep:
    Vary neutralization proportion across [0, 0.01, 0.05, 0.10, 0.15, 0.20].
    Baseline is 0.01. Goal: reduce std without losing mean.

Experiment 2 — Ensemble weight search:
    Grid search over (lgbm_w, nn_w, tran_w) summing to 1, step 0.1.
    Find weights that maximise CORR Sharpe on validation.

Usage:
    python local/07_td_research.py
    python local/07_td_research.py --memory medium  # faster, less accurate
"""
import argparse
import json
import pickle
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution, neutralize

from model_defs.transformer_layers import FeatureEmbedding, TransformerEncoderBlock  # noqa: F401


REPORT_DIR = Path("local/reports")
BASELINE = {"lgbm": 0.5, "nn": 0.25, "tran": 0.25, "neutralization": 0.01}


def sharpe(s: pd.Series) -> float:
    s = s.dropna()
    return s.mean() / s.std(ddof=0) if s.std(ddof=0) != 0 else np.nan


def load_validation(data_version: str, feature_cols: list, memory: str) -> pd.DataFrame:
    napi = NumerAPI()
    napi.download_dataset(f"{data_version}/validation.parquet")
    napi.download_dataset("v4.3/meta_model.parquet", round_num=842)

    val = pd.read_parquet(
        f"{data_version}/validation.parquet",
        columns=["era", "data_type", "target"] + feature_cols,
    )
    val = val[val["data_type"] == "validation"].drop(columns="data_type")

    train_eras = pd.read_parquet(f"{data_version}/train.parquet", columns=["era"])
    last_train_era = int(train_eras["era"].unique()[-1])
    embargo = [str(last_train_era + i).zfill(4) for i in range(4)]
    val = val[~val["era"].isin(embargo)]

    val["meta_model"] = pd.read_parquet("v4.3/meta_model.parquet")["numerai_meta_model"]

    if memory == "low":
        val = val[val["era"].isin(val["era"].unique()[::4])]
    elif memory == "medium":
        val = val[val["era"].isin(val["era"].unique()[::2])]

    return val


def get_raw_predictions(val: pd.DataFrame, feature_cols: list,
                        lgbm_models: dict, nn_model, transformer_model) -> dict:
    X = val[feature_cols].values.astype(np.float32)
    X_df = val[feature_cols]

    # LGBM
    lgbm_preds = pd.DataFrame(index=val.index)
    for target, model in lgbm_models.items():
        n_iter = getattr(model, "best_iteration_", None) or model.n_estimators
        lgbm_preds[target] = model.predict(X_df, num_iteration=n_iter)
    lgbm_raw = lgbm_preds.rank(pct=True).mean(axis=1)

    # NN
    nn_raw = pd.Series(
        nn_model.predict(X, batch_size=2048, verbose=0).reshape(-1),
        index=val.index,
    )

    # Transformer
    tran_raw = pd.Series(
        transformer_model.predict(X, batch_size=2048, verbose=0).reshape(-1),
        index=val.index,
    )

    return {"lgbm": lgbm_raw, "nn": nn_raw, "tran": tran_raw}


def blend_and_score(val: pd.DataFrame, raw: dict, feature_cols: list,
                    lgbm_w: float, nn_w: float, tran_w: float,
                    neutralization: float) -> dict:
    lgbm_r = val.groupby("era")[val.index].apply(lambda _: None)  # placeholder

    # Per-era rank each signal, then blend
    def per_era_rank(s):
        return val.groupby("era")[s.name].rank(pct=True)

    ranked = pd.DataFrame({
        "lgbm": val.groupby("era").apply(lambda g: raw["lgbm"][g.index].rank(pct=True)).droplevel(0),
        "nn":   val.groupby("era").apply(lambda g: raw["nn"][g.index].rank(pct=True)).droplevel(0),
        "tran": val.groupby("era").apply(lambda g: raw["tran"][g.index].rank(pct=True)).droplevel(0),
    })

    ensemble = (lgbm_w * ranked["lgbm"] + nn_w * ranked["nn"] + tran_w * ranked["tran"])

    val2 = val.copy()
    val2["prediction"] = ensemble

    if neutralization > 0:
        val2["prediction"] = neutralize(val2[["prediction"]], val2[feature_cols], proportion=neutralization)["prediction"]

    val2["prediction"] = val2.groupby("era")["prediction"].rank(pct=True)

    per_era_corr = val2.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]], x["target"]).iloc[0]
    )
    per_era_mmc = val2.dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]).iloc[0]
    )

    return {
        "corr_mean": per_era_corr.mean(),
        "corr_std": per_era_corr.std(ddof=0),
        "corr_sharpe": sharpe(per_era_corr),
        "mmc_mean": per_era_mmc.mean(),
        "mmc_sharpe": sharpe(per_era_mmc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args()

    with open("local/config.json") as f:
        config = json.load(f)
    DATA_VERSION = config["DATA_VERSION"]

    with open(Path(DATA_VERSION) / "features.json") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"]["medium"]

    print("Loading models...")
    with open("local/models/lgbm_models.pkl", "rb") as f:
        lgbm_models = pickle.load(f)
    nn_model = tf.keras.models.load_model("local/models/nn_model.keras", compile=False)
    transformer_model = tf.keras.models.load_model(
        "local/models/transformer_model.keras",
        compile=False,
        custom_objects={"FeatureEmbedding": FeatureEmbedding, "TransformerEncoderBlock": TransformerEncoderBlock},
    )

    print(f"Loading validation data (memory={args.memory})...")
    val = load_validation(DATA_VERSION, feature_cols, args.memory)
    print(f"Validation shape: {val.shape}, eras: {val['era'].nunique()}")

    print("Computing raw model predictions (one-time)...")
    raw = get_raw_predictions(val, feature_cols, lgbm_models, nn_model, transformer_model)

    # ── Experiment 1: Neutralization sweep ──────────────────────────────────
    print("\n=== Experiment 1: Neutralization Sweep ===")
    neut_values = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]
    neut_results = []
    lw, nw, tw = BASELINE["lgbm"], BASELINE["nn"], BASELINE["tran"]
    for neut in neut_values:
        scores = blend_and_score(val, raw, feature_cols, lw, nw, tw, neut)
        neut_results.append({"neutralization": neut, **scores})
        print(f"  neut={neut:.2f}  corr_sharpe={scores['corr_sharpe']:.4f}  "
              f"mmc_sharpe={scores['mmc_sharpe']:.4f}  corr_mean={scores['corr_mean']:.4f}  corr_std={scores['corr_std']:.4f}")

    neut_df = pd.DataFrame(neut_results)
    best_neut_row = neut_df.loc[neut_df["corr_sharpe"].idxmax()]
    print(f"\n  >> Best neutralization: {best_neut_row['neutralization']} "
          f"(corr_sharpe={best_neut_row['corr_sharpe']:.4f})")

    # ── Experiment 2: Ensemble weight search ────────────────────────────────
    print("\n=== Experiment 2: Ensemble Weight Search ===")
    steps = [round(x * 0.1, 1) for x in range(11)]
    weight_results = []
    for lw, nw in product(steps, steps):
        tw = round(1.0 - lw - nw, 1)
        if tw < 0 or tw > 1:
            continue
        scores = blend_and_score(val, raw, feature_cols, lw, nw, tw,
                                  best_neut_row["neutralization"])
        weight_results.append({"lgbm_w": lw, "nn_w": nw, "tran_w": tw, **scores})

    weight_df = pd.DataFrame(weight_results).sort_values("corr_sharpe", ascending=False)
    best_w = weight_df.iloc[0]
    print(f"\n  Top 5 weight combos by CORR Sharpe:")
    print(weight_df[["lgbm_w", "nn_w", "tran_w", "corr_sharpe", "mmc_sharpe"]].head(5).to_string(index=False))
    print(f"\n  >> Best weights: lgbm={best_w['lgbm_w']} nn={best_w['nn_w']} tran={best_w['tran_w']} "
          f"(corr_sharpe={best_w['corr_sharpe']:.4f})")

    # ── Save results ─────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    neut_df.to_csv(REPORT_DIR / "td_research_neutralization.csv", index=False)
    weight_df.to_csv(REPORT_DIR / "td_research_weights.csv", index=False)

    print(f"\n=== Summary ===")
    print(f"Baseline  — corr_sharpe={BASELINE['neutralization']} neut, weights {BASELINE['lgbm']}/{BASELINE['nn']}/{BASELINE['tran']}")
    baseline_scores = neut_df[neut_df["neutralization"] == BASELINE["neutralization"]].iloc[0]
    print(f"           corr_sharpe={baseline_scores['corr_sharpe']:.4f}")
    print(f"Best      — neut={best_neut_row['neutralization']}, weights lgbm={best_w['lgbm_w']} nn={best_w['nn_w']} tran={best_w['tran_w']}")
    print(f"           corr_sharpe={best_w['corr_sharpe']:.4f}")
    improvement = best_w['corr_sharpe'] - baseline_scores['corr_sharpe']
    print(f"Improvement: {improvement:+.4f}")
    print(f"\nResults saved to {REPORT_DIR}/")
    print(f"\nTo apply best settings, run:")
    print(f"  python local/06_predict_submit.py --weights {best_w['lgbm_w']} {best_w['nn_w']} {best_w['tran_w']}")


if __name__ == "__main__":
    main()
