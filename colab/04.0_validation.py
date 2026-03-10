# Validation
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era", "data_type"] + feature_cols + target_cols)
validation["target"] = validation[MAIN_TARGET]
validation = validation[validation["data_type"] == "validation"]

for target in TARGET_CANDIDATES:
    validation[f"prediction_{target}"] = models[target].predict(validation[feature_cols])

validation["prediction"] = (
    validation.groupby("era")[[f"prediction_{t}" for t in TARGET_CANDIDATES]].rank(pct=True).mean(axis=1)
)

from numerai_tools.scoring import numerai_corr, correlation_contribution
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(f"v4.3/meta_model.parquet")["numerai_meta_model"]

per_era_corr = validation.groupby("era").apply(lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna()))
per_era_mmc = validation.dropna().groupby("era").apply(lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"]))

summary = pd.DataFrame({
    "mean": [per_era_corr.mean(), per_era_mmc.mean()],
    "sharpe": [per_era_corr.mean()/per_era_corr.std(), per_era_mmc.mean()/per_era_mmc.std()]
}, index=["CORR", "MMC"])
print(summary)
