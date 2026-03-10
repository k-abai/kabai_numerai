# NN & Ensemble Evaluation
correlations = validation.groupby("era").apply(lambda d: numerai_corr(d[[f"prediction_lgbm_{t}" for t in TARGET_CANDIDATES] + ["prediction_nn"]], d["target"]))
# ... (simplified metrics calculation)
print(correlations.mean())
