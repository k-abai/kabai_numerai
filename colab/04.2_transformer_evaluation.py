# Transformer & Multi-Model Evaluation
validation["prediction_transformer"] = transformer_model.predict(validation[feature_cols].values.astype(np.float32)).flatten()
validation["LNT_Ensemble"] = (
    validation["prediction"].rank(pct=True) * 0.5 +
    validation["prediction_nn"].rank(pct=True) * 0.3 +
    validation["prediction_transformer"].rank(pct=True) * 0.2
)
# ...
