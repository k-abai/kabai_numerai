import cloudpickle
import numpy as np

def predict_LNT(live_features):
    lgbm_preds = pd.DataFrame(index=live_features.index)
    for target in TARGET_CANDIDATES:
        lgbm_preds[target] = models[target].predict(live_features[feature_cols])
    lgbm_ranked = lgbm_preds.rank(pct=True).mean(axis=1).rank(pct=True)
    nn_ranked = pd.Series(final_model.predict(live_features[feature_cols].values.astype(np.float32)).flatten(), index=live_features.index).rank(pct=True)
    tf_ranked = pd.Series(transformer_model.predict(live_features[feature_cols].values.astype(np.float32)).flatten(), index=live_features.index).rank(pct=True)
    combined = lgbm_ranked * 0.5 + nn_ranked * 0.3 + tf_ranked * 0.2
    return combined.rank(pct=True, method="first").to_frame("prediction")

with open("LNT_ensemble.pkl", "wb") as f:
    f.write(cloudpickle.dumps(predict_LNT))
