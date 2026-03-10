# Modeling
import lightgbm as lgb

models = {}
for target in TARGET_CANDIDATES:
    print(f"Training model for {target}...")
    model = lgb.LGBMRegressor(
        n_estimators=30_000,
        learning_rate=0.001,
        max_depth=10,
        num_leaves=2**10,
        colsample_bytree=0.1,
        min_data_in_leaf=10000,
        device="gpu" 
    )
    
    model.fit(
      train[feature_cols],
      train[target]
    )
    models[target] = model
