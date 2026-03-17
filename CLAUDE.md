# Numerai — Claude Instructions

## Submit Predictions

### Prerequisites
Credentials are in `.env` (PUBLIC_ID and SECRET_KEY from the `NUMERAI_MCP_AUTH` token).
Pass them inline or export to env before running.

### Run
```powershell
cd C:\Users\kekea\Downloads\numerai
venv\Scripts\python local/06_predict_submit.py --model_id <MODEL_UUID>
```

### What it does
1. Downloads current round live data
2. Loads `local/models/lgbm_models.pkl`, `nn_model.keras`, `transformer_model.keras`
3. Generates ensemble predictions (default: LGBM 50% + NN 25% + Transformer 25%)
4. Saves CSV to `local/models/legomax_predictions.csv`
5. Submits to the specified model on Numerai

### Arguments
| Arg | Default | Description |
|-----|---------|-------------|
| `--model_id` | legomax UUID | **Required for other models** |
| `--size` | `medium` | Feature set: small / medium / all |
| `--target` | `target` | LGBM target(s), comma-separated |
| `--weights` | `0.5 0.25 0.25` | Weights for [lgbm, nn, transformer] |
| `--out` | `local/models/legomax_predictions.csv` | Output CSV path |

## Project Structure
```
local/
  01_download.py              # Download training/validation data
  02_explore.py               # EDA
  03_0_train_lgbm.py          # Train LightGBM
  03_1_train_nn.py            # Train Neural Network
  03_2_train_transformer.py   # Train Transformer
  04_*/validate_*.py          # Validation scripts
  05_submit.py                # Build cloudpickle for Numerai Compute
  06_predict_submit.py        # Generate predictions and submit CSV
  config.json                 # DATA_VERSION and feature config
  models/
    lgbm_models.pkl           # Trained LGBM
    nn_model.keras            # Trained NN
    transformer_model.keras   # Trained Transformer
  model_defs/
    transformer_layers.py     # Custom Keras layers (FeatureEmbedding, TransformerEncoderBlock)
```
