# Numerai Model Training Project

Self-contained project for training and submitting models to Numerai.

## Project Structure
- `colab/`: Scripts optimized for Google Colab (TPU/GPU support).
- `local/`: Scripts optimized for local CLI with CUDA GPU acceleration (CPU/GPU support).
- `docs/`: Technical documentation and architecture diagrams.

## Quick Start (Local)
1. Install requirements:
   ```bash
   pip install -r local/requirements_local.txt
   ```
2. Download data:
   ```bash
   python local/01_download.py 
   ```
3. Train models:
   ```bash
   python local/03_0_train_lgbm.py --size medium
   python local/03_1_train_nn.py --size medium
   python local/03_2_train_transformer.py --size medium
   ```
4. Validate models:
   It is important to use the same size for validation used in training per model.
   ```bash
   python local/04_0validate_lgbm.py --size medium --memory low
   python local/04_1validate_nn.py --size medium --memory low
   python local/04_2validate_tran.py --size medium --memory low

   ```
## Explore Auxilery Targets
Use and modify the explore scipt to find correlations and explore auzilery targets to train on. Default is ender but may change.
   ``` bash
   python local/02_explore.py --size medium --main target_ender_20
   ```

## Development
This project was migrated from a playground environment and follows modular design principles for easy experimentation.
