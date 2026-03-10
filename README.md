# Numerai Model Training Project

Self-contained project for training and submitting models to Numerai.

## Project Structure
- `colab/`: Scripts optimized for Google Colab (TPU/GPU support).
- `local/`: Scripts optimized for local CLI with CUDA GPU acceleration.
- `docs/`: Technical documentation and architecture diagrams.

## Quick Start (Local)
1. Install requirements:
   ```bash
   pip install -r local/requirements_local.txt
   ```
2. Download data:
   ```bash
   python local/01_download.py --size medium
   ```
3. Train models:
   ```bash
   python local/03_0_train_lgbm.py --size medium
   python local/03_1_train_nn.py --size medium
   python local/03_2_train_transformer.py --size medium
   ```

## Development
This project was migrated from a playground environment and follows modular design principles for easy experimentation.
