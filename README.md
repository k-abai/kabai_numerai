# Numerai Model Training Project

Self-contained project for training and submitting models to Numerai. Numerai's large tabular data and high model density makes creating a competitive model much more difficult then percieved. This gets even more difficult with limited compute. This module seeks to address the issue by utilizing an ensemble of known and more novel techniques optimized for local hosting, low compute enviroments, with scalable capabilties to generate orthogonal alpha and reduce risk.

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
   python local/03_2_train_transformer.py --size medium --memory low  
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

## MCP Agent Usage (Custom Scripted Agent)

This project utilizes a **Custom Scripted Agent** architecture designed for scalability and minimal overhead. Instead of configuring multiple distinct agents, we use a single Claude Code context that delegates tasks to specialized markdown-based "mini-agents".

### Why this approach?
- **Scalability**: Add new capabilities by simply dropping a new `.md` script in the `agent/scripts/` folder.
- **Unified Context**: No need to jump between multiple agent configurations.
- **Customization**: Easily tweak prompts for specific workflows (like daily uploads or status checks) without re-configuring the entire agent.

### Setup & Execution
1. **API Keys**: Configure your Numerai keys in the `.env` file at the project root.
   ```bash
   NUMERAI_MCP_AUTH="Token YOUR_PUBLIC_KEY\$YOUR_PRIVATE_KEY"
   ```
2. **Launch**: Navigate to `agent/` and start Claude Code.
3. **Command**: Initialize by telling Claude:
   > "Read `claude.md` and [your task, e.g., 'run MCP setup' or 'do daily upload']"

### Project Agent Structure
- `agent/claude.md`: The master orchestrator prompt.
- `agent/scripts/mcp_setup.md`: Automated MCP authentication and setup.
- `agent/scripts/daily_upload.md`: Step-by-step PKL upload workflow.
- `agent/scripts/check_status.md`: Performance and submission status queries.
