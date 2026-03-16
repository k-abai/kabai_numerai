# Check Status Mini-Agent

Your goal is to verify the status of the user's models and tournament standing using the MCP server.

## Instructions
1. Use the `get_model_profile` and `get_model_performance` tools.
2. If the user doesn't specify a model, use GraphQL or `get_leaderboard` (filtered by the user's models if possible) to list the user's active models.
3. Report the current round, validation status of recent uploads, and the core metrics (`corr20Rep`, `mmc20Rep`, `return13Weeks`, `nmrStaked`).
4. Format the output in a clean, legible markdown table.
