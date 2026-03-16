# MCP Setup Mini-Agent

Your goal is to configure the Numerai MCP server for Claude Code.

## Instructions
1. Read the `../.env` file (one level up from the `agent/` folder) to get the `NUMERAI_MCP_AUTH` variable.
   - The user's `.env` should contain: `NUMERAI_MCP_AUTH="Token PUBLIC_KEY\$PRIVATE_KEY"`
2. If it is empty or missing, pause and ask the user to fill it in the `.env` file first using their generated MCP key.
3. Once populated, export or set that environment variable for the current terminal session so that Claude Code can use it.
   - Example for Windows PowerShell: `$env:NUMERAI_MCP_AUTH="Token PUBLIC_KEY`$PRIVATE_KEY"` (Note the PowerShell backtick escape for the $).
   - Example for bash: `export NUMERAI_MCP_AUTH="Token PUBLIC_KEY\$PRIVATE_KEY"`
4. Run the MCP add command (Do not hardcode the key in the command line if possible; use the environment variable):
   ```bash
   claude mcp add --transport http numerai https://api-tournament.numer.ai/mcp --header "Authorization: ${env:NUMERAI_MCP_AUTH}"
   ```
5. Verify the connection by using the `check_api_credentials` tool.
6. Once verified, inform the user that MCP is successfully configured and they are ready to upload.
