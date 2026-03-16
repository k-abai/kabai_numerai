# Daily Upload Mini-Agent

Your goal is to upload the latest target ensemble model to the Numerai tournament using the MCP server.

## Instructions
1. Confirm the model file exists at `local/models/numerai_upload2.pkl` (relative to the project root).
2. The model slot name needs to be identified. Ask the user which model slot this upload belongs to, or use a default if it was previously specified in the chat.
3. Use the MCP tool `upload_model` with the PKL upload workflow:
   - Step 1: `upload_model(operation="get_upload_auth")` (Provide tournament = 8 and the model slot id if needed).
   - Step 2: Extract the presigned S3 URL from the response of Step 1.
   - Step 3: Automatically craft and execute a `curl` or `Invoke-RestMethod` command to PUT the `local/models/numerai_upload2.pkl` file to the presigned URL.
   - Step 4: Call `upload_model(operation="create")` providing the necessary model metadata to register the upload.
   - Step 5: Wait momentarily, then call `upload_model(operation="list")`. Loop until the validationStatus indicates success.
   - Step 6: Finally, call `upload_model(operation="assign")` to assign the validated upload to the model slot.
4. Conclude by telling the user the upload was successful and proactively displaying the validation metrics of the model.
