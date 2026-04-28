# Outstanding Development Plans: Gemini Integration

## Current State
- The fragile Node.js proxy and `@google/gemini-cli` dependency have been removed.
- A native Go-based `Runner` implementation for Gemini has been created in `cmd/launch/gemini.go`.
- The registry in `cmd/launch/registry.go` needs to be updated to reflect the new native integration (removing installation checks for external binaries).

## Pending Tasks
1. **Finalize Registry Update**: Ensure `cmd/launch/registry.go` correctly marks the Gemini integration as "installed" by default since it no longer depends on an external binary.
2. **API Key Management**: Implement a more robust way to handle the `GEMINI_API_KEY`. Currently, it's a hard check on the environment variable. It should be integrated with Ollama's configuration system or support cloud-native token injection.
3. **Build & Test**: 
    - Build the `ollama` binary.
    - Verify `ollama launch gemini -- "prompt"` works end-to-end.
    - Validate streaming output from the `streamGenerateContent` endpoint.
4. **PR Cleanup**: The previous PR (which included the proxy) should be closed or completely overwritten with this native implementation to avoid merging "hacky" code into the main codebase.

## Technical Debt to Address
- **Error Handling**: Improve error messages when the API returns 401 (Unauthorized) or 429 (Too Many Requests).
- **Model Selection**: Allow users to specify the Gemini model (e.g., `gemini-1.5-pro` vs `gemini-1.5-flash`) via `ollama launch` arguments.
