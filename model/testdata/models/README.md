# Test Model Directory

This directory is used for storing model files (like `.gguf` files) that are required to run the tests in `model_external_test.go`. 

## Usage

- Place any model files you need for testing in this directory
- The test file will look for any model files here (e.g., `llama3.gguf`)
- All non-markdown files in this directory are git-ignored to prevent large model files from being committed to the repository
- Only `.md` files (like this README) will be tracked in git
