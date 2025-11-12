# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests.  To run integration tests you must pass the integration tag.  `go test -tags=integration ./...` Some tests require additional tags to enable to allow scoped testing to keep the duration reasonable.  For example, testing a broad set of models requires `-tags=integration,models` and a longer timeout (~60m or more depending on the speed of your GPU.). To view the current set of tag combinations use `find integration -type f | xargs grep "go:build"`


The integration tests have 2 modes of operating.

1. By default, on Unix systems, they will start the server on a random port, run the tests, and then shutdown the server.  On Windows you must ALWAYS run the server on OLLAMA_HOST for the tests to work.
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote based on your `OLLAMA_HOST` environment variable

> [!IMPORTANT]
> Before running the tests locally without the "test existing" setting, compile ollama from the top of the source tree  `go build .` in addition to GPU support with cmake if applicable on your platform.  The integration tests expect to find an ollama binary at the top of the tree.


Many tests use a default small model suitable to run on many systems.  You can override this default model by setting `OLLAMA_TEST_DEFAULT_MODEL`

## Tool Calling Tests

The tool calling tests are split into two files:

- **`tools_test.go`** - Tests using the native Ollama API (`api.Tool`)
- **`tools_openai_test.go`** - Tests using the OpenAI-compatible API format

### Running Tool Calling Tests

Run all tool calling tests:
```bash
go test -tags=integration -v -run Test.*Tool.* ./integration
```

Run only OpenAI-compatible tests:
```bash
go test -tags=integration -v -run TestOpenAI ./integration
```

Run only native API tests:
```bash
go test -tags=integration -v -run TestAPIToolCalling ./integration
```

### Parallel Execution

The OpenAI-compatible tests (`tools_openai_test.go`) support parallel execution for cloud models. Run with parallel execution:
```bash
go test -tags=integration -v -run TestOpenAI -parallel 3 ./integration
```

Cloud models (models ending with `-cloud`) will run in parallel, while local models run sequentially. This significantly speeds up test execution when testing against external endpoints.

### Testing Specific Models

To test a specific model, set the `OPENAI_TEST_MODELS` environment variable:
```bash
OPENAI_TEST_MODELS="gpt-oss:120b-cloud" go test -tags=integration -v -run TestOpenAI ./integration
```

### External Endpoints

To test against an external OpenAI-compatible endpoint (e.g., Ollama Cloud):
```bash
OPENAI_BASE_URL="https://ollama.com/v1" OLLAMA_API_KEY="your-key" go test -tags=integration -v -run TestOpenAI ./integration
```

### Environment Variables

The tool calling tests support the following environment variables:

- **`OPENAI_BASE_URL`** - When set, tests will run against an external OpenAI-compatible endpoint instead of a local server. If set, `OLLAMA_API_KEY` must also be provided.
- **`OLLAMA_API_KEY`** - API key for authenticating with external endpoints (required when `OPENAI_BASE_URL` is set).
- **`OPENAI_TEST_MODELS`** - Override the default model list and test only the specified model(s). Can be a single model or comma-separated list.