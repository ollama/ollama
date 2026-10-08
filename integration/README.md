# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests. To run integration tests, pass the `integration` tag and one of the scoped tags:

```bash
go test -tags=integration,fast -v -count 1 ./integration/
go test -tags=integration,release -v -count 1 -timeout 30m ./integration/
go test -tags=integration,library -v -count 1 -timeout 120m ./integration/
```

Tags:

- `fast`: quick runner/model smoke coverage.
- `release`: release regression coverage.
- `library`: broad library coverage requiring about 2.5 TiB of disk space.

Scope wiring and model selections live in `integration/reg_fast_test.go`, `integration/reg_release_test.go`, and `integration/reg_library_test.go`.

The integration tests have 2 modes of operating.

1. By default, on Unix systems, they will start the server on a random port, run the tests, and then shutdown the server.  On Windows you must ALWAYS run the server on OLLAMA_HOST for the tests to work.
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote based on your `OLLAMA_HOST` environment variable

Set `OLLAMA_TEST_LOG_SERVER=1` to print the managed server log after each test
run, even when the tests pass. This only applies when the integration test
harness starts the server.

> [!IMPORTANT]
> Before running the tests locally without the "test existing" setting, compile ollama from the top of the source tree  `go build .` in addition to GPU support with cmake if applicable on your platform.  The integration tests expect to find an ollama binary at the top of the tree.


## Testing a New Model

When implementing a new model architecture, use `OLLAMA_TEST_MODEL` to run the
integration suite against the created Ollama model tag with either the `fast` or
`release` coverage. This should be final validation after the model-specific
unit tests, reference activation tests, cache tests, and quality checks pass.
Integration tests are much slower and should not be used as a substitute for
focused coverage.

```bash
# Build the binary first
go build .

# Run integration tests against it
OLLAMA_TEST_MODEL=mymodel go test -tags=integration,fast -v -count 1 ./integration/
```

To run against an existing server instead of letting the test harness start and
stop one:

```bash
OLLAMA_TEST_EXISTING=1 \
OLLAMA_HOST=http://127.0.0.1:11434 \
OLLAMA_TEST_MODEL=mymodel:base-mlx-bf16 \
  go test -tags=integration -v -count=1 -timeout 30m ./integration
```

The override relies on model capabilities. Tests for audio, vision, embeddings,
tool calling, and thinking call `/api/show` and skip when the model does not
advertise the required capability. Make sure the model manifest/config exposes
only the capabilities the model actually supports:

- `completion` for normal text generation
- `vision` for image input
- `audio` for audio input
- `tools` for tool calling
- `thinking` for thinking output
- `embedding` for embedding models

If a completion-only model runs audio, vision, or tool tests, fix the model's
advertised capabilities. If a model supports one of those features but the
related tests skip, fix the capability metadata before treating the integration
run as complete.
