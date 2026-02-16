# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests.  To run integration tests you must pass the integration tag.  `go test -tags=integration ./...` Some tests require additional tags to enable to allow scoped testing to keep the duration reasonable.  For example, testing a broad set of models requires `-tags=integration,models` and a longer timeout (~60m or more depending on the speed of your GPU.). To view the current set of tag combinations use `find integration -type f | xargs grep "go:build"`


The integration tests have 2 modes of operating.

1. By default, on Unix systems, they will start the server on a random port, run the tests, and then shutdown the server.  On Windows you must ALWAYS run the server on OLLAMA_HOST for the tests to work.
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote based on your `OLLAMA_HOST` environment variable

> [!IMPORTANT]
> Before running the tests locally without the "test existing" setting, compile ollama from the top of the source tree  `go build .` in addition to GPU support with cmake if applicable on your platform.  The integration tests expect to find an ollama binary at the top of the tree.


Many tests use a default small model suitable to run on many systems.  You can override this default model by setting `OLLAMA_TEST_DEFAULT_MODEL`