# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests.  To run integration tests you must pass the integration tag.  `go test -tags=integration ./...`


The integration tests have 2 modes of operating.

1. By default, they will start the server on a random port, run the tests, and then shutdown the server.
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote
