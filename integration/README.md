# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests.  To run integration tests you must pass the integration tag.  `go test -tags=integration ./...`


The integration tests have 2 modes of operating.

1. By default, they will start the server on a random port, run the tests, and then shutdown the server.  Make sure you build locally before running
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote

Some integration tests, particularly scheduler stress scenarios, need to know how much VRAM is available, and will skip if you don't set `OLLAMA_MAX_VRAM` to the size of the GPU in bytes.

By combining `OLLAMA_HOST` and `OLLAMA_TEST_EXISTING` you can test against remote systems.  For example:

```
OLLAMA_HOST=remotehost:11434 OLLAMA_TEST_EXISTING=1 go test -tags integration -v ./integration
```