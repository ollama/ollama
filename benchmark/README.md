# Benchmark

Performance benchmarking for Ollama.

## Prerequisites
- Ollama server running locally (`127.0.0.1:11434`)
- Desired models pre-downloaded (e.g., `llama3.2:1b`)

## Run Benchmark
```bash
# Run all tests
go test -bench=. -timeout 30m ./...
```

## New Runner Benchmark
```bash
go test -bench=Runner
```

or to test multiple models:
```bash
# run this from within the benchmark directory
# requires: llama3.2:1b, llama3.1:8b, llama3.3:70b
sh new_runner.sh
```
