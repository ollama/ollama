# Benchmark

Performance benchmarking for Ollama.

## Prerequisites
- Ollama server running locally (`127.0.0.1:11434`)
- benchstat tool: `go install golang.org/x/perf/cmd/benchstat@latest`

## Run Benchmark
```bash
# Run benchmark and save results
go test -bench=. -m $MODEL_NAME ./... > bench1.txt

# Run again to compare (e.g. with different model or after code changes)
go test -bench=. -m $MODEL_NAME ./... > bench2.txt

# Compare results
benchstat bench1.txt bench2.txt
```
