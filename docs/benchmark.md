# Benchmark

Go benchmark tests that measure end-to-end performance of a running Ollama server. Run these tests to evaluate model inference performance on your hardware and measure the impact of code changes.

## When to use

Run these benchmarks when:
- Making changes to the model inference engine
- Modifying model loading/unloading logic
- Changing prompt processing or token generation code
- Implementing a new model architecture
- Testing performance across different hardware setups

## Prerequisites
- Ollama server running locally with `ollama serve` on `127.0.0.1:11434`
## Usage and Examples

>[!NOTE]
>All commands must be run from the root directory of the Ollama project.

Basic syntax:
```bash
go test -bench=. ./benchmark/... -m $MODEL_NAME
```

Required flags:
- `-bench=.`: Run all benchmarks
- `-m`: Model name to benchmark

Optional flags:
- `-count N`: Number of times to run the benchmark (useful for statistical analysis)
- `-timeout T`: Maximum time for the benchmark to run (e.g. "10m" for 10 minutes)

Common usage patterns:

Single benchmark run with a model specified:
```bash
go test -bench=. ./benchmark/... -m llama3.3
```

## Output metrics

The benchmark reports several key metrics:

- `gen_tok/s`: Generated tokens per second
- `prompt_tok/s`: Prompt processing tokens per second
- `ttft_ms`: Time to first token in milliseconds
- `load_ms`: Model load time in milliseconds
- `gen_tokens`: Total tokens generated
- `prompt_tokens`: Total prompt tokens processed

Each benchmark runs two scenarios:
- Cold start: Model is loaded from disk for each test
- Warm start: Model is pre-loaded in memory

Three prompt lengths are tested for each scenario:
- Short prompt (100 tokens)
- Medium prompt (500 tokens)
- Long prompt (1000 tokens)
