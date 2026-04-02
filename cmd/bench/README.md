Ollama Benchmark Tool
---------------------

A Go-based command-line tool for benchmarking Ollama models with configurable parameters, warmup phases, TTFT tracking, VRAM monitoring, and benchstat/CSV output.

## Features

 * Benchmark multiple models in a single run
 * Support for both text and image prompts
 * Configurable generation parameters (temperature, max tokens, seed, etc.)
 * Warmup phase before timed epochs to stabilize measurements
 * Time-to-first-token (TTFT) tracking per epoch
 * Model metadata display (parameter size, quantization level, family)
 * VRAM and CPU memory usage tracking via running process info
 * Controlled prompt token length for reproducible benchmarks
 * Benchstat and CSV output formats

## Building from Source

```
go build -o ollama-bench ./cmd/bench
./ollama-bench -model gemma3 -epochs 6 -format csv
```

Using Go Run (without building)

```
go run ./cmd/bench -model gemma3 -epochs 3
```

## Usage

### Basic Example

```
./ollama-bench -model gemma3 -epochs 6
```

### Benchmark Multiple Models

```
./ollama-bench -model gemma3,gemma3n -epochs 6 -max-tokens 100 -p "Write me a short story" | tee gemma.bench
benchstat -col /name gemma.bench
```

### With Image Prompt

```
./ollama-bench -model qwen3-vl -image photo.jpg -epochs 6 -max-tokens 100 -p "Describe this image"
```

### Controlled Prompt Length

```
./ollama-bench -model gemma3 -epochs 6 -prompt-tokens 512
```

### Advanced Example

```
./ollama-bench -model llama3 -epochs 10 -temperature 0.7 -max-tokens 500 -seed 42 -warmup 2 -format csv -output results.csv
```

## Command Line Options

| Option  	| Description | Default |
|----------|-------------|---------|
| -model	| Comma-separated list of models to benchmark	| (required)		|
| -epochs	| Number of iterations per model		| 6			|
| -max-tokens	| Maximum tokens for model response		| 200			|
| -temperature	| Temperature parameter				| 0.0			|
| -seed		| Random seed					| 0 (random)		|
| -timeout	| Timeout in seconds				| 300			|
| -p		| Prompt text					| (default story prompt)	|
| -image	| Image file to include in prompt		| 			|
| -k		| Keep-alive duration in seconds		| 0			|
| -format	| Output format (benchstat, csv)		| benchstat		|
| -output	| Output file for results			| "" (stdout)		|
| -warmup	| Number of warmup requests before timing	| 1			|
| -prompt-tokens	| Generate prompt targeting ~N tokens (0 = use -p)	| 0		|
| -v		| Verbose mode					| false			|
| -debug	| Show debug information			| false			|

## Output Formats

### Benchstat Format (default)

Compatible with Go's benchstat tool for statistical analysis. Uses one value/unit pair per line, standard `ns/op` for timing metrics, and `ns/token` for throughput. Each epoch produces one set of lines -- benchstat aggregates across repeated runs to compute statistics.

```
# Model: gemma3 | Params: 4.3B | Quant: Q4_K_M | Family: gemma3 | Size: 4080218931 | VRAM: 4080218931
BenchmarkModel/name=gemma3/step=prefill 1 78125.00 ns/token 12800.00 token/sec
BenchmarkModel/name=gemma3/step=generate 1 19531.25 ns/token 51200.00 token/sec
BenchmarkModel/name=gemma3/step=ttft 1 45123000 ns/op
BenchmarkModel/name=gemma3/step=load 1 1500000000 ns/op
BenchmarkModel/name=gemma3/step=total 1 2861047625 ns/op
```

Use with benchstat:
```
./ollama-bench -model gemma3 -epochs 6 > gemma3.bench
benchstat -col /step gemma3.bench
```

Compare two runs:
```
./ollama-bench -model gemma3 -epochs 6 > before.bench
# ... make changes ...
./ollama-bench -model gemma3 -epochs 6 > after.bench
benchstat before.bench after.bench
```

### CSV Format

Machine-readable comma-separated values:

```
NAME,STEP,COUNT,NS_PER_COUNT,TOKEN_PER_SEC
# Model: gemma3 | Params: 4.3B | Quant: Q4_K_M | Family: gemma3 | Size: 4080218931 | VRAM: 4080218931
gemma3,prefill,128,78125.00,12800.00
gemma3,generate,512,19531.25,51200.00
gemma3,ttft,1,45123000,0
gemma3,load,1,1500000000,0
gemma3,total,1,2861047625,0
```

## Metrics Explained

The tool reports the following metrics for each epoch:

 * **prefill**: Time spent processing the prompt (ns/token)
 * **generate**: Time spent generating the response (ns/token)
 * **ttft**: Time to first token -- latency from request start to first response content
 * **load**: Model loading time (one-time cost)
 * **total**: Total request duration

Additionally, the model info comment line (displayed once per model before epochs) includes:

 * **Params**: Model parameter count (e.g., 4.3B)
 * **Quant**: Quantization level (e.g., Q4_K_M)
 * **Family**: Model family (e.g., gemma3)
 * **Size**: Total model memory in bytes
 * **VRAM**: GPU memory used by the loaded model (when Size > VRAM, the difference is CPU spill)
