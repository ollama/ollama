Ollama Benchmark Tool
---------------------

A Go-based command-line tool for benchmarking Ollama models with configurable parameters and multiple output formats.

## Features

 * Benchmark multiple models in a single run
 * Support for both text and image prompts
 * Configurable generation parameters (temperature, max tokens, seed, etc.)
 * Supports benchstat and CSV output formats
 * Detailed performance metrics (prefill, generate, load, total durations)

## Building from Source

```
go build -o ollama-bench bench.go
./ollama-bench -model gpt-oss:20b -epochs 6 -format csv
```

Using Go Run (without building)

```
go run bench.go -model gpt-oss:20b -epochs 3
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

### Advanced Example

```
./ollama-bench -model llama3 -epochs 10 -temperature 0.7 -max-tokens 500 -seed 42 -format csv -output results.csv
```

## Command Line Options

| Option  	| Description | Default |
|----------|-------------|---------|
| -model	| Comma-separated list of models to benchmark	| (required)		|
| -epochs	| Number of iterations per model		| 1			|
| -max-tokens	| Maximum tokens for model response		| 0 (unlimited)		|
| -temperature	| Temperature parameter				| 0.0			|
| -seed		| Random seed					| 0 (random)		|
| -timeout	| Timeout in seconds				| 300			|
| -p		| Prompt text					| "Write a long story."	|
| -image	| Image file to include in prompt		| 			|
| -k		| Keep-alive duration in seconds		| 0			|
| -format	| Output format (benchstat, csv)		| benchstat		|
| -output	| Output file for results			| "" (stdout)		|
| -v		| Verbose mode					| false			|
| -debug	| Show debug information			| false			|

## Output Formats

### Markdown Format

The default markdown format is suitable for copying and pasting into a GitHub issue and will look like:
```
 Model | Step | Count | Duration | nsPerToken | tokensPerSec |
|-------|------|-------|----------|------------|--------------|
| gpt-oss:20b | prefill | 124 | 30.006458ms | 241987.56 | 4132.44 |
| gpt-oss:20b | generate | 200 | 2.646843954s | 13234219.77 | 75.56 |
| gpt-oss:20b | load | 1 | 121.674208ms | - | - |
| gpt-oss:20b | total | 1 | 2.861047625s | - | - |
```

### Benchstat Format

Compatible with Go's benchstat tool for statistical analysis:

```
BenchmarkModel/name=gpt-oss:20b/step=prefill 128 78125.00 ns/token 12800.00 token/sec
BenchmarkModel/name=gpt-oss:20b/step=generate 512 19531.25 ns/token 51200.00 token/sec
BenchmarkModel/name=gpt-oss:20b/step=load 1 1500000000 ns/request
```

### CSV Format

Machine-readable comma-separated values:

```
NAME,STEP,COUNT,NS_PER_COUNT,TOKEN_PER_SEC
gpt-oss:20b,prefill,128,78125.00,12800.00
gpt-oss:20b,generate,512,19531.25,51200.00
gpt-oss:20b,load,1,1500000000,0
```

## Metrics Explained

The tool reports four types of metrics for each model:

 * prefill: Time spent processing the prompt
 * generate: Time spent generating the response
 * load: Model loading time (one-time cost)
 * total: Total request duration

