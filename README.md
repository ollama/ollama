<div align="center">
  <h1>ollama-vocab-tokenizer</h1>
  <p>
    Extended Ollama fork with direct <code>/api/tokenize</code> and <code>/api/detokenize</code> endpoints.<br/>
    Provides fast, model-aligned tokenization and detokenization over HTTP.
  </p>
</div>

---


> [!WARNING]  
> **Experimental API:** *The /api/tokenize and /api/detokenize endpoints are not part of upstream Ollama’s stable API.*  
> **Compatibility risk:** *They may break with future Ollama releases, since internal tokenizer APIs can change without notice.*  
> **Performance trade-offs:** *Cold-starts may still load full model runners; more vocab-only optimization is under active development.*  
> **No stability guarantees:** *This fork is intended for experimentation, benchmarking, and development.*  



## Overview

`ollama-vocab-tokenizer` extends [Ollama](https://github.com/ollama/ollama) with two new HTTP API endpoints:

- **`POST /api/tokenize`** → Convert text into tokens
- **`POST /api/detokenize`** → Convert tokens back into text

These endpoints expose model-aligned tokenization without requiring a full model generation call, enabling:

- Fast text preprocessing for downstream tasks
- Round-trip validation of tokenizer behavior
- Observability into tokenization latency
- Vocabulary-only optimization with LRU caching
- Scheduler-backed fallback for compatibility

---

## Installation

Follow the standard Ollama installation, then build this fork:

```bash
git clone https://github.com/icedmoca/ollama-vocab-tokenizer.git
cd ollama-vocab-tokenizer
go build -o ollama .
```

Run the server:

```bash
./ollama serve
```

---

## Usage

### Tokenize text

```bash
curl -s http://localhost:11434/api/tokenize   -H "Content-Type: application/json"   -d '{
    "model": "mistral:latest",
    "content": "Hello world"
  }'
```

Response:

```json
{
  "model": "mistral:latest",
  "tokens": [7080, 29477],
  "total_duration": 1234567,
  "load_duration": 456789
}
```

### Detokenize tokens

```bash
curl -s http://localhost:11434/api/detokenize   -H "Content-Type: application/json"   -d '{
    "model": "mistral:latest",
    "tokens": [7080, 29477]
  }'
```

Response:

```json
{
  "model": "mistral:latest",
  "content": "Hello world",
  "total_duration": 123456,
  "load_duration": 45678
}
```

### Round-trip test

```bash
TOKENS=$(curl -s http://localhost:11434/api/tokenize   -H "Content-Type: application/json"   -d '{"model": "mistral:latest", "content": "Hello world"}' | jq -r '.tokens')

curl -s http://localhost:11434/api/detokenize   -H "Content-Type: application/json"   -d "{"model": "mistral:latest", "tokens": $TOKENS}" | jq -r '.content'
```

---

## Benchmarks

| Model            | Mode | Path                 | P50 latency |
|------------------|------|----------------------|-------------|
| mistral:latest   | cold | fallback scheduler   | ~3.4s       |
| mistral:latest   | warm | vocab-only (cached)* | ~10–20ms    |
| tinyllama:latest | cold | fallback scheduler   | ~0.9s       |
| tinyllama:latest | warm | vocab-only (cached)* | ~10–20ms    |

\*Currently fallback path; numbers reflect warm process + cache.

---

## Debugging

Enable verbose tokenizer logs:

```bash
export OLLAMA_TOKENIZER_DEBUG=1
./ollama serve
```
