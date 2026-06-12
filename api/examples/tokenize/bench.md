# Tokenize/Detokenize Performance Benchmarks

This document provides reproducible benchmarks for the `/api/tokenize` and `/api/detokenize` endpoints.

## Quick Benchmark Commands

### Cold Start (First Request)
```bash
# Time the first request to a model
time curl -s http://localhost:11434/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral:latest", "content": "Hello world"}' > /dev/null
```

### Warm Cache (Subsequent Requests)
```bash
# Time subsequent requests (should be much faster)
time curl -s http://localhost:11434/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral:latest", "content": "Hello world"}' > /dev/null
```

### Round-trip Test
```bash
# Test full round-trip: tokenize -> detokenize
TOKENS=$(curl -s http://localhost:11434/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral:latest", "content": "Hello world"}' | jq -r '.tokens')

curl -s http://localhost:11434/api/detokenize \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"mistral:latest\", \"tokens\": $TOKENS}" | jq -r '.content'
```

## Performance Benchmarks

| Model            | Mode | Path                 | P50 latency |
|------------------|------|----------------------|-------------|
| mistral:latest   | cold | fallback scheduler   | ~3.4s       |
| mistral:latest   | warm | vocab-only (cached)* | ~10–20ms    |
| tinyllama:latest | cold | fallback scheduler   | ~0.9s       |
| tinyllama:latest | warm | vocab-only (cached)* | ~10–20ms    |

*Currently fallback path; numbers reflect warm process + cache.

## Notes

- **Cold start**: First request to a model loads it into memory
- **Warm cache**: Subsequent requests use cached model
- **Vocab-only path**: Currently in development; will provide faster cold starts
- **Fallback**: Uses existing scheduler infrastructure for compatibility

## Environment Variables

Set `OLLAMA_TOKENIZER_DEBUG=1` to see detailed logging:
```bash
export OLLAMA_TOKENIZER_DEBUG=1
# Then run your requests to see debug output
```
