# Add /api/tokenize and /api/detokenize endpoints

## Problem Statement

Token counting, prompt chunking, and model-aligned behavior require access to a model's vocabulary without loading the full model into memory. Currently, users must either:
1. Load the entire model (expensive VRAM allocation)
2. Use external tokenizers (not model-aligned)
3. Implement custom solutions (maintenance burden)

## What This Changes

- **New endpoints**: `POST /api/tokenize` and `POST /api/detokenize`
- **Text-only scope**: Multimodal support reserved for future versions
- **Clean schema**: No `keep_alive` or `media_type` in public API
- **Vocab-only optimization**: Designed for lightweight vocabulary loading

## Design Notes

### Vocab-Only Target
- **Primary path**: Load only vocabulary (no context weights/VRAM)
- **Fallback**: Use existing scheduler infrastructure for compatibility
- **Future**: Wire to llama.cpp vocab-only API once available

### Implementation Details
- **LRU cache**: 8-model capacity with automatic eviction
- **Concurrency-safe**: Mutex-protected operations
- **Debug logging**: `OLLAMA_TOKENIZER_DEBUG=1` for observability
- **Sentinel errors**: Clear fallback detection for monitoring

### Architecture
```
server/routes.go → tokenizerloader.Get() → vocab-only OR fallback
                                    ↓
                              scheduler-backed tokenization
```

## Performance Benchmarks

| Model            | Mode | Path                 | P50 latency |
|------------------|------|----------------------|-------------|
| mistral:latest   | cold | fallback scheduler   | ~3.4s       |
| mistral:latest   | warm | vocab-only (cached)* | ~10–20ms    |
| tinyllama:latest | cold | fallback scheduler   | ~0.9s       |
| tinyllama:latest | warm | vocab-only (cached)* | ~10–20ms    |

*Currently fallback path; numbers reflect warm process + cache.

## Testing Strategy

- **Unit tests**: LRU eviction, concurrency safety, debug flag execution
- **Race detection**: `go test -race` passes
- **Integration**: End-to-end curl testing with small models
- **Cross-platform**: Linux, Darwin, Windows builds verified

## Compatibility

- **No breaking changes**: New endpoints only
- **Existing API**: Unchanged behavior for current endpoints
- **Fallback support**: Graceful degradation when vocab-only unavailable

## Follow-ups

### Immediate
- [ ] Wire `openVocabOnly()` to llama.cpp vocab-only API (PR #8106)
- [ ] Implement `Tokenize()`/`Detokenize()` on `vocabOnlyModel`

### SDK Integration
- [ ] ollama-python: [tracking doc](upstream-links/tokenize-python.md)
- [ ] ollama-js: [tracking doc](upstream-links/tokenize-js.md)

## Maintainer Alignment

- ✅ **No keep_alive/media_type**: Clean public API surface
- ✅ **Text-only scope**: Multimodal reserved for future
- ✅ **Internal optimization**: Vocab-only path when available
- ✅ **Fallback compatibility**: Existing infrastructure preserved
- ✅ **Minimal surface**: Focused, maintainable endpoints

## Files Changed

- `server/routes.go` - New handlers and request/response structs
- `server/tokenizerloader/loader.go` - Vocab-only loader with LRU cache
- `server/tokenizerloader/loader_test.go` - Unit tests
- `docs/api.md` - API documentation with examples
- `api/examples/tokenize/bench.md` - Performance benchmarks
- `upstream-links/tokenize-python.md` - Python SDK tracking
- `upstream-links/tokenize-js.md` - JavaScript SDK tracking

## Example Usage

```bash
# Tokenize text
curl http://localhost:11434/api/tokenize -d '{
  "model": "mistral:latest",
  "content": "Hello world"
}'

# Detokenize tokens
curl http://localhost:11434/api/detokenize -d '{
  "model": "mistral:latest",
  "tokens": [2050, 1187]
}'
```
