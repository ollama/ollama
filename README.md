
# ollama-vocab-tokenizer

---
> Added tokenization endpoints that let clients convert text to tokens (and back) using a model’s vocabulary without loading the full model into GPU/VRAM. This enables fast, model aligned utilities (token counting, prompt splitting, input validation) while avoiding inference runner overhead. Targeted a vocab only loading path with a small in process LRU cache; when vocab only isn’t available, the system gracefully falls back to the scheduler backed path.
>
> **`Ollama Fork`** ***+*** **`Tokenizer API`** [https://github.com/ollama/ollama/pull/12030#issuecomment-4106961583](https://github.com/ollama/ollama/pull/12030#issuecomment-4106961583)
>
---
> ### [https://deepwiki.com/icedmoca/ollama-vocab-tokenizer ](https://deepwiki.com/icedmoca/ollama-vocab-tokenizer)
>
---
> [!WARNING]  
> **Experimental API:** *The /api/tokenize and /api/detokenize endpoints are not part of upstream Ollama’s stable API.*  
> **Compatibility risk:** *They may break with future Ollama releases, since internal tokenizer APIs can change without notice.*  
> **Performance trade-offs:** *Cold-starts may still load full model runners; more vocab-only optimization is under active development.*  
> **No stability guarantees:** *This fork is intended for experimentation, benchmarking, and development.*  

---

### [Tokenize / Detokenize: vocab only endpoints with cache + fallback](https://github.com/icedmoca/ollama/commit/48806d3e196a44fd0452eb09354fd9cf1a82b921)


### Goals
- Provide model-aligned tokenization and detokenization over HTTP
- Prefer a minimal, fast, vocab-only path; avoid VRAM/context weights where possible
- Keep public API text-only (no multimodal complexity), with a clean schema
- Avoid persistent model lifetimes per request (no keep_alive)
- Make behavior observable (debug logs, durations), predictable, and easy to wire into upstream llama.cpp vocab-only support

### Public API
- POST `/api/tokenize`
  - Request: `{ "model": "<model>" , "content": "<text>", "options": { … } }`
  - Response: `{ "model": "<model>", "tokens": [<int>...], "total_duration": <ns>, "load_duration": <ns> }`

- POST `/api/detokenize`
  - Request: `{ "model": "<model>" , "tokens": [<int>...], "options": { … } }`
  - Response: `{ "model": "<model>", "content": "<text>", "total_duration": <ns>, "load_duration": <ns> }`

Notes:
- Text-only. No `media_type` in public schema.
- No `keep_alive`. Loading is optimized internally.
- Durations are included for observability.

### Key Implementation Details
- Handlers: `server/routes.go`
  - `TokenizeHandler` and `DetokenizeHandler` parse requests, validate model names, then use `tokenizerloader.Get(ctx, model)` to obtain a tokenizer.
  - Handlers log which path was used (vocab-only vs fallback) and return tokens/content with timing information.
  - Routes are registered alongside other inference endpoints.

- Loader & Cache: `server/tokenizerloader/loader.go`
  - `Tokenizer` interface: `Tokenize(string) ([]int, error)`, `Detokenize([]int) (string, error)`, `Close() error`.
  - `Get(ctx, model)` returns `(Tokenizer, isFallback, error)`.
    - Cache hit → vocab-only path
    - Cache miss → attempts `openVocabOnly(...)`
    - Unavailable → graceful fallback to scheduler-backed tokenizer
  - LRU cache (default capacity 8) with eviction and `Close()` on eviction.
  - Debug logging gated by env var: `OLLAMA_TOKENIZER_DEBUG=1`.
  - Exported sentinel: `ErrVocabOnlyUnavailable`.
  - Test hooks (exported):
    - `ResetForTest()` → clears LRU and restores opener
    - `SetOpenVocabOnlyForTest(func(context.Context, string) (Tokenizer, error))` → dependency injection for tests

- Fallback wiring (scheduler): `server/routes.go`
  - Registers small function hooks to call `scheduleRunner` and execute `r.Tokenize(...)` / `r.Detokenize(...)`.
  - Avoids import cycles by keeping fallback APIs minimal and function-based.

### Vocab-only Path (today vs future)
- Today: `openVocabOnly` defaults to returning `ErrVocabOnlyUnavailable`. This preserves behavior across all models by using the fallback path.
- Future (easy flip): Wire `openVocabOnly` to llama.cpp’s vocab-only loader (via bindings), returning a tiny vocab-only model handle. This should avoid VRAM allocation and make cold times approach warm-cache latencies. The cache will then store those tiny tokenizers.

### Debugging & Observability
- Set both for detailed logs:
  - `OLLAMA_DEBUG=1` (enables slog.Debug)
  - `OLLAMA_TOKENIZER_DEBUG=1` (emits loader-specific logs)
- Expected log lines:
  - Vocab-only cache hit: `tokenizer: vocab-only cache hit`
  - Vocab-only load: `tokenizer: vocab-only load successful`
  - Fallback: `tokenizer: vocab-only unavailable, falling back to scheduler`
  - Handler path: `tokenize: using vocab-only` or `tokenize: using fallback scheduler`

### Tests
- Loader unit tests (`server/tokenizerloader`):
  - LRU eviction without panic
  - Basic concurrency safety (adds/gets under goroutines)
  - Debug flag path executes
  - Vocab-only preferred (mock success), fallback when unavailable (mock sentinel)
  - Exported test hooks simplify dependency injection without import cycles

- Handler tests (`server`):
  - Mount a minimal gin router with just `/api/tokenize` and `/api/detokenize`
  - Test vocab-only success (overridden opener), and fallback path (opener returns `ErrVocabOnlyUnavailable`)
  - Verify tokens/content and HTTP 200 responses

### Benchmarks (local, illustrative)
- Hardware: NVIDIA GeForce RTX 2060 SUPER (8 GiB VRAM); server started from this repo
- Model: `mistral:latest`
- Results observed:
  - Cold: ≈ 0.95s (`time curl … /api/tokenize … > /dev/null`)
  - Warm (5x): ~16–25ms real per call; P50 around ~18–19ms
- Returned examples:
  - Tokenize `"hello"` → `[7080, 29477]`
  - Detokenize `[2050]` → `" fam"`

Notes:
- These warm numbers reflect a resident process and scheduler-backed path today. Once vocab-only is wired, cold times should drop and warm times should remain in the same ballpark or better.

### Developer Workflow (local)
- Build & run from repo root:
  - `go build -o bin/ollama .`
  - `OLLAMA_TOKENIZER_DEBUG=1 OLLAMA_DEBUG=1 ./bin/ollama serve`
- Smoke tests:
  - `curl -s localhost:11434/api/tokenize -H 'Content-Type: application/json' -d '{"model":"mistral:latest","content":"hello"}' | jq`
  - `curl -s localhost:11434/api/detokenize -H 'Content-Type: application/json' -d '{"model":"mistral:latest","tokens":[2050]}' | jq`
- Tests & formatting:
  - `go test ./server/tokenizerloader -race -count=1`
  - `go test ./server -run Tokenize -count=1`
  - `go test ./... -count=1`
  - `go fmt ./...` and `go vet ./...`

### Files & Notable Changes
- Endpoints & routes
  - `server/routes.go`: request/response structs, handlers, route registration, fallback hook registration
- Loader & cache
  - `server/tokenizerloader/loader.go`: `Tokenizer` interface; `Get`; LRU cache; debug logs; exported error; fallback wrapper
  - `server/tokenizerloader/testutil.go`: `ResetForTest`, `SetOpenVocabOnlyForTest`
- Tests
  - `server/tokenizerloader/loader_test.go` (LRU + concurrency)
  - `server/tokenizerloader/loader_vocabbasic_test.go` (vocab-only vs fallback)
  - `server/routes_tokenize_handler_test.go` (gin-mounted handlers, vocab-only + fallback)
- Docs & examples
  - `docs/api.md`: Tokenize / Detokenize section (text-only, no keep_alive, examples, notes)
  - `api/examples/tokenize/bench.md`: reproduction steps + micro-bench
  - `upstream-links/tokenize-python.md`, `upstream-links/tokenize-js.md`: SDK tracking crumbs

### Compatibility & Surface Area
- New endpoints; no breaking changes
- Public schema does not include `keep_alive` or `media_type`
- Internals optimized for future vocab-only; present-day correctness via fallback

### Security & Stability
- Tokenization is read-only over model vocabulary; does not execute inference
- LRU cache is guarded with mutexes; eviction closes tokenizers
- Debug logs gated by environment variables

### Known Limitations / Future Work
- `openVocabOnly` currently returns `ErrVocabOnlyUnavailable`; fallback path is used today
- Wire vocab-only opening with llama.cpp bindings once available; then the cache will hold tiny vocab-only objects, further reducing cold latencies and memory usage
- Consider TTL/refresh policies and cache size tuning based on real workloads

### PR-ready Summary
- Expose `/api/tokenize` and `/api/detokenize` (text-only) with a vocab-only-first design and LRU cache; fall back to scheduler if vocab-only is unavailable
- Remove `keep_alive` and `media_type` from public schema for these endpoints
- Add debug logging behind `OLLAMA_DEBUG` + `OLLAMA_TOKENIZER_DEBUG`
- Provide tests (LRU, concurrency, debug, handler path selection)
- Update docs with examples and micro-bench table; add SDK tracking notes

### Example cURL
```bash
curl -s http://127.0.0.1:11434/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","content":"hello"}' | jq

curl -s http://127.0.0.1:11434/api/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","tokens":[2050]}' | jq
```

### One-liner for server with logs
```bash
OLLAMA_DEBUG=1 OLLAMA_TOKENIZER_DEBUG=1 ./bin/ollama serve
```

### Bench (quick local reproduction)
```bash
time curl -s http://127.0.0.1:11434/api/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistral:latest","content":"hello"}' > /dev/null

for i in {1..5}; do
  time curl -s http://127.0.0.1:11434/api/tokenize \
    -H 'Content-Type: application/json' \
    -d '{"model":"mistral:latest","content":"hello"}' > /dev/null
done
```

### Bottom Line
- Clean, fast, model-aligned tokenization endpoints with a future-proof design: vocab-only target, LRU cache, and safe fallback. It’s production-usable today and ready for direct wiring to llama.cpp vocab-only APIs as they land upstream.

