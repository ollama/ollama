# Ollama Fork — Gemma 4 MTP

## Previous conversation

Read `2026-05-28-045922-previous-conversation.txt` for full context on what's been done, what crashed, what was fixed, and what's left. It covers the entire history of wiring the DRAFT directive, fixing CUDA crashes, and testing MTP end-to-end.

This is a fork of ollama with Multi-Token Prediction (MTP) support for Gemma 4.

## Dev Container

`docker-compose.yml` defines a dev container (`ollama-dev`) with CUDA 13 devel, Go 1.26, cmake/ninja/ccache, and Claude Code. Source is bind-mounted at `/ollama`.

### First time setup inside the container

1. Build ggml CUDA backend (results cached in `ggml-build` volume):
   ```
   cmake --preset 'CUDA 13' -DCMAKE_CUDA_ARCHITECTURES=86
   cmake --build --preset 'CUDA 13' -j$(nproc)
   cmake --install build --component CUDA --strip
   cmake --preset CPU
   cmake --build --preset CPU -j$(nproc)
   cmake --install build --component CPU --strip
   ```

2. Build ollama: `go build -o /usr/bin/ollama .`

3. Create model (only needed once, stored in `ollama-data` volume):
   ```
   ollama serve &
   ollama create gemma4-mtp -f /models/Modelfile --quantize q4_K_M
   ```

### Iterating on Go code

`go build -o /usr/bin/ollama .` then restart `ollama serve`. No Docker rebuild needed.

### Testing MTP

```
curl http://localhost:11434/api/generate -d '{"model":"gemma4-mtp","prompt":"What is 2+2?","stream":false,"options":{"temperature":0,"num_ctx":4096}}'
```

MTP only activates at temperature=0 (greedy decoding). Check logs for "MTP eligible" and "MTP accepted" messages.

## Current MTP Status

- DRAFT Modelfile directive wired through standard GGUF convert path (no MLX) ✓
- ForwardMTP + MTPDraft run without CUDA crashes at temperature=0 ✓
- MTP acceptance working: ~74% of cycles accept tokens, ~51 t/s on RTX A6000 ✓
- Verified correct output matches non-MTP baseline ✓

### Bugs fixed in this session
1. **Attention scale**: Draft used 1/sqrt(hd), should be 1.0 (QK-norm like target)
2. **Constant position**: Draft used incrementing positions, should be constant (last token pos) per HF reference
3. **Hidden state buffer reuse**: GGML allocator reused hidden tensor memory; fixed with ggml_set_output
4. **Hidden float extraction**: Tensor.Floats() returned nil for FromFloats-created tensors; pass []float32 directly
5. **Input token**: Draft must use the INPUT token (position P), not the sampled OUTPUT token
6. **Reserve=true full range**: reserve=true spanned entire cache; changed to use sequence's actual cell range
7. **Verify batch structure**: Include sampled token in verify batch, compare at correct positions
8. **Pipeline integration**: Use rollback-based verify + seq.inputs injection for accepted drafts

## Key Files

- `model/models/gemma4/model.go` — ForwardMTP, MTPDraft, MTPVerify, HasDraft
- `model/models/gemma4/model_draft.go` — DraftModel, Draft(), Q-only attention, KV sharing
- `model/models/gemma4/model_text.go` — TextModel.Forward vs ForwardWithHidden
- `runner/ollamarunner/mtp.go` — isMTPEligible, runMTPCycle
- `runner/ollamarunner/runner.go` — forwardBatch (line ~634), computeBatch (line ~650)
- `convert/convert.go` — ConvertModelWithDraft
- `convert/convert_gemma4.go` — DraftKV metadata emission

## Hardware

RTX A6000 (49GB VRAM). Model at Q4_K_M + num_ctx=4096 uses ~23.7 GiB.
