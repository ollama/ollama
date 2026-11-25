# PR Overview: Ollama Dual-GPU Co-Residency Fixes

## TL;DR

This PR pulls in upstream PR #17857 (draft/MoE memory accounting) and extends it with:
1. **Two corrections** on top of #17857 (host weights + draft embeddings accounting)
2. **Five additional fixes** discovered during testing

All fixes deployed on `llm.cd34.com` with 35+ successful two-model pairs on 2× RTX 3090.

## What Was Pulled In

**Upstream PR #17857 (dhiltgen)** - Draft/MoE memory accounting for `ollama ps`:
- Memory buffer attribution to target/draft/projector models
- Device PCI ID capture for GPU verification
- Per-device free memory tracking

## What We Added on Top of #17857

1. **Host weights count only the target** (not the draft) - fixes guard that rejected fully offloaded gemma loads
2. **Fully offloaded draft embeddings excluded from CPU accounting** - fixes incorrect "1%/99%" reporting

## Five Additional Fixes Discovered During Testing

1. **VRAM predictor** now includes context/parallelism (was missing KV cache)
2. **80% safety margin removed** from pre-flight check (prediction already includes reserve)
3. **Eviction targets correct GPUs** only (was evicting wrong card)
4. **Auto context fits single GPU** (was splitting dense models across both cards)
5. **NumGPU clamping** only affects launch options (was causing unnecessary reloads)

## Testing

- `go test ./llm ./server ./ml` passes
- 35+ successful two-model pairs tested
- `ollama ps` matches `nvidia-smi` within ±256..+768 MiB

## Files Changed

- `server/sched.go`: 5 new helpers + 80% margin removal
- `llm/llama_server.go`: #17857 port + 2 corrections
- `llm/vram_predict.go` (new): context/parallel-aware predictor
- `llm/testdata/*.log` (new): real load fixtures

## Related Upstream Issues

- #17857 (draft/MoE accounting) - **pulled in and extended**
- #16599 (80% margin) - **fixed**
- #17251 (draft MoE accounting) - **fixed**
- #19279, #18756 (eviction) - **fixed**
