# PR: VRAM Accounting and GPU Splitting Fixes

## TL;DR

Fixes 6 critical bugs preventing two 24GB models from co-residing on 2× RTX 3090.

**What was pulled in:**
- Upstream PR #17857 (draft/MoE memory accounting) with 2 corrections

**What was added:**
- 5 additional fixes discovered during testing

All tests pass. Deployed on llm.cd34.com with 35+ successful two-model pairs.

## What Was Pulled In (PR #17857)
- Memory buffer attribution to target/draft/projector models
- Device PCI ID capture for verification
- Per-device free memory tracking

## Two Corrections Added on Top of #17857
1. Host weights count only the target (not draft)
2. Fully offloaded draft embeddings excluded from CPU accounting

## Five Additional Fixes
1. VRAM predictor now includes context/parallelism (was missing KV cache)
2. 80% safety margin removed from pre-flight check
3. Eviction targets correct GPUs only
4. Auto context fits single GPU (was splitting dense models)
5. NumGPU clamping only affects launch options (was causing reloads)

## Test Results
- All tests pass (`go test ./llm ./server ./ml`)
- 35+ successful two-model pairs tested
- `ollama ps` matches `nvidia-smi` within expected tolerances

## Files Changed
- `llm/vram_predict.go` (new): context/parallel-aware predictor
- `llm/llama_server.go`: #17857 port + 2 corrections + device mapping
- `server/sched.go`: eviction, auto-ctx, clamping fixes
- Test files and fixtures

## Related Issues
#17857, #19463, #18679, #18349, #16599, #17251, #19279, #18756
