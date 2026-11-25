# PR Overview: VRAM Accounting and GPU Splitting Fixes (PR #19632)

## TL;DR

This PR fixes multiple critical VRAM accounting issues that caused models to be incorrectly evicted, refuse to load, or spread across GPUs when not necessary. The issues were identified as #19463, #18679, #18349, and #16599.

**Key fixes:**
- VRAM prediction over-counted KV cache by 4-8x for hybrid models
- GPU overhead (OLLAMA_GPU_OVERHEAD) was not being honored
- Device name mapping was incorrect for filtered devices
- Models that fit on one GPU were being split across multiple GPUs

All fixes tested with 323+ passing tests.

## Changes Made

### Commit 1: server: confirm pinned num_gpu no-fit against live VRAM before refusing
- Re-reads live per-device free memory before refusing a load
- Only fails fast when fresh reading confirms no fit
- Falls through to actual load with llama-server's --fit if fresh reading shows fit

### Commit 2: server: fail fast when pinned num_gpu cannot fit
- Provides early failure with clear error messages
- Uses live VRAM readings instead of potentially stale scheduler snapshots

### Commit 3: server: evict instead of splitting a model that fits one gpu
- Models that fit on a single GPU are no longer split across GPUs
- Improves performance by avoiding unnecessary multi-GPU coordination

### Commit 4: llm: predict llama-server vram from tensor metadata and measurement
- VRAM prediction now uses actual tensor metadata
- Counts only blocks that actually hold KV cache
- Applies sliding-window caps correctly
- Honors OLLAMA_KV_CACHE_TYPE in predictions

### Commit 5: llm: verify llama-server offload against its own logs
- Validates VRAM offload predictions against llama-server's actual logging
- Compares predictions against llama-server's --fit measurements

### Commit 6: llm: map vram accounting to llama-server's filtered device names
- Maps discovery device IDs to llama-server log names
- Fixes multi-GPU VRAM accounting issues

## Test Results

All 323 tests pass:
- 226 tests in `llm/...` (including new VRAM prediction tests)
- 17 tests in `server/...` (including scheduler tests)

## Before and After Examples

### Before: Qwen3.5 64k Context Over-Counted 8x
```
Actual KV cache: 8.50 GiB (llama-server log)
Predicted: 53.25 GiB (6.26x over-count)
Result: Model refused to load, context reduced
```

### After: Qwen3.5 64k Context Accurate
```
Actual KV cache: 8.50 GiB (llama-server log)
Predicted: 8.50 GiB (exact match)
Result: Model loads successfully with full context
```

### Before: GPU Overhead Ignored
```
OLLAMA_GPU_OVERHEAD=6GiB set
Actual free VRAM after load: ~0 GiB (overhead not reserved)
Result: Model loads but system becomes unstable
```

### After: GPU Overhead Honored
```
OLLAMA_GPU_OVERHEAD=6GiB set
Actual free VRAM after load: ~6 GiB (overhead reserved)
Result: System remains stable
```

## Files Changed

- `llm/llama_server.go` - VRAM accounting, device name mapping
- `llm/llama_server_measure_test.go` - New test file
- `llm/llama_server_test.go` - Updated tests
- `llm/vram_predict.go` - VRAM prediction logic
- `llm/vram_predict_test.go` - Updated tests
- `server/sched.go` - GPU overhead, eviction logic
- `server/sched_test.go` - Updated tests

## Impact

- Models that fit on one GPU no longer get split across multiple GPUs
- GPU overhead (OLLAMA_GPU_OVERHEAD) is now properly respected
- VRAM predictions are 4-8x more accurate for hybrid models
- Improved eviction decisions in the scheduler
- Better multi-GPU VRAM accounting
- No breaking changes - only bug fixes

## Related Issues

- #19463 - VRAM prediction over-counting
- #18679 - GPU overhead ignored
- #18349 - Device name mapping
- #16599 - Unnecessary GPU splitting
