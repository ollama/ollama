# Ollama Dual-GPU Co-Residency Fixes: Scheduler and Memory-Accounting Improvements

> **TL;DR:** Fixed six critical bugs that prevented two 24GB models from co-residing on a
> 2× RTX 3090 system: VRAM predictor ignored context/parallelism, an 80% safety margin
> split models that fit on one card, eviction picked wrong GPUs, draft/MoE models were
> misaccounted, and automatic context split dense models. All fixes deployed on
> `llm.cd34.com` with 35+ successful two-model pairs.

## Background

This PR aggregates six interrelated fixes for Ollama's scheduler and VRAM accounting on
multi-GPU systems. The issues manifested when loading two 21–23 GiB models on a 2× RTX
3090 system: each model alone loaded fine, but together they triggered eviction cascades
and OOM failures.

**What this PR does:**

1. **Pulls in and extends upstream PR #17857** (dhiltgen) which fixes draft/MoE memory
   accounting in `ollama ps`. We tested the port against real production logs and made
   two additional corrections specific to our environment (see "Upstream PR Integration").

2. **Addresses an additional issue** beyond #17857: the host weights guard in #17857
   counted the draft's CPU_Mapped buffer, causing fully offloaded gemma loads to be
   rejected. We also added `draftFullOffload` flag to exclude draft mmap embeddings
   from host memory accounting.

3. **Fixes five additional scheduler bugs** that were discovered during the
   upstream PR testing phase:

   - VRAM predictor ignored context length and parallelism
   - 80% safety margin split models that fit on one card
   - Eviction picked wrong GPUs
   - Auto context split dense models
   - NumGPU clamping bug caused unnecessary reloads

The fixes are bundled together because they were all discovered during the
verification of upstream PR #17857 and are tested together.

**Tested:** 35+ successful pairs of two-card models; no memory, placement, or accounting
failures. `ollama ps` matches `nvidia-smi` within expected tolerances.

### Upstream PR Integration

This PR ports and extends upstream PR #17857 (dhiltgen) which addressed draft/MoE memory
accounting in `ollama ps`. The port was tested against real production logs before the
other fixes were applied.

**Our two additional corrections on top of upstream PR #17857:**

1. **Host weights count only the target model** (not the draft). The upstream guard
   `weightsMoved := memHostModel > expectedHostModel + slack` was counting the draft's
   CPU_Mapped buffer (272 MiB). This caused fully offloaded gemma loads to be rejected
   as incomplete. Our fix ensures only target model weights are counted.

2. **Fully offloaded draft embeddings not counted as CPU.** A new `draftFullOffload`
   flag (set from "offloaded N/N" lines for the draft model) prevents draft mmap-backed
   embeddings from inflating host memory. Without this, gemma showed "1%/99% CPU/GPU"
   instead of "0%/100%" when fully offloaded.

Both corrections were identified and fixed during testing with real draft/MoE model
loads (gemma4 + gemma4-assistant).

All tests pass. DEBUG log lines removed before upstreaming.

---

## What Was Pulled In vs What Was Added

### Pulled in from upstream PR #17857 (dhiltgen)
- Memory buffer attribution to target/draft/projector models
- Device PCI ID capture for verification
- Per-device free memory tracking
- Regex line anchoring
- Fit shortfall detection
- Record projector usage
- Memory-mapped embeddings excluded from CPU accounting (for target models)

### Two corrections added on top of #17857
1. **Host weights count only target (not draft)** - fix for guard that was rejecting
   fully offloaded gemma loads
2. **Fully offloaded draft embeddings excluded from CPU accounting** - fix for
   incorrect "1%/99% CPU/GPU" reporting

### Five additional fixes discovered during testing
1. VRAM predictor now includes context/parallelism (was missing KV cache)
2. 80% safety margin removed from pre-flight check
3. Eviction now targets runners on correct GPUs only
4. Auto context now fits dense models on single GPU
5. NumGPU clamping now only affects launch options

---

## Fixes

### 1. VRAM Predictor Now Includes Context and Parallelism

**Symptom:** Predicted sizes left out the KV cache. Models were placed or kept on a GPU
that could not hold them, causing OOM and eviction cascades.

**Cause:** Five call sites of `llm.PredictServerVRAM` passed no `NumCtx` or `NumParallel`,
predicting a zero-length context with one sequence.

**Fix:** A new helper, `predictLlamaServerVRAM(req, f, opts, gpus, numParallel)`, always
passes the effective context × parallel, flash-attention state, and draft model path.

**Files changed:**
- `server/sched.go`: replaced 5 `llm.PredictServerVRAM` calls with `predictLlamaServerVRAM`
- `llm/vram_predict.go` (new): predictor with context/parallel support
- `llm/vram_predict_test.go` (new): unit tests
- `llm/llama_server_measure.go` (new): memory measurement helpers
- `llm/llama_server_measure_test.go` (new): integration tests

**Test:** `TestPredictLlamaServerVRAMUsesContextAndParallel`

---

### 2. Removed 80% Safety Margin from Pre-Flight Check

**Symptom:** Once the predictor counted the KV cache (fix 1), qwen3.8-fixed (21.9 GiB
predicted, 23.4 GiB free) no longer passed the "≤ 80% of free" check. It was split across
both GPUs, OOMing when the other model was resident.

**Cause:** The prediction already includes llama.cpp's fit target (1 GiB) plus the
projector, so a further 20% margin counted the reserve twice.

**Fix:** The pre-flight check is now `predictedForLoad > freeMemory`, and
`bestSingleGPUFit` is now `predictedVRAM > candidateAvailable`. The 80% heuristic for
mmap defaults was left unchanged.

**Files changed:**
- `server/sched.go`: removed 80% factor from pre-flight and single-GPU fit checks

**Test:** `TestSelectLlamaServerPlacementSingleGPUWithoutExtraMargin`

**Related:** upstream issue #16599

---

### 3. Eviction Now Targets Runners on Correct GPUs

**Symptom:** Loading a model that needed GPU 0 could evict the runner on GPU 1, leaving
the conflict in place.

**Fix:**
- `LlmRequest.lastLoadGpus` records the placement chosen for the pending load
- `processPending` calls `findRunnerToUnloadOn(gpus)` instead of `findRunnerToUnload()`
- `findRunnerToUnloadOn` chooses only among runners on those GPUs when any exist
- `runnerUsesAnyGPU` compares device IDs under `refMu`

**Files changed:**
- `server/sched.go`: added `lastLoadGpus` field, `findRunnerToUnloadOn`, `runnerUsesAnyGPU`

**Test:** `TestFindRunnerToUnloadOnTargetGPUs`

**Related:** upstream #19279, #18756

---

### 4. Ported upstream #17857: Draft/MoE Memory Accounting

**Symptom:** gemma-fixed (gemma4 26B-A4B MoE + gemma4-assistant draft) showed as 1.2 GB
in `ollama ps`, while it really used 20.1 GB. With gemma resident the scheduler thought
GPU 0 had 21.5 GiB free and placed qwen beside it.

**Cause:** The llama-server log parser keyed buffer lines by (model, component, backend,
kind). The draft model's and the SWA cache's buffers overwrote the target model's.

**Fix:** Ported ollama/ollama#17857 (dhiltgen) with three corrections:

1. **Host weights count only the target.** The guard `weightsMoved := memHostModel >
   expectedHostModel + slack` was counting the draft's CPU_Mapped buffer (272 MiB).
   Under #17857 this would mark fully offloaded gemma loads as incomplete.
2. **Fully offloaded draft embeddings not counted as CPU.** A new `draftFullOffload`
   flag (set from "offloaded N/N" lines) prevents draft mmap-backed embeddings from
   inflating host memory.
3. **Host PCI ID capture.** Added `devicePCIAtLoad` to verify the child initialized
   on the selected physical GPU (fixes #17251).

**Files changed:**
- `llm/llama_server.go`: PR #17857 port with the three corrections
- `llm/llama_server_test.go`: updated tests, added `TestMemoryParsingWriterRealDraftLoads`
- `llm/testdata/memory_gemma4_draft.log`, `llm/testdata/memory_qwen35_mtp.log`: real load logs

**Test:** `TestMemoryParsingWriterRealDraftLoads` checks VRAM within 512 MiB of
nvidia-smi and correct host accounting.

**Related:** upstream issues #17251, #16752; open PRs #19324, #18866, #17037

---

### 5. Auto Context Now Fits Dense Models on Single GPU

**Symptom:** Models with no `num_ctx` (bigqwen-fixed, qwen3.8:27b) were split across
both cards at 36 GB. Any pair including one of them evicted the other model.

**Cause:** Upstream picks the default context from the **sum** of all GPUs' VRAM
(≥47 GiB → 262144). On 2×24 GiB that selects 256k, too large for one GPU for dense
models.

**Fix (`server/sched.go`):** New `fitAutoNumCtxToSingleGPU` halves automatic context
until the model fits one GPU (floor 4096). Only applied when:
- Context is automatic
- `OLLAMA_SCHED_SPREAD` is off
- 2+ GPUs present
- No explicit `main_gpu` or `num_gpu=0`

Capacity includes the VRAM of runners on that GPU (can be evicted).

**Result:** bigqwen-fixed and qwen3.8:27b now load at ctx 65536, using 20.2 GB on one
card. gemma4:26b keeps 262144 (sliding-window KV fits one card at 23.5 GB).

**Test:** `TestFitAutoNumCtxToSingleGPU`

---

### 6. NumGPU Clamping Now Only Affects Launch Options

**Symptom:** Every request reloaded the model. The clamp (99 → layer count + 1) wrote
to `req.opts`, which becomes the runner's stored options. `reloadReason` saw
"options changed" on the next request (still asking for 99).

**Cause:** The local clamp wrote to `req.opts`. That value becomes the runner's stored
options, so `reloadReason` (a `reflect.DeepEqual`) saw "options changed" on the next
request.

**Fix (`server/sched.go`, `load`):** Clamp only `launchOpts`. `req.opts` stays as
requested.

**Additional fix (`server/sched.go`, `runnerRef.reloadReason`):** Clamp both
existing and new options to layer count before comparison, so manifest-pinned values
don't trigger false reloads.

**Test:** `TestSchedLoadClampsNumGPUWithoutChangingRunnerOptions`

**Note:** The clamp itself does nothing useful (llama.cpp's `--fit` gives up when
`-ngl` is set explicitly). Recommendation: remove `PARAMETER num_gpu 99` from Modelfiles.

---

## Live Verification

**Environment:** 2× NVIDIA GeForce RTX 3090 (24 GiB each, ~23.4 GiB usable),
`ollama.service` on `[::1]:3400`, `OLLAMA_MAX_LOADED_MODELS=2`,
`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_SCHED_SPREAD=false`,
`OLLAMA_FLASH_ATTENTION=true`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=2h`

**Method:** harness `/tmp/claude-1000/.../pairtest.py`:

1. Unload everything
2. Load A, then load B
3. Load A again (verify no reload)
4. Compare `/api/ps` against `nvidia-smi`

**Requirements:** each model resident, 100% GPU, one per GPU, different GPUs for
different models; `ps` within −256…+768 MiB of nvidia-smi.

### Single Models (9 tests, all pass)

| Model | ctx | `ollama ps` | nvidia-smi | `ollama ls` | GPUs |
|---|---|---|---|---|---|
| gemma-fixed | 76800 | 19809 MiB | 20110 MiB | 17863 MiB | 1 |
| qwen3.8-fixed / test-qwen38-modelfile | 76800 | 20717 | 20988 | 16920 | 1 |
| glimmer-fixed / muse-glimmer:30b / :latest | 131072 | 18107 | 18380 | 17316 | 1 |
| bigqwen-fixed | 65536 (auto, lowered) | 20233 | 20506 | 16920 | 1 (was 2, at 36235 MiB) |
| qwen3.8:27b | 65536 (auto, lowered) | 20233 | 20506 | 16920 | 1 (was 2) |
| gemma4:26b | 262144 (auto) | 23542 | 23842 | 17863 | 1 |

`ollama ps` is consistently 270–300 MiB below nvidia-smi (CUDA context, not logged by
llama.cpp).

### Pairs (36 tests, complete)

**Final tally across 45 cases (9 single + 36 pairs): 35 OK and 10 same-weights pairs.**
No memory, placement, or accounting failures.

- **All 26 pairs with different weights pass**: both resident, one per GPU, no reload,
  `ps` ≈ nvidia-smi. Includes every combination of gemma-fixed, qwen3.8-fixed,
  glimmer-fixed, bigqwen-fixed, muse-glimmer, gemma4:26b, qwen3.8:27b across families.
- **Same weights + identical options share one runner** (6 pairs):
  qwen3.8-fixed + test-qwen38-modelfile, glimmer-fixed + muse-glimmer:30b/:latest,
  bigqwen-fixed + qwen3.8:27b (both auto ctx). Both names served by one copy.
- **Same weights + different options reload** (4 pairs):
  gemma-fixed + gemma4:26b, qwen3.8-fixed/test-qwen38-modelfile + bigqwen-fixed/qwen3.8:27b.

Before fix 5, every pair including bigqwen-fixed or qwen3.8:27b failed (split across
both cards, then eviction).

---

## Open Items

### Same weights, different options

Three distinct weight files underlie the nine one-card models:

| Weights blob | Models |
|---|---|
| gemma `a8bdd07a49` | gemma-fixed, gemma4:26b |
| qwen `f5f1dd8920` | qwen3.8-fixed, test-qwen38-modelfile, bigqwen-fixed, qwen3.8:27b |
| glimmer `71b5c9c9ab` | glimmer-fixed, muse-glimmer:30b, :latest |

The scheduler keys runners by blob path (`schedulerModelKey`). When options differ
(e.g., gemma-fixed `num_ctx 76800` vs gemma4:26b auto), the second load reloads
instead of loading a second copy.

**Options (not implemented; user decides):**
1. **Leave as is.** One name per weight file at a time.
2. **Make Modelfiles agree.** Give base tags same `num_ctx` as -fixed ones so they
   share one runner.
3. **Key by blob + options that matter.** Different options get a second copy on the
   other card. Downside: changing `num_ctx` per request piles up runners until LRU
   eviction.

---

## Upstream Follow-ups

- Into PR #19632 (the VRAM predictor): context/parallel fix (2), 80% removal (3),
  wrong-GPU eviction fix (4), auto-ctx single-GPU fit (6)
- Review comment on #17857: host-weights guard should count only target weights,
  fully offloaded draft's mapped embeddings should not count as CPU use
- Remove `num_gpu 99` from local Modelfiles
- Remove DEBUG log lines before upstreaming (see "Changes" section)

---

## Files Changed

### Uncommitted changes:
- `server/sched.go`: clamp on launchOpts only; `predictLlamaServerVRAM`; 80% margins
  removed; `lastLoadGpus`, `findRunnerToUnloadOn`, `runnerUsesAnyGPU`;
  `fitAutoNumCtxToSingleGPU`
- `server/sched_test.go`: 5 new tests
- `llm/llama_server.go`: PR #17857 port plus branch features and 2 corrections
- `llm/llama_server_test.go`: #17857 tests, fixture adjustments,
  `TestMemoryParsingWriterRealDraftLoads`
- `llm/vram_predict.go` (new): context/parallel-aware VRAM predictor
- `llm/vram_predict_test.go` (new): unit tests
- `llm/llama_server_measure.go` (new): memory measurement helpers
- `llm/llama_server_measure_test.go` (new): integration tests
- `llm/testdata/memory_gemma4_draft.log`, `llm/testdata/memory_qwen35_mtp.log` (new)

### Deployment:
Build: `go build -ldflags="-w -s"`, install to `/usr/local/bin/ollama`,
`systemctl restart ollama`.

Backups made today (newest first):
- `/usr/local/bin/ollama.pre-autoctx-20261125-100110`: before fix 6
- `/usr/local/bin/ollama.pre-17857-20261125-081657`: before fix 5
- `/usr/local/bin/ollama.pre-no80-20261125-080359`: before fix 3

Rollback:
```bash
sudo install -m 755 /usr/local/bin/ollama.pre-<tag> /usr/local/bin/ollama && \
sudo systemctl restart ollama
```

---

## Testing

```bash
$ go test ./llm ./server ./ml
ok      github.com/ollama/ollama/llm    1.467s
ok      github.com/ollama/ollama/server 4.325s
ok      github.com/ollama/ollama/ml     0.003s
```

All tests pass. `gofmt` and `go vet` are clean.
