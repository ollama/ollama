# Issue #17018 — Vulkan Multi-GPU Allocation Anomaly

> 0 layers assigned to GPU0 causing static CPU fallback despite sufficient VRAM.
> 4 GPUs (2× AMD Pro VII 16 GB + 2× MI60 32 GB = 96 GB), model needs ~69 GB, yet
> GPU0 gets 0 layers and the ~3.1 GB output layer leaks to CPU host RAM.
> Reporter's workaround: `GGML_VK_VISIBLE_DEVICES=0,3,1,2` (reorder devices).

## 1. Root cause

When a model is too large for any single GPU, Ollama spreads it across a GPU
group and hands that group to `llama-server`, which auto-detects the per-device
layer split (its `common_params_fit_impl` back-to-front fit). Two facts combine
to strand a GPU:

1. **`llama-server` treats the *first* device in the list as the main GPU.** The
   main GPU holds the non-split **output tensor** and any KV-cache overflow, and
   the back-to-front layer fit is sensitive to the device ordering it is given.

2. **Ollama passed the GPU group in raw enumeration order.** In
   `server/sched.go`, `selectLlamaServerPlacement` → `bestGPUGroupByAvailableMemory`
   returned the selected group unsorted. That order flows straight through to the
   runner and to `GGML_VK_VISIBLE_DEVICES` (`ml.GetDevicesEnv`). So the physically
   first-enumerated GPU (a smaller 16 GB Pro VII in the report) became the main
   GPU. The fit then packs the larger back GPUs, runs low, and leaves the small
   front GPU with 0 block layers — while the output tensor, having no room on that
   stranded main GPU, falls back to CPU.

The codebase already has the correct comparator for this — `ml.ByFreeMemory`
(`ml/device.go`), whose own comment reads *"iGPUs are reported first, thus
Reverse() yields the largest discrete GPU first"* — but it was **only applied for
logging** (`discover/types.go: LogDetails`), never to the device list actually
handed to the runner. Reordering devices by hand (the reporter's
`GGML_VK_VISIBLE_DEVICES` trick) is exactly the lever that fixes the split; the
fix moves that ordering into Ollama so it is deterministic and no longer
hardware-enumeration-dependent.

Note: the fine-grained per-layer packing loop itself lives in llama.cpp's
vendored `common_params_fit_impl` (and Ollama's native `ml/backend/ggml/ggml.go`),
neither of which is part of this checkout — so the layer-index math cannot be
altered here. The reachable, in-tree lever is the **device ordering** Ollama
chooses before delegating the split, which is what this change corrects.

## 2. The fix and why

In `bestGPUGroupByAvailableMemory` (the sole selector for the multi-GPU / spread
path), order the chosen group with the most-free GPU first (discrete before
integrated) using the existing `ByFreeMemory` comparator before returning it:

```go
best = slices.Clone(best)
sort.Sort(sort.Reverse(ml.ByFreeMemory(best)))
return best
```

Why this ordering:

- The largest-free GPU becomes the **main GPU**, giving the output tensor and KV
  overflow a home instead of spilling to CPU — directly addressing the "3.1 GB
  output layer leaked to CPU" symptom.
- The split becomes **deterministic** and independent of hardware enumeration
  order, so a smaller front-enumerated GPU is no longer left at 0 layers. This is
  the in-code equivalent of the reporter's `GGML_VK_VISIBLE_DEVICES` workaround.
- It reuses the repo's own documented convention (`ByFreeMemory` /
  `sort.Reverse`) rather than inventing new ordering logic, and the single-GPU
  and explicit-`main_gpu` paths are untouched.

The group is cloned before sorting so the in-place sort never mutates the slices
owned by `ByLibrary`/the caller.

## 3. Files changed

- `server/sched.go` — `bestGPUGroupByAvailableMemory` now returns the spread group
  ordered largest-free-GPU-first (clone + `sort.Reverse(ml.ByFreeMemory(...))`),
  with an explanatory comment referencing the issue.
- `server/sched_test.go` — added a focused subtest to `TestSelectLlamaServerPlacement`
  (`"multi-GPU split orders largest free GPU first"`) modeling the issue's 4-GPU
  Vulkan layout and asserting the 31 GB device is selected first.

## 4. Risk / uncertainty

- **Behavior change scope:** This reorders the device list for *all* multi-GPU
  spread loads (CUDA/ROCm/Vulkan/Metal), not just Vulkan. The change is ordering
  only — the same set of GPUs is selected — and largest-free-first is a strictly
  reasonable main-GPU choice, so regressions are unlikely. Single-GPU compaction
  and explicit `main_gpu` requests are unaffected (they don't go through this
  function's spread return).
- **Not a proof of the exact llama.cpp fit outcome:** The precise per-layer
  packing decision still happens inside llama.cpp's `common_params_fit_impl`,
  which is outside this checkout. This change makes the *input ordering* to that
  fit deterministic and output-friendly (the documented, reporter-validated lever)
  rather than rewriting the fit math. The reporter's empirically-found order
  (`0,3,1,2`) is one working permutation; largest-free-first is a principled
  general ordering that keeps the output layer on a GPU with headroom.
- **Existing tests:** All previous `TestSelectLlamaServerPlacement` cases still
  pass — none asserted the ordering of a multi-GPU group, and the two spread
  cases only check library/count, which are unchanged.

## 5. How I verified it

- `go build` / `go vet ./server/` — clean.
- `go test ./server/ -run TestSelectLlamaServerPlacement -v` — all 7 subtests pass,
  including the new `multi-GPU split orders largest free GPU first`.
- `go test ./server/` (full package) — `ok`.
- `go test ./ml/` (source of `ByFreeMemory`) — `ok`.

Manual reasoning against the report: with the 4-GPU layout, the new ordering puts
a 32 GB MI60 first as the main GPU, so the output tensor lands on it instead of
CPU, and no 16 GB GPU is left as a stranded front device — matching the outcome
the reporter obtained via `GGML_VK_VISIBLE_DEVICES` reordering.
