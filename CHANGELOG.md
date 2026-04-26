# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [UNRELEASED]

### Added
- Intel Level Zero backend: `rope_f16`, `rms_norm_f16`, `add_f16` SPIR-V kernels for F16 KV-cache tensor support
- Intel Level Zero backend: 3D batched `mul_mat` (q8_0, q4_0, f16) with broadcast — matches GGML CUDA convention for all GQA/MHA attention projections
- Intel Level Zero backend: push-constant API for stride-aware kernel dispatch (5 push-const struct families: `ze_rope_pc` 112 B, `ze_rms_norm_pc` 88 B, `ze_softmax_pc` 128 B, `ze_binop_pc` 144 B, `mul_mat_pc` 160 B)
- Per-kernel unit-test infrastructure for L0 backend (`ml/backend/ggml/ggml/src/ggml-level-zero/tests/ze_kernel_test.cpp`); deferred runtime execution pending Intel Arc hardware
- Llama 3.2 1B integration coherence test for L0 backend (`TestL0LlamaCoherence` in `integration/level_zero_test.go`)
- Intel Level Zero performance regression test (`TestL0TokensPerSec`): threshold ≥ 2.0× CPU baseline tokens/sec

### Changed
- Intel Level Zero backend: all SPIR-V kernels now stride-aware — accept `nb[0..3]` byte strides via push constants (IDX macro matching GGML CUDA convention per ADR-L0-001 §3.2)
- Intel Level Zero backend: `ggml_l0_graph_compute` dispatcher rewritten to pass push-constant structs via `zeKernelSetArgumentValue(kernel, 0, sizeof(pc), &pc)` before buffer args
- Intel Level Zero backend: `ggml_backend_l0_supports_op` and `ggml_l0_dev_supports_op` (both copies) expanded — accept 3D batched `MUL_MAT` (removed 2D-only guard), F16 `ROPE` / `RMS_NORM`, and broadcast `ADD` / `MUL`

### Fixed
- Intel Level Zero backend: NaN logits on 3D batched `mul_mat` — root cause was stride-naive indexing that ignored `nb[1]`/`nb[2]`, causing out-of-bounds reads on all GQA attention projections in Llama 3.2 1B (ADR bug RC1)
- Intel Level Zero backend: F16 ROPE on KV-cache K-tensor view caused `GGML_ABORT` at `ggml-backend.cpp:844` — root cause was `supports_op` returning `false` for `GGML_TYPE_F16` ROPE ops, sending them to CPU fallback on an L0-allocated tensor (ADR bug RC2)
- Intel Level Zero backend: `rms_norm` / `softmax` / `add` / `mul` produced wrong results after `ggml_permute` / `ggml_reshape` / `ggml_view` — root cause was stride-naive indexing ignoring non-contiguous layouts (ADR bug RC3)
- Intel Level Zero backend: `softmax.cl` used `exp2f` (C99, not valid in OpenCL C) — replaced with OpenCL built-in `exp2`
- Intel Level Zero backend: `PFN_zeMemAllocShared` / `PFN_zeMemAllocHost` call-site mismatch between parallel contributors — aligned to `PFN_zeMemAllocHost` per `ze_buffer.hpp` header
