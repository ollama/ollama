# llama.cpp compatibility shim

This directory holds an in-process compatibility layer that lets upstream
`llama-server` load GGUFs produced by older versions of Ollama (and files
pulled from the Ollama registry) without re-converting or re-downloading.

The layer is applied automatically at build time via CMake `FetchContent`'s
`PATCH_COMMAND` — there is no separate "apply patches" step.

## Files

- `llama-ollama-compat.h`, `llama-ollama-compat.cpp` — the shim itself. These
  are regular source files owned by Ollama; they get copied into the fetched
  llama.cpp source tree during configure.
- `upstream-edits.patch` — small additive edits to upstream files so the
  shim gets called. Currently ~48 lines touching 6 files. Kept as a real
  `git` patch so re-generation on upstream bumps is one command.

## What the shim does

The shim runs at two well-defined points in the loader:

1. **After `gguf_init_from_file`**, for both the main model loader and the
   `mtmd/clip` loader: inspects the just-parsed metadata and decides whether
   the file is an Ollama-format GGUF. If so, it mutates the in-memory
   `gguf_context` and `ggml_context` (KV names, tensor names, tensor types)
   so the rest of the loader sees an upstream-shape file.

2. **After `load_all_data`**: applies any numerical fix-ups that need the
   tensors in their final backend buffers (e.g. RMSNorm `+1` if a future
   arch needs it — gemma3 doesn't).

Non-Ollama files are detected by the absence of Ollama-specific KV keys
(e.g. `gemma3.mm.tokens_per_image`) or embedded `v.*` / `mm.*` tensors in
the main model file. When no markers are present every compat function is
an immediate no-op.

## Currently supported architectures

| Arch | Text loader | Clip (mmproj) loader |
|---|---|---|
| `gemma3` | KV injection (`layer_norm_rms_epsilon`, `rope.freq_base`, `rope.freq_base_swa`), tokenizer vocab truncation, drop `v.*`/`mm.*` tensors | Arch rewrite to `clip`, KV synthesis (`clip.vision.*`, `clip.projector_type=gemma3`), tensor renames (`v.patch_embedding`→`v.patch_embd`, `mlp.fc{1,2}`→`ffn_{down,up}`, etc.), F16→F32 promotion for patch/position embeddings (Metal IM2COL requirement) |
| `qwen35moe` | head_count_kv array → scalar, rope dimension_sections pad 3→4, `ssm_dt`→`ssm_dt.bias` rename, drop `v.*`/`mm.*`/`mtp.*` tensors | Arch rewrite to `clip`, KV synthesis (`clip.vision.*`, `clip.projector_type=qwen3vl_merger`), per-block QKV merge (concat at load time), patch_embed reshape + F16→F32 + slice-as-temporal-pair (reclaiming an orphan `v.blk.0.attn_k` slot for the second pair) |
| `gptoss` | Arch rename `gptoss`→`gpt-oss` (incl. KV prefix), inject `gpt-oss.expert_feed_forward_length` from `ffn_gate_exps` shape, tensor renames (`attn_out`→`attn_output`, `attn_sinks`→`attn_sinks.weight`, `ffn_norm`→`post_attention_norm`) | n/a |
| `lfm2` | Tensor rename `output_norm.weight`→`token_embd_norm.weight`, fix stale `lfm2.feed_forward_length` from `ffn_gate` shape | n/a |
| `mistral3` | RoPE YaRN renames (`rope.scaling.beta_*`→`rope.scaling.yarn_beta_*`), `rope.scaling_beta`→`attention.temperature_scale`, drop `v.*`/`mm.*` tensors | Arch rewrite to `clip`, KV synthesis (`clip.vision.*`, `clip.projector_type=pixtral`), tensor renames (`v.patch_conv`→`v.patch_embd`, `v.encoder_norm`→`v.pre_ln`, `attn_output`→`attn_out`, `attn_norm`/`ffn_norm`→`ln1`/`ln2`, `mm.linear_{1,2}`→`mm.{1,2}`, `mm.norm`→`mm.input_norm`, `mm.patch_merger.merging_layer`→`mm.patch_merger`), zero-fill `v.token_embd.img_break` (reclaims `output_norm.weight` slot — Ollama's monolithic blob doesn't ship this tensor and per-row dequant of token_embd Q4_K is heavyweight; zero-fill makes [IMG_BREAK] insertion a no-op), F32 promote of `v.patch_embd.weight` (Metal IM2COL), LLaMA-style RoPE permute on vision Q/K (Ollama's converter skips repacking `v.*` tensors but pixtral expects HF-permuted layout) |
| `qwen35` | Same fixes as `qwen35moe` (head_count_kv array→scalar, rope dimension_sections pad 3→4, `ssm_dt`→`ssm_dt.bias`, drop `v.*`/`mm.*`/`mtp.*`) but for the non-MoE qwen3.5 (e.g. 9B). Both arches share `apply_qwen35_text_fixes`. | n/a |
| `gemma4` | Drop `a.*`/`v.*`/`mm.*` (audio + vision + projector) from the text loader. Covers both E2B/E4B (dense) and 26B-A4B (MoE). | n/a |
| `deepseekocr` | Arch rename `deepseekocr`→`deepseek2-ocr` (incl. KV prefix), inject `expert_feed_forward_length` from `ffn_down_exps` shape, `expert_shared_count` from `ffn_down_shexp` shape, default `attention.layer_norm_rms_epsilon`, drop `s.*`/`v.*`/`mm.*` | Arch rewrite to `clip`, KV synthesis (`clip.vision.*`, `clip.vision.sam.*`, `clip.projector_type=deepseekocr`, defaults for `feed_forward_length`/`projection_dim`/`window_size`/image stats), prefix-only rename `s.*`→`v.sam.*` (substring rename would corrupt `mm.layers`), CLIP leaf renames (`self_attn.{out,qkv}_proj`→`attn_{out,qkv}`, `layer_norm{1,2}`→`ln{1,2}`, `mlp.fc{1,2}`→`ffn_{up,down}`, `pre_layrnorm`→`pre_ln`), SAM leaf renames (`attn.proj`→`attn.out`, `attn.rel_pos_{h,w}`→`attn.pos_{h,w}.weight`, `norm{1,2}`→`{pre,post}_ln`), projector renames (`mm.layers`→`mm.model.fc`, `mm.image_newline`/`view_seperator`→`v.*`), F32 promote of `v.patch_embd.weight`, `v.sam.patch_embd.weight`, `v.position_embd.weight` |
| `nemotron_h_moe` | For latent-FFN variants (e.g. nemotron-3-super 120B-A12B): inject `moe_latent_size` from `ffn_latent_in.weight` ne[1], rename `ffn_latent_{in,out}`→`ffn_latent_{down,up}`. For all variants: drop `mtp.*` (Multi-Token Prediction tensors that Ollama emits as one-tensor-per-expert; ~1040 extras on the 120B). Standard variants (e.g. nemotron-cascade-2 30B-A3B) load with no rename, only the MTP skip. | n/a |

Usage:

```
llama-server --model /path/to/ollama-blob --mmproj /path/to/ollama-blob
```

Passing the same monolithic GGUF as both `--model` and `--mmproj` works —
each loader applies its own translation.

Additional architectures are added by implementing a `handle_<arch>()`
and (for vision models) `handle_<arch>_clip()` in `llama-ollama-compat.cpp`
and dispatching them from `translate_metadata` / `translate_clip_metadata`.

## Regenerating `upstream-edits.patch`

After upstream changes the insertion points (rare), re-apply the edits to
a fresh checkout and run:

```
cd /path/to/llama.cpp
git diff -- \
    ggml/include/gguf.h \
    ggml/src/gguf.cpp \
    src/CMakeLists.txt \
    src/llama-model-loader.cpp \
    src/llama-model.cpp \
    tools/mtmd/clip.cpp \
    > /path/to/ollama/llama/compat/upstream-edits.patch
```

## Why not fork llama.cpp or vendor it?

Forking means tracking upstream manually. Vendoring means snapshotting all of
llama.cpp's source in the Ollama tree (the old `llama/llama.cpp/` layout).
This shim keeps upstream unmodified on disk and the Ollama-specific logic
isolated in two files plus a small diff — upstream bumps are usually just
`LLAMA_CPP_VERSION` changes.

## Maintenance: non-public API dependencies

The compat code is mostly written against stable public APIs (`gguf.h`,
`ggml.h`, `ggml-backend.h`). There are three places where we lean on
something that isn't strictly public:

| Hack | Why | Escape hatch if upstream changes |
|---|---|---|
| Direct writes to `ggml_tensor::type` / `ne[]` / `nb[]` | No sanctioned mutator exists for post-creation tensor reshape/retype. Struct is public so this works today. | Ask upstream to expose `ggml_tensor_set_{type,shape}` helpers, or introduce them in our compat util and submit a PR. |
| `const_cast<char *>(gguf_get_tensor_name(...))` in `rename_tensor` | Pointer aims into a mutable `char[GGML_MAX_NAME]` buffer inside a `std::vector` element; the const is API hygiene. Lets us rename gguf tensors without a new public helper. | Add `gguf_rename_tensor` to `gguf.h` (10 lines) and drop the `const_cast`. |
| `llama_model_loader` forward-decl from `src/llama-model-loader.h` | Used only as an opaque pointer key for our skip-prefix registry. Never dereferenced. | Replace with `const void *` in our registry signatures. Zero behavioral change. |

None of these have changed in years. If an upstream bump breaks any of
them, each has a trivial workaround. See the top of
`llama-ollama-compat-util.h` for the inline notes.

## Documented hacks inside per-arch handlers

- **`reclaim_slot_as` (qwen35moe patch_embed split)** — repurposes an
  orphaned `v.blk.0.attn_k` slot (left over after the QKV merge) as a
  newly-synthesized `v.patch_embd.weight.1`. Needed because clip.cpp's
  `ctx_meta` is sized for exactly the original tensor count (no_alloc
  branch of `gguf_init_from_file` uses `n_tensors * ggml_tensor_overhead()`
  with zero slack). Comment in the helper and call site explains the
  reasoning; replacement would be a 1-line upstream patch that adds small
  slack to the ctx size.

- **Load-op registry overrides `file_offset`** — `maybe_load_tensor` gets
  passed the gguf offset by its caller but ignores it when a registered
  op exists. Intentional: the ops capture their own source offsets at
  translate time (before our renames invalidate them). Documented in the
  op-registration helpers.
