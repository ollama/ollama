# Gemma 4 MTP → vendored llama.cpp port plan

Goal: port the Gemma 4 Multi-Token Prediction (MTP) speculative-decode path from our
Go engine (`model/models/gemma4/model_draft.go`, `runner/ollamarunner/mtp.go`,
`kvcache/causal.go`) into the **vendored llama.cpp** (`llama/llama.cpp/`), structured so
it survives an uprev of the vendored base.

Reference implementation: `AtomicBot-ai/atomic-llama-cpp-turboquant`, branch
`feature/turboquant-kv-cache`, cloned at `/mnt/ssdpool/projects/atomic-llama-cpp-turboquant`.
Their design matches ours: assistant loaded **into the target context** (single
`llama_context`, no second tokenizer/KV/sampler), cross-attending the target's KV cache.

## Prerequisite RESOLVED: gemma4 TARGET arch is upstream

Verified: **upstream llama.cpp master has native `LLM_ARCH_GEMMA4`** (target), in
`src/llama-arch.cpp`. It does NOT have `gemma4_assistant` (fork-only). Our vendored base
`ec98e2002` predates the gemma4 merge, which is why it's missing locally.

**Decision: do NOT port the gemma4 target.** It arrives when we uprev the vendored base past
upstream's gemma4 merge. We carry only the MTP (`gemma4_assistant`) patches, cut against an
upstream base that already has gemma4 target.

### Dev substrate decision
- **NOT ollama-upstream.** It no longer vendors llama.cpp source in-tree (only `llama/compat/`
  + `llama/server/` shims over an external llama.cpp); nothing to port into. (Also: its Go
  engine is NOT deleted — `model/models/gemma4/` is alive on main. The "deleted backend" was a
  misread.) Its `llama/compat/llama-ollama-compat.cpp::handle_gemma4` is just GGUF metadata
  fixup (incl. the BPE-tokenizer flip that makes `<|channel>`/`<|thought>`/`<|turn>` register
  as control tokens) — useful prior art, not a dev base.
- **Develop against upstream llama.cpp HEAD** (has gemma4 target; common ancestor of our fork
  post-uprev, ollama's external llama.cpp, and atomic). A clean series there applies to all.
- Pragmatic build/test base: **atomic's tree itself** (it already = upstream llama.cpp +
  gemma4 + gemma4_assistant + TurboQuant), develop an MTP-minus-TurboQuant variant there,
  then express as patches vs upstream.

## TurboQuant entanglement (no clean historical extraction)

Hoped to grab a pre-TurboQuant MTP commit; **doesn't exist.** In atomic, TurboQuant landed
FIRST and MTP was built on top — `feature/gemma-mtp` tip still has 43 turbo refs in
`src/llama-graph.cpp`, and turbo commits sit *beneath* the MTP commits in history. So:
- MTP source-of-truth diff range (the additive MTP work): **`72f60cf85^..5d1ce14d4`** plus the
  async/centroid follow-ups (`46fb4eea7`, `7d2b8870e`, `926ce1e14`, `f50b381ce`, `5d1ce14d4`).
- Those diffs **embed turbo-KV handling inside the new MTP code** (esp. `build_attn_mtp`
  unpadding). TurboQuant stripping is **manual**, done while porting each MTP hunk onto plain
  upstream KV cache — not achievable by branch/commit selection.

## POST-UPREV STATE (commit a88a73bb on branch `uprev-main`)

The uprev to upstream ollama main is DONE (merged 14 commits incl. #16031). It
**removed in-tree `llama/llama.cpp/` source entirely.** llama.cpp is now obtained via
**CMake FetchContent** from a pinned `LLAMA_CPP_GIT_TAG` (see `llama/server/CMakeLists.txt`
+ `llama/compat/compat.cmake`); fork customizations apply via `llama/compat/` (compat source
linked into the fetched tree) + `llama/patches/` (now 1 patch, was 37).

### New MTP integration approach (simpler than patching in-tree)
1. Build MTP into a **llama.cpp fork**: base = `/mnt/ssdpool/projects/llama.cpp-upstream`
   branch `gemma4-mtp` (arch registration for `gemma4_assistant` already landed there),
   + atomic's `gemma4_assistant` graph/decode + TurboQuant (kept, per user).
2. Push that fork somewhere ollama can fetch.
3. **Repoint ollama's FetchContent** `GIT_REPOSITORY`/`GIT_TAG` (`LLAMA_CPP_GIT_TAG`) at our
   MTP llama.cpp fork. One pin change wires the engine in. Keep `llama/compat/handle_gemma4`
   (the BPE tokenizer fix for `<|channel>`/`<|thought>` control tokens).

### Working checkouts under /mnt/ssdpool/projects/
- `ollama-fork` — branch `uprev-main` (uprev committed a88a73bb). origin = wow-look-at-my/ollama.
- `llama.cpp-upstream` — branch `gemma4-mtp` at upstream d749821; arch reg done (uncommitted edits in src/llama-arch.{h,cpp}). This becomes our llama.cpp MTP fork.
- `atomic-llama-cpp-turboquant` — branch feature/turboquant-kv-cache; reference for gemma4_assistant + TurboQuant.

### Next steps
1. Find the pinned `LLAMA_CPP_GIT_TAG` in `llama/server/CMakeLists.txt`; confirm it has gemma4 target + the Qwen3.5 MTP framework (DECODER_MTP).
2. In `llama.cpp-upstream` (gemma4-mtp): commit arch reg, port atomic's gemma4-assistant.cpp (turbo-stripped or with turbo per user), build_attn_mtp, model load, wire into DECODER_MTP idiom.
3. Re-layer fork build customizations (task #11). Verify build. Repoint FetchContent. Bench.

## Patch workflow (legacy — pre-uprev in-tree approach, no longer used post-uprev)

Per `llama/README.md` + `Makefile.sync` (base `ec98e2002`, 37 patches via `git am -3`):
1. `make -f Makefile.sync clean apply-patches`  → checks out upstream into `llama/vendor/`, applies patches
2. iterate inside `llama/vendor/`, commit there
3. `make -f Makefile.sync format-patches`        → regenerates `llama/patches/00NN-*.patch`
New MTP work becomes a series of appended patch files.

## Source map: atomic fork → vendored llama.cpp

| Concern | Atomic fork (reference) | Vendored target | Notes |
|---|---|---|---|
| Assistant arch enum + name | `src/llama-arch.h:66`, `src/llama-arch.cpp:62` | `src/llama-arch.{h,cpp}` | add `LLM_ARCH_GEMMA4_ASSISTANT` |
| MTP tensor names (`mtp.*`) | `src/llama-arch.cpp:558-561`, `.h:567-572` | same | pre/post_projection, centroids, token_ordering |
| MTP KV metadata keys | `src/llama-arch.h:349-354` | same | n_centroids, centroid_top_k, n_embd_backbone, k_eq_v, ordered_embeddings, requires_target_arch |
| Model load + accessors | `src/llama.cpp:1187-1265` | `src/llama.cpp` | `llama_model_load_mtp_from_file`, `..._get/has_mtp_assistant`, `..._mtp_n_embd_backbone` |
| Public C API | `include/llama.h:503-506,1009-1045` | `include/llama.h` | load + `llama_decode_mtp{,_async,_wait}` |
| Graph builder | `src/models/gemma4-assistant.cpp` (43-299) | `src/models/gemma4-assistant.cpp` (new) | `gemma4_mtp_build_one_step`, centroid head, in-graph argmax |
| Cross-attention | `src/llama-graph.{h,cpp}` (h:986-1000, cpp:2490-2567) | same | `build_attn_mtp` — **strip TurboQuant unpadding** (TURBO3_0/4_0/2_0 paths) |
| KV cache MTP slot | `src/llama-kv-cache.cpp:992-1022`, `-iswa.cpp:220-232` | same | `mtp_slot_info`, `init_mtp` (append-only, read-only cross-attn) |
| Context async worker | `src/llama-context.{h,cpp}` (h:418-445, cpp:2744-2843,1248-1354) | same | `sched_mtp`, `mtp_worker_loop`, `decode_mtp_run`, `ensure_sched_mtp` |
| Public async wrappers | `src/llama-context.cpp:4249-4272` | same | — |
| Speculative driver | `common/speculative.cpp:578-684,2033-2073` | `common/speculative.cpp` | `common_speculative_state_mtp`, `prepare_next`, drain points, `set_h_idx` |
| CLI flag | `common/arg.cpp:3494-3516` | `common/arg.cpp` | `--mtp-head` (+ `-md` alias) |
| Go cgo exposure | n/a | `llama/llama.go:168-175,378-447` | new `DecodeMTP*` wrappers over the new C API |

## What to STRIP from the atomic reference

The atomic fork entangles MTP with **TurboQuant** (WHT-rotated KV cache, TURBO{2,3,4}_0
quant types). For a clean MTP-only port, drop the turbo unpadding/rotation branches in
`build_attn_mtp` (cpp:2527-2556) and keep just: get target K/V → `build_attn_mha` →
output proj. TurboQuant is an independent feature we are not porting.

## Where OUR Go impl is the better reference (port our fixes, not theirs)

Our SWA-safe speculation (`kvcache/causal.go` BeginSpeculation/Commit/Rollback; fix commits
`65a0e5b3`, `d1c2c0eb`, `b8c0324f`) is the subtle correctness work. The atomic fork relies
on an append-only invariant (all MTP step positions strictly > max stored pos, so masks
uniformly pass) rather than rollback. Validate that invariant holds in our integration; if
we ever write speculative cells into the target cache, our rollback semantics are needed.

## MAJOR FINDING: upstream already has a generic MTP framework

Upstream llama.cpp HEAD (`d749821`) already ships MTP infrastructure, used by **Qwen 3.5/3.6**:
- `LLAMA_CONTEXT_TYPE_MTP` (include/llama.h:203) + `ctx_type` param (llama.h:345)
- `LLM_GRAPH_TYPE_DECODER_MTP` (llama-graph.h:36); context maps CTX_TYPE_MTP → this (llama-context.cpp:28)
- Real draft heads: `src/models/qwen35.cpp:133` ("DECODER_MTP draft head for Qwen3.5/3.6 dense"),
  `src/models/qwen35moe.cpp:156,554` (MoE). They branch inside the model's `build_*` on
  `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP`.

**Consequence — atomic's custom MTP machinery is mostly obsolete for an upstream-based port.**
Atomic forked an OLDER llama.cpp without this framework, so it invented its own `decode_mtp`
async API, worker thread, `init_mtp` KV path, and `common_speculative_state_mtp` driver. Upstream
now provides the generic path. So:
- **DON'T port** atomic's `decode_mtp{,_async,_wait}`, `mtp_worker_loop`, `ensure_sched_mtp`,
  `llama_model_load_mtp_from_file` custom API, or its speculative driver. Follow the **Qwen 3.5
  idiom** instead (DECODER_MTP graph branch + LLAMA_CONTEXT_TYPE_MTP + however the spec loop drives it).
- **DO** implement the gemma4 draft head in that idiom, using atomic's `gemma4-assistant.cpp` as a
  **reference for the gemma4-specific math only**: cross-attention into the target's shared KV
  (last layer per attn-type), single fixed RoPE position, centroid-routed LM head + in-graph argmax.
- The graph builder itself is turbo-clean (turbo lives only in the `build_attn_mtp` helper), so the
  gemma4-specific graph math ports almost verbatim — it just needs to be hosted in upstream's idiom.

NEXT: study qwen35.cpp's DECODER_MTP branch end-to-end (how the draft head is built, how the
context/KV provide the target hidden state + KV, and how the speculative loop calls it). Mirror
that shape for gemma4; borrow only gemma4 math from atomic. Tasks #6/#7/#8 shrink accordingly.

## Sequencing

1. [gate] Resolve gemma4 target arch (uprev vs port). 
2. Arch + tensor + KV-metadata registration (`llama-arch.*`) — smallest, self-contained patch.
3. Model load path + accessors + public API decls.
4. Graph builder `gemma4-assistant.cpp` + `build_attn_mtp` (TurboQuant stripped).
5. KV `mtp_slot_info` / `init_mtp`.
6. Context async worker + `decode_mtp{,_async,_wait}`.
7. Speculative driver + `--mtp-head` flag.
8. Go cgo `DecodeMTP*` wrappers.
9. `format-patches`; build; bench acceptance vs our Go impl (target ~85-88% accept, +30-50%).

## Open questions

- Is gemma4 target upstream yet (decides step 1)?
- Edge (E2B/E4B) centroid head vs dense (26B/31B): port both or dense-only first? (dense-first is simpler; 31B is the bench target.)
- Async depth-2 worker now, or land a synchronous `decode_mtp` first and add the worker later? (sync-first de-risks correctness before concurrency.)
