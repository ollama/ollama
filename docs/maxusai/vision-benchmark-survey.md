# Vision benchmark survey: what exists elsewhere, and what we should adopt

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-08-02.

Companion to [vision-suite/README.md](vision-suite/README.md) — that documents the
home-grown ground-truth suite; this documents **everything else** that claims to verify
that a vision-language model behaves as expected, and picks the subset worth running per
release.

> **The one thing to take away:** nothing outside this repo tests the things our patches
> actually change. Upstream ollama asserts case-insensitive substrings on three cartoon
> images; llama.cpp greps for `new york` on one newspaper photo; the external harnesses
> (lmms-eval, VLMEvalKit) score real benchmarks but only through the endpoint, and their
> grounding scorers mangle two of our three models' coordinate dialects. The recommended
> battery is therefore **four cheap layers**, not one suite: a no-GPU image-token formula
> test, a ggml op check plus engine smoke test, our own ground-truth suite, and one external
> benchmark slice for absolute-number sanity.

## Contents

1. [Inventory](#1-inventory)
2. [How each complements our vision-suite](#2-how-each-complements-our-vision-suite)
3. [Official expected scores](#3-official-expected-scores)
4. [Recommended per-release regression battery](#4-recommended-per-release-regression-battery)
5. [The adapter](#5-the-adapter-extbenchpy)

---

## 1. Inventory

### 1.1 Upstream ollama — `integration/`

Present in our tree at [integration/](../../integration/); the fork predates upstream's
`4713800b` and so still carries the image-generation tests upstream deleted. Everything below runs against an **existing** server: set
`OLLAMA_TEST_EXISTING=1` and `OLLAMA_HOST=http://127.0.0.1:11435` and the harness never
spawns its own binary ([integration/README.md](../../integration/README.md),
`InitServerConnection` in `integration/utils_test.go`).

| Suite | Where | What it validates | How it asserts | Effort vs :11435 |
|---|---|---|---|---|
| Vision behavior cases (7) | `integration/vision_test.go`, registered in `reg_release_test.go` | multi-turn KV-cache reuse of image tokens, object counting, scene/cultural recognition, spatial reasoning, small-detail (glasses), multi-image in one message, plain description — on 3 embedded cartoon images (210×120 "The Ollamas" Abbey-Road parody, 400×250 docs image, 415×293 ollama.com screenshot); temp 0, seed 42 | case-insensitive **any-of substring** over streamed content (`containsExpectedResponse`); e.g. `vision-count` accepts `{"4","four"}`, `vision-detail` accepts `{"glasses","spectacles","eyeglasses"}` | **trivial** — one `go test -tags=integration,release -run TestVision ./integration/` per model, no build needed |
| `vision-text` (OCR) | `integration/llm_image_test.go` | reads the caption "The Ollamas" from the parody image | substring any-of `{"the ollam","ollamas"}` (truncated needle tolerates "the ollams") | **trivial** — same run |
| `vision-split-batch` | `integration/llm_image_test.go` | image embedding when a ~1.4 KB lorem-ipsum system prompt pushes the image across a batch boundary — upstream's only *mechanism* (not semantics) vision test | substring `"the ollam"` | **medium** — hardcoded to `qwen3.5:2b` and `t.Skip`s whenever `OLLAMA_TEST_MODEL` is set; needs a 3-line edit to use our models |
| Native-Jinja image markers | `integration/generate_jinja_test.go` | `/api/generate` with images routes through native chat templates: exact `ImageCount`, `[img-0]`/`[img-1]` markers present, no template sentinel leakage | **exact** field equality + required substring — the only non-fuzzy image assertions upstream has | **high / skip** — both tests `t.Skip` under `OLLAMA_TEST_EXISTING` (they need a harness-started server with `OLLAMA_GO_TEMPLATE=0`) |
| Image-gen → vision-judge | `integration/imagegen_test.go` (fork-local only) | z-image-turbo generates, `qwen2.5vl:3b` describes it | substring any-of over the description | **n/a** — skips on ROCm ("CUDA GPU is not available"); deleted upstream in [`4713800b`](https://github.com/ollama/ollama/commit/4713800b08b2ddf5e14acf8398953cf7b12f169b) |
| Library sweep | `integration/reg_library_test.go` | `vision-text` across ~200 library models | same substring | trivial with `OLLAMA_TEST_MODEL`, prohibitive without (~2.5 TiB of pulls) |

Notes that matter for us:

- **Only one of upstream's three default vision models is ours.**
  `releaseVisionModels = {nemotron3:33b, gemma4, qwen3.6:27b}`
  ([integration/reg_release_test.go:37](../../integration/reg_release_test.go)), and resolving those
  tags against `registry.ollama.ai` manifests (2026-08-02) gives:

  | upstream entry | resolves to | ours | match |
  |---|---|---|---|
  | `nemotron3:33b` | 28 GB, q4_K_M (`:33b` and `:33b-q4_K_M` share one manifest) | `nemotron3:33b-q4_K_M` | **exact** |
  | `gemma4` | `gemma4:e4b` — 9.6 GB / 128K (`latest`, `e4b`, `e4b-it-q4_K_M` all one digest) | `gemma4:31b-it-q4_K_M` — 20 GB / 256K | **no** |
  | `qwen3.6:27b` | 17 GB **dense** 27B | `qwen3.6:35b-a3b-q4_k_m` — 24 GB **MoE** | **no** |

  So `nemotron3:33b` is verbatim ours and carries **no known-flake entry** — upstream expects
  a clean pass, which makes any failure under our b9888+002 build a real signal. The other
  two rows test different architectures than we ship.
- Consequently, upstream's known-flake note that gemma4 "counts five animals in the Ollamas
  image instead of four" is an observation about **gemma4:e4b**, not about our 31B — it is
  *not* independent corroboration of the campaign's gemma4 counting results. (The skip table
  also matches model strings exactly, so `gemma4:31b` would run those cases regardless.)
- **Never run without `OLLAMA_TEST_MODEL`**: `PullIfMissing` would pull ~27 GB of weights
  we do not ship (`gemma4:e4b`, `qwen3.6:27b`, plus `qwen3.5:2b` for split-batch) onto the
  shared server.
- Each vision test preloads with `KeepAlive: 10s`, so it **evicts whatever the campaign has
  resident**. Check `/sys/class/kfd/kfd/proc/` and coordinate first.
- **No integration test uses the OpenAI-compatible `/v1` path for images at all** — verified
  by grep over `vision_test.go` and `llm_image_test.go`. Upstream *does* test `/v1` for
  audio (`openai-audio-transcription`, `openai-chat-audio` in `audio_test.go`), so the
  omission is specific to vision. Since OpenWebUI and ChatOllama drive our deployment through
  that path, **nobody upstream tests the endpoint our users actually hit.**

**What ollama's CI actually validates about model behavior: nothing.** Upstream `main` has
five workflows (`test.yaml`, `latest.yaml`, `release.yaml`, `test-install.yaml`,
`test-llamacpp-update.yaml`); grepping all five finds zero occurrences of `-tags=integration`,
`OLLAMA_TEST_EXISTING`, or `OLLAMA_TEST_MODEL`. The only Go test step is a bare
`go test -count=1 -benchtime=1x ./...` ([test.yaml:412](https://github.com/ollama/ollama/blob/main/.github/workflows/test.yaml)),
which excludes everything under `integration/` by build constraint — those files are never
even *type-checked* in CI. The proof is sitting on `main` right now: commit `4713800b`
(2026-07-28) deleted `integration/imagegen_test.go` but left
`integrationTestCase("image-generation", "", runImageGeneration)` at `reg_release_test.go:120`,
so `go test -tags=integration,release ./integration/` — the exact command in upstream's own
README — fails to compile with `undefined: runImageGeneration`, and has for five days.

Two consequences. First, treat upstream's vision expectations as *documentation of intent*,
not as a maintained contract. Second, the fork is not affected — our tree predates
`4713800b` and still has `imagegen_test.go`; `go vet -tags "integration release" ./integration/`
passes locally. If we ever rebase past that commit we inherit the break.

### 1.2 llama.cpp — mtmd battery and backend ops

| Suite | Where | What it validates | How it asserts | Effort |
|---|---|---|---|---|
| `mtmd/tests.sh` | [tools/mtmd/tests.sh @ b9888](https://github.com/ggml-org/llama.cpp/blob/b9888/tools/mtmd/tests.sh) — **byte-identical on master** | end-to-end image→text sanity for 23 vision + 6 audio models through `llama-mtmd-cli` at temp 0, flash-attn on/off; the image is `test-1.jpeg`, the NYT front page of 1969-07-21 ("Men Walk On Moon") | `grep -iq "new york" \|\| (grep -iq "men" && grep -iq "walk")` — no scoring, no thresholds | **trivial for a subset** (the patched b9888 `llama-mtmd-cli` is already built, see below); **medium** verbatim (needs its own CURL-enabled build + 30–50 GB of HF pulls) |
| `mtmd/tests/test-deepseek-ocr.py` | [same tree](https://github.com/ggml-org/llama.cpp/blob/b9888/tools/mtmd/tests/) | quantitative OCR fidelity vs `test-1-ground-truth.txt`; on master it specifically guards the **multi-tile dynamic-resolution** path (`test-1-positive.png`, expected CER 0.0000 ± 0.03) | **CER (jiwer) + chrF (sacrebleu)** after `rapidfuzz` crops the output to the ground-truth span; passes iff `CER ≤ hf_cer + tol` and `chrF ≥ hf_chrf − tol` — tolerance bands anchored to the HF reference implementation's measured scores | **low to borrow** (~60 lines of scoring, `pip install jiwer sacrebleu rapidfuzz`); medium to run as-is |
| `test-backend-ops` (UPSCALE subset) | [tests/test-backend-ops.cpp @ b9888 L8976–8986](https://github.com/ggml-org/llama.cpp/blob/b9888/tests/test-backend-ops.cpp) | numerical correctness of `GGML_OP_UPSCALE` per backend — **including `GGML_SCALE_MODE_BICUBIC`**, which is exactly what patch 002 calls (`resize_position_embeddings(GGML_SCALE_MODE_BICUBIC)`) | element-wise **NMSE < 1e-7** vs the CPU backend | **low–medium** (~1–2 h, mostly compile) — see below |
| `server/tests/unit/test_vision_api.py` | [b9888](https://github.com/ggml-org/llama.cpp/blob/b9888/tools/server/tests/unit/test_vision_api.py) | the OpenAI-compatible `/chat/completions` image path: `image_url` as https URL, `data:` URI and raw base64; malformed image, 404 URL, non-image bytes; `multimodal` capability flag | regex on choice content (`"(cat)+"`) against a CIFAR-10 toy model, plus HTTP error assertions | **low** — the harness spawns its own server, but the 7-case request corpus ports directly onto our `/v1` |

Key facts for our situation:

- **The roster does not include any of our models.** `tests.sh` has no `nemotron_v2_vl`
  entry at all (even though `tools/mtmd/models/nemotron-v2-vl.cpp` exists at b9888),
  gemma-4 appears only as the tiny E2B sibling, and there is no Qwen3.5/3.6 MoE. The
  upstream battery would not have caught either the nemotron 256-token cap or the b10091
  payload regression.
- **Nothing above runs in llama.cpp CI** except `test_vision_api.py`. A GitHub code search
  over `.github/workflows` finds zero references to `mtmd`, and `test-backend-ops` runs
  only on Vulkan/WebGPU/WASM/CPU — **never ROCm, never gfx1151**. Running the UPSCALE
  subset locally is additive validation, not duplicated work.
- **The BICUBIC dependency is genuinely asserted on ROCm, not skipped.** Verified at b9888
  (= commit `cb295bf5`): `ggml/src/ggml-cuda/upscale.cu:154–256` implements a real 4×4
  separable bicubic convolution (`upscale_f32_bicubic`), the HIP backend compiles it
  unchanged (`ggml/src/ggml-hip/CMakeLists.txt:63` globs `../ggml-cuda/*.cu`, and
  `upscale.cu` contains no `__HIP__` gating), and `ggml_backend_cuda_device_supports_op`
  returns an unconditional `true` for `GGML_OP_UPSCALE` (`ggml-cuda.cu:5549–5557`) — so
  `test-backend-ops` cannot skip it. Two caveats worth knowing: `supports_op` does *not*
  inspect the scale mode, and the kernel dispatch in `upscale.cu:281–291` has no terminal
  `else`, so an unimplemented mode would silently no-op with an uninitialized `dst` rather
  than fall back to CPU (harmless for patch 002, which uses BICUBIC); and
  `GGML_SCALE_FLAG_ANTIALIAS` is honoured for BILINEAR only, so bicubic downscales are
  un-antialiased on this backend.
- **How the fork gets llama.cpp:** FetchContent, not vendoring —
  [llama/server/CMakeLists.txt](../../llama/server/CMakeLists.txt) declares
  `GIT_TAG` from the top-level `LLAMA_CPP_VERSION` file with a `PATCH_COMMAND` that applies
  every `llama/compat/*.patch` idempotently. Because `LLAMA_BUILD_TOOLS=ON`, **every fork
  build already produces `llama-mtmd-cli` and `llama-mtmd-debug`** as a side effect. The
  b9888+001+002 gfx1151 binaries exist at
  `/opt/github/MaxusAI/ollama-0321/build/llama-server-rocm_v7_2/bin/`. Two caveats: fork
  builds set `LLAMA_CURL=OFF`, so `-hf` downloads don't work (pass local blobs with
  `-m <blob> --mmproj <blob>` — the compat layer accepts ollama monolithic blobs); and
  the fork force-sets `LLAMA_BUILD_TESTS=OFF`, so `test-backend-ops` needs a separate
  **vanilla** b9888 build. That's fine and in fact preferable: neither fork patch touches
  `ggml/`, so vanilla ggml results transfer to the shipped engine exactly.
- Version-pin note: `LLAMA_CPP_VERSION` on this branch reads `b10091` (main took the
  upstream bump in `cc626766`). The **b9888** pin that the canonical image
  `maxusai-ollama:0.32.1-rocm-dynres-a4788474` is built from lives in the separate
  `/opt/github/MaxusAI/ollama-0321` checkout (`release/0.32.1-dynres`). See
  [amd-upgrade-gate.md](amd-upgrade-gate.md).

### 1.3 External harnesses

| Harness | Can it hit :11435? | Coverage | Scoring | Effort |
|---|---|---|---|---|
| **lmms-eval** ([repo](https://github.com/EvolvingLMMs-Lab/lmms-eval)) | **Yes** — `--model openai_compatible`, env `OPENAI_API_BASE=http://127.0.0.1:11435/v1` + any dummy `OPENAI_API_KEY`; base64 images via the OpenAI SDK | OCRBench, DocVQA, InfoVQA, ChartQA, TextVQA, MMMU, CountBenchQA, RealWorldQA, MMBench, RefCOCO/+/g | per-task rule scorers (contains / ANLS / relaxed-acc / exact / IoU); judge needed **only** for `mmbench_en_dev` (use `mmbench_en_dev_static` instead) | **low** — pip install (heavy torch wheels but a pure client at runtime), `--limit 50` for slices, `--include_path` to fork task YAMLs. Raise the default **10 s timeout**, cap the default **32-way concurrency** |
| **VLMEvalKit** ([repo](https://github.com/open-compass/VLMEvalKit)) | **Yes** — `--model lmdeploy` with `LMDEPLOY_API_BASE=http://127.0.0.1:11435/v1/chat/completions` (a generic OpenAI-SDK wrapper despite the name; temp 0 default), or a `--config` JSON pinning `LMDeployAPI` | same set plus OCRBench v2, ScreenSpot, GroundingME | rule-based for the VQA family; judge-assisted MCQ extraction with a **documented exact-match fallback** when no key is set | **medium** — heavier install, no `--limit` (use the `*_MINI` datasets or truncate the TSV); datasets are self-hosted TSVs in `~/LMUData`, **no HF auth** |
| **mistral-evals** ([repo](https://github.com/mistralai/mistral-evals)) | **Nearly** — raw POST to `{url}/v1/chat/completions` with base64 data URIs, temp 0 hardcoded; but `_wait_till_healthy()` GETs `/health`, which ollama does not serve, and crashes on the non-JSON 404. Two-line patch | ChartQA, DocVQA, MathVista, MMMU, VQAv2 | ANLS, relaxed correctness ±5 % in two extraction variants, VQA-match, MCQ acc | **low–medium** — tiny deps (no torch), but no `--limit` flag |
| **lm-evaluation-harness** | **No** | multimodal is an in-process prototype (`hf-multimodal`, `vllm-vlm`); `local-chat-completions` rejects image tasks — [issue #3302](https://github.com/EleutherAI/lm-evaluation-harness/issues/3302). The README itself defers to lmms-eval | — | n/a for vision |
| **OpenCompass** | **No** | deprecated its multimodal arm on 2024-04-26 and moved it to VLMEvalKit | — | n/a |
| ollama-community suites | native API, but no ground truth | [ollama-grid-search](https://github.com/dezoito/ollama-grid-search) (951★) is human-inspection A/B; the rest are tokens/sec throughput | — | **not an eval** |
| **Pure-dataset DIY** | **Trivially** — we own the client | any HF benchmark, driven through our own runner | our own scorers | **trivial** — see [§5](#5-the-adapter-extbenchpy) |

**No external suite validates vision accuracy against ollama natively.** The
OpenAI-compatible `/v1` shim is the only interoperability point, and it accepts images
**only** as base64 `data:` URIs inside `image_url` parts (remote URLs are not fetched) —
every harness above already encodes that way.

---

## 2. How each complements our vision-suite

### What ours covers that none of them do

1. **Coordinate-dialect-aware bbox scoring.** Models emit the convention they were trained
   on regardless of what the prompt asks. Our scorer searches
   {pixel, norm-1000} × {xyxy, yxyx} and accepts the keys `bbox`, `bbox_2d`, `box_2d`, then
   reports the winning dialect alongside IoU. The external grounding scorers are strictly
   weaker: **lmms-eval's `refcoco_bbox_rec` hard-requires normalized 0–1 xyxy** and parses
   anything else to `[0,0,0,0]`, and **VLMEvalKit auto-detects the *scale* but not the
   *order*** — neither handles gemma4's Gemini-style `box_2d` yxyx. See
   [§3.1](#31-the-grounding-trap) for what this does to the numbers.
2. **A think:true × think:false axis on every cell.** No external harness runs both modes as
   a matrix, and the finding that `think:true` + `format:"json"` yields an *empty* response
   on nemotron3 and qwen3.6 is exactly the kind of failure a substring-matching suite scores
   as "wrong answer" rather than "no answer".
3. **The `format:"json"` grammar-constrained path.** Every external harness asks for free
   text. Our suite runs structured output, which is what our applications use — and
   [ADR 0002](adr/0002-deferred-format-constraining.md) exists because that path had a real bug.
4. **Both ollama endpoints.** `ENDPOINT=generate|chat` covers `/api/generate` and
   `/api/chat`; external harnesses only speak `/v1`. Given the generate-side think+format
   fix, endpoint choice is a variable that changes results.
5. **Image-token budget as a first-class measurement.** `measure.py` reads
   `prompt_eval_count` across a geometry ladder — it measures the mechanism, not the output.
6. **Known ground truth at controlled font sizes.** Synthetic scenes let us vary exactly one
   thing (14px serial vs 22px line item) and attribute a failure to fine-text resolution.

### What they cover that ours does not

1. **Absolute, comparable numbers.** Our scores are self-referential — good for
   release-over-release, useless for "is 0.78 IoU *good*?". One OCRBench slice answers that.
2. **Real photographic and scanned inputs.** Every one of our images is synthetic; DocVQA,
   OCRBench and RealWorldQA are scans and photos with the noise, skew and compression our
   generator never produces.
3. **Statistical power.** Three images versus 1,000 OCRBench items. Our suite can't
   distinguish a 2-point regression from noise; a 500-item slice can.
4. **The engine below ollama.** llama.cpp's `mtmd` tests and `test-backend-ops` exercise the
   projector and the ggml op directly, so they can tell "the model is bad" apart from
   "ollama's plumbing is bad" — which is precisely the ambiguity the b10091 investigation hit.
5. **Multi-turn image KV-cache reuse and multi-image-in-one-message.** Upstream's
   `vision-multiturn` and `vision-multi-image` cover request shapes ours doesn't.
6. **`/v1` conformance edge cases.** llama.cpp's `test_vision_api.py` covers malformed
   images, non-image bytes and data-URI variants — error paths, not accuracy.

### A gap this survey closed

`measure.py` read its geometry ladder from a `testimgs/` directory that **nothing in the repo
generated** (`gen_scenes.py` writes `visimgs/`), so the token-budget protocol was not
reproducible from a clean checkout — and because its `open()` sits outside the `try`, a missing
file surfaced as an uncaught traceback rather than a recorded error. Added
[vision-suite/gen_geoms.py](vision-suite/gen_geoms.py) to render the eight geometries
deterministically. `measure.py` still reports rather than asserts; the pass/fail gate is the Go
test in [§4](#4-recommended-per-release-regression-battery).

---

## 3. Official expected scores

### 3.1 The grounding trap

This deserves its own section because it is the single largest way an external harness will
lie to us, and because grounding is what our own scorer is built around.

**Why each model's dialect is what it is** — the asymmetry is in the vendor documentation,
not in the models:

| model | dialect | vendor contract |
|---|---|---|
| qwen3.6 | `bbox_2d`, **xyxy**, norm-1000 | **documented** — the [Qwen3-VL 2d_grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb) states the coordinate system "has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000", and its plotting code pins slot order to xyxy. The 3.6 card itself publishes only the score |
| gemma4 | `box_2d`, **yxyx**, norm-1000 | **documented** — [ai.google.dev](https://ai.google.dev/gemma/docs/capabilities/vision/image): "coordinates are expressed as normalized values relative to a 1000x1000 grid", format `[y_min, x_min, y_max, x_max]`. This is the Gemini family convention inherited wholesale, and Google is the **sole yxyx holdout** |
| nemotron3 | self-chosen key; norm-1000 under `think:false`, **coarse pixel** under `think:true` | **undocumented** — the [Nemotron-3-Nano-Omni card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) specifies no bbox format at all; grounding is evidenced only by `CVBench2D 83.95` |

That last row is the important one: NVIDIA never specified a grounding contract, so the model
has no canonical convention to fall back on. Its dialect must be **detected per response**,
not assumed — which is exactly why our scorer reports the winning dialect instead of
presuming one.

**What the external harnesses would report.** Both harness scorers were read at source, and a
decode-ceiling simulation was run: a *hypothetical perfect grounder* emitting each dialect,
pushed through each harness's real parse/scale/score code against 200 real RefCOCO val
ground-truth boxes. These are ceilings — a real model cannot beat them.

| dialect emitted by a **perfect** grounder | lmms-eval mean IoU | lmms-eval ACC@0.5 | VLMEvalKit mean IoU | VLMEvalKit P@1 |
|---|---|---|---|---|
| norm-1000 xyxy (**qwen3.6**, **nemotron3 think:false**) | **0.0000** | **0.00 %** | 0.9541 | 95.5 % |
| norm-1000 yxyx (**gemma4**) | **0.0000** | **0.00 %** | 0.2193 | 13.0 % |
| pixel xyxy (**nemotron3 think:true**) | **0.0000** | **0.00 %** | 0.0534 | 0.0 % |
| normalized 0–1 xyxy (harness-native) | 1.0000 | 100 % | 1.0000 | 100 % |

Three conclusions follow, and each is a trap of a different shape:

1. **lmms-eval returns exactly 0.0 for all three of our models.** Not "low" — zero, on all
   seven metrics. Its prompt asks for 0–1 floats and its regex happily parses
   `[74, 361, 324, 808]` as such, so a norm-1000 answer is a *silent scale error*, not a
   parse failure: intersection with a 0–1 ground-truth box is identically empty. The fix
   exists 40 lines away in the sibling `screenspot` task (`if max(pred) > 1: pred = [x/1000 …]`)
   and was simply not ported to `refcoco`. There is no YAML, doc or CLI scale option —
   GitHub code search for `box_scale`/`scale_bbox` returns zero results. **Publishing a raw
   lmms-eval RefCOCO number for any of our models would report a false catastrophic
   regression on a model with an official 92.0.**
2. **VLMEvalKit is trustworthy for qwen3.6 only, and misleading for gemma4.** Its
   `_to_absolute` auto-detects the *scale* (`≤1.5` → 0–1, `≤1000` → norm-1000, else pixel) but
   hard-codes xyxy — `coords[0::2]` is always scaled by width — with no order parameter and no
   yxyx branch. Worse, its min/max swap *masks* the damage: a transposed box never errors, it
   just scores wrong. gemma4 lands at **13 % P@1 against a true ~0.78 IoU** — high enough to
   read as a genuinely weak grounder, low enough to be believed. That is the most dangerous
   number in this document.
3. **For nemotron3, VLMEvalKit inverts the think-mode verdict.** Our own measurement is that
   `think:true` *improves* nemotron's grounding (center-hits 0–1 → 5/6). But `think:true`
   switches it into pixel space, and since RefCOCO images are COCO images (≤640 px on the long
   side), every pixel prediction has `max_val ≤ 1000` and is misrouted into the norm-1000
   branch — divided by 1000, collapsing a correct box to ~0.1 % of the image. **The harness
   scores the quality-improving configuration as a total failure.** Any think-mode A/B run
   through VLMEvalKit will conclude the opposite of the truth.

There is also a ceiling artifact worth budgeting for: 6.5 % of RefCOCO ground-truth boxes
touch the right or bottom edge, so their exact norm-1000 encoding exceeds 1000 and falls past
the `≤1000` branch. It is a cliff, not a gradient — `1000` scores, `1001` scores zero — costing
~4.5 pp for any norm-1000 model even when everything else is correct.

**Therefore:** use the external harnesses as *transport and dataset plumbing*, and score
grounding with our own dialect-aware scorer. Concretely, run lmms-eval with `--log_samples`,
discard its reported metrics, and re-score the dumped generations — or skip the install and use
`extbench.py refcoco`, which pulls the same dataset and applies the 3×2 dialect search directly.

### 3.2 Published numbers

Model identities first, since the ollama tags do not name the upstream releases:

| ollama tag | official model | sources |
|---|---|---|
| `nemotron3:33b` (= `:33b-q4_K_M`, one manifest) | **NVIDIA Nemotron-3-Nano-Omni-30B-A3B** — omni MoE, ~30B total / ~3B active | [HF card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16), [tech report arXiv:2604.24954](https://arxiv.org/abs/2604.24954) |
| `gemma4:31b` | **google/gemma-4-31B-it** — dense 30.7B flagship | [HF card](https://huggingface.co/google/gemma-4-31B-it), [tech report arXiv:2607.02770](https://arxiv.org/abs/2607.02770) |
| `qwen3.6:35b-a3b` | **Qwen/Qwen3.6-35B-A3B** — natively multimodal MoE, 35B/3B active, thinking on by default | [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |

**Nemotron-3-Nano-Omni is the only one of the three that publishes both reasoning modes** —
a direct official analogue of our `think` axis. Critically, NVIDIA states these were measured
with **VLMEvalKit on a vLLM backend**, so the harness is known.

> **Read the tech report, not the model card.** The two disagree in scope, and the card is the
> one people check. The HF card's evaluation table contains only OCRBench**V2** (EN) 67.04,
> CharXiv-Reasoning 63.6, MathVista-mini 82.8, OCR-Reasoning 54.14, CVBench2D 83.95,
> MMLongBench-Doc 57.5, OSWorld 47.4, Video-MME 72.2 — reasoning-mode only, with **no
> OCRBench v1, DocVQA, ChartQA, TextVQA, AI2D, MMMU or RefCOCO rows at all** (DocVQA and
> ChartQA appear on that page only in the *training-data* table). Everything below comes from
> **Table 7 of the tech report**, verified directly against
> [arxiv.org/html/2604.24954v2](https://arxiv.org/html/2604.24954v2). Note also that OCRBench
> v1 (scored /1000) and OCRBench v2 (0–100, EN/ZH split) are different benchmarks — never put
> them in one column.

| benchmark | reasoning OFF | reasoning ON | locally reproducible? |
|---|---|---|---|
| MMMU (val) | 55.2 | **70.8** | yes — `mmmu_val` (900 items, judge-free) |
| OCRBench v1 | **88.3** | 86.6 | yes — `extbench.py ocrbench` |
| OCRBench v2 EN / ZH | 65.8 / 52.0 | 67.0 / 52.7 | yes (VLMEvalKit) |
| **ChartQA (test)** | **89.9** | 90.3 | **yes, no substitution** — ChartQA ships public test answers |
| DocVQA (test) | 93.3 | **95.6** | only via `docvqa_val` — see the val↔test note below |
| InfoVQA (test) | 83.6 | 86.8 | only via `infovqa_val` (submission-gated test) |
| TextVQA (val) | **85.1** | 81.0 | yes |
| AI2D (test) | 88.5 | 88.5 | yes, judge-free |
| MathVista-mini | 71.9 | **82.8** | yes, but judge-required |
| CharXiv RQ | 49.1 | **63.6** | yes, but judge-required (GPT-4o) |
| MMLongBench-Doc | 46.1 | **57.5** | yes — judge-free in lmms-eval, judge-based in VLMEvalKit (**not comparable**) |
| CV-Bench (2D) | 84.2 | 84.0 | yes — use `cv_bench_2d`, not `cv_bench` |
| **RefCOCO** | 80.6 | **90.5** | yes — but see [§3.1](#31-the-grounding-trap) |
| ScreenSpot / -v2 / -Pro | 90.3 / 93.4 / 59.3 | 89.3 / 92.8 / 57.8 | yes, judge-free |

Two things to draw from this. First, **the direction of the think effect is
benchmark-dependent**: reasoning is worth +15.6 on MMMU and +14.5 on CharXiv, but *costs* 1.7
on OCRBench, 1.5 on ScreenSpot-Pro and 4.1 on TextVQA. That is a falsifiable prediction — if
our OCRBench slice shows think-on *helping* nemotron, our think path differs from NVIDIA's.
Second, **nemotron does publish RefCOCO** (80.6 → 90.5 with reasoning), which contradicts the
impression its HF card gives; the card publishes only `CVBench2D`. Since that 90.5 was measured
through VLMEvalKit's scorer, NVIDIA's reference implementation must emit norm-1000 — our
observation that `think:true` flips it to pixel space under llama.cpp is therefore a
*fork-path* behavior worth tracking, not a property of the weights.

**ChartQA is the best direct-comparison anchor we have.** It is the only benchmark where a
vendor's official *test* number is reproducible locally with no split substitution, and it is
judge-free, 70 MB, and already wired into `extbench.py`.

For DocVQA, the val↔test substitution is defensible with a measured offset: nemotron's own
report gives DocVQA **(Test) 95.6** in Table 7 and DocVQA **(Val) 95.3** in Table 11 — same
model, same harness, so **Δ(test − val) ≈ +0.3–0.4 ANLS**. No comparable same-model pair exists
for InfoVQA or MMMU, so do not assume an offset there.

**Gemma 4 publishes only five vision benchmarks, all thinking-ON**, and none of them overlap
the cheap rerunnable set. The report's own resolution ablation is the interesting part:

| benchmark | 31B @ 1120 vision tokens (Table 6) | 31B @ 280 tokens (Table 12) | Δ |
|---|---|---|---|
| MMMU Pro | 76.9 | 75.8 | −1.1 |
| MATH-Vision | 85.6 | 83.4 | −2.2 |
| MedXPertQA MM | 61.3 | 60.7 | −0.6 |
| **InfographicVQA** | **92.0** | **82.8** | **−9.2** |
| OmniDocBench 1.5 (↓ lower better) | 0.131 | 0.201 | −0.070 |

This is official, vendor-measured evidence that **document benchmarks degrade far harder than
reasoning benchmarks when the image-token budget shrinks** — the same mechanism as our nemotron
256-token cap and the b10091 payload regression, confirmed by the model's own authors on their
own model. It is also a ready-made experimental design: Table 6 vs Table 12 is an
officially-sanctioned 1120-vs-280 vision-token ablation we can replicate through our own
`--image-max-tokens` knob.

**There is no official non-thinking vision number for Gemma 4 anywhere.** Verified against both
arXiv v1 and v2: every vision table is captioned "(thinking)", and the report's only "without
thinking" table (Table 9) is text-only long-context. The Qwen card's Gemma4-31B column does not
state its mode either, but its MMMU-Pro cell is byte-identical to Google's thinking@1120 value,
which implies the whole column is thinking-mode. **So our `think:false` gemma4 vision cells have
no vendor anchor at all** — either run gemma4 in thinking mode to make Table 6 apply, or label
those cells explicitly unanchored. Also note Google never states whether its InfographicVQA
92.0 is val or test, while nemotron explicitly labels its 83.6/86.8 as test; do not table the
two side by side without flagging that.

**Qwen3.6-35B-A3B publishes the broadest table, and is the only one with grounding numbers**
(thinking presumed on; no non-thinking scores published): MMMU 81.7, MMMU-Pro 75.3,
MathVista-mini 86.4, RealWorldQA 85.3, MMBench-EN-dev-v1.1 92.8, HallusionBench 69.8,
OmniDocBench 1.5 = 89.9, CharXiv-RQ 78.0, CC-OCR 81.9, AI2D 92.7, and
**RefCOCO (avg) 92.0**, ODinW13 50.8, EmbSpatialBench 84.3, RefSpatialBench 64.3. It does
*not* publish OCRBench v1, DocVQA, ChartQA or CountBenchQA — Qwen replaced the legacy doc/OCR
set with OmniDocBench and CC-OCR.

That RefCOCO 92.0 is the single most useful number in this document: it predicts near-ceiling
grounding, so if our scorer reports qwen3.6 far below it, **suspect the harness or the dialect
decode before suspecting the model** — which is exactly the failure mode [§3.1](#31-the-grounding-trap)
quantifies.

#### Which benchmarks are actually anchored across models

A benchmark is only useful for "are we where this model should be?" if more than one of our
models has a published number *and* it can be scored locally without a judge LLM.

| benchmark | models with published numbers | judge? | verdict |
|---|---|---|---|
| **AI2D (test)** | **3** — 88.5 / 89.0\* / 92.7 | **no** | **best cross-model anchor.** 3,088 items, rule-based letter extraction |
| **ChartQA (test)** | 1 (nemotron 89.9/90.3) | no | **best exactness anchor** — official *test* number reproducible with no substitution |
| MathVista-mini | **3** — 71.9·82.8 / 79.3\* / 86.4 | **yes** | strongest anchor on paper, but needs a judge LLM in both harnesses |
| CharXiv RQ | **3** — 49.1·63.6 / 67.9\* / 78.0 | **yes** (GPT-4o) | same trade-off |
| OCRBench v1 | 1 (nemotron, both modes) | no | cheapest run; the only both-modes OCR anchor |
| MMMU-Pro | 2 (gemma4 76.9, qwen 75.3) | no | judge-free; note MMMU-Pro's *test* split ships answers and **is** locally scoreable, unlike MMMU test |
| CC-OCR | 2 (75.7\* / 81.9) | no | judge-free, 7,058 items |
| HallusionBench | 2 (67.4\* / 69.8) | lmms-eval **yes**, VLMEvalKit optional | use VLMEvalKit's `exact_matching` mode |
| RefCOCO | 2 (nemotron 80.6/90.5, qwen 92.0) | no | see [§3.1](#31-the-grounding-trap) — harness scorers mangle our dialects |
| OmniDocBench | 2 nominally | no | **unusable** — harnesses ship v1.0, both vendors report v1.5, and Google's edit-distance metric is not convertible to Qwen's 0–100 score |
| RealWorldQA, CountBenchQA, DocVQA, TextVQA | ≤1 | no | fine as drift detectors, not as absolute anchors |

\* = measured by the Qwen team, not the model's own vendor.

**There is no public leaderboard to fill the remaining holes.** The OpenVLM Leaderboard is the
obvious candidate and it does not have our models: its backing `OpenVLM.json` was last modified
2026-02-27 (285 models, newest evaluation dated 2026/02/13), and greps for `qwen3.6` and
`nemotron` return nothing while the only Gemma entries are Gemma 3 and PaliGemma — both our 2026
releases post-date the last refresh by months. Its schema also has **no DocVQA and no
CountBenchQA column at all**, so it could not have filled those holes even if current.
VLMEvalKit registers Gemma-4 (`vlmeval/config.py`, class `vlm.Gemma4`) but has **no Qwen3.6
entry**, and lmms-eval has neither. Consequently **no third-party measured OCRBench, DocVQA,
ChartQA, TextVQA or CountBenchQA number exists for gemma4:31b or qwen3.6 at all** — for those
cells, a first local baseline *is* the reference, which is the main argument for Layer 4 being
part of the battery rather than a one-off.

Two traps in that table are worth stating plainly. **OmniDocBench looks like a two-model anchor
and is not** — different benchmark version, incompatible metrics. And **MMLongBench-Doc numbers
are harness-dependent**: lmms-eval scores it rule-based, VLMEvalKit uses GPT-4 extraction, so
the two do not produce comparable numbers and nemotron's 46.1/57.5 only means something against
the VLMEvalKit path.

#### Caveats that apply to every row

1. **Precision — the quantization penalty is small, and smaller than it looks.** All official
   numbers are BF16; we serve q4_K_M. Two pieces of evidence bound the gap:
   - NVIDIA's own 4-bit table for *this exact model*: across 9 multimodal benchmarks, NVFP4
     costs a mean **−0.38** vs BF16 (worst cell −1.2 on Video-MME; OCRBenchV2 −0.03; CVBench2D
     actually +1.07). Measured in non-reasoning mode; NVFP4 ≈ 4.98 bpw.
   - A weight-only W4 study on LLaVA-Next-8B ([arXiv:2404.14047](https://arxiv.org/abs/2404.14047),
     Table 12), which quantizes the LLM and leaves the vision encoder alone — the same shape as
     our setup: AI2D −1.0 to −1.3, DocVQA −0.8 to −1.2, ChartQA −1.2 to −3.5, MMBench −1.1 to
     −2.1. Naive RTN costs roughly 3× a calibrated method on ChartQA.

   **Why that transfers to us:** ollama never quantizes the projector. `createModel` routes only
   `application/vnd.ollama.image.model` and the draft layer to `quantizeLayer`
   ([server/create.go:739–761](../../server/create.go)); `application/vnd.ollama.image.projector`
   is a separate layer that is never touched, and published GGUFs ship `mmproj` at BF16/F16/F32
   only. So our vision tower stays at source precision, exactly as in NVIDIA's recipe.
   **Working band: expect ≤2–3 points from quantization on rule-scored vision benchmarks.**
2. **Do not confuse W4A4 results with ours.** Weight+activation 4-bit collapses VLMs
   catastrophically (Qwen2.5-VL-7B OCRBench 83.8 → 0.2–13.2 under naive W4A4). GGUF Q4_K_M is
   weight-only with FP16 activations. Those papers are a warning against a *different* method,
   not a tolerance for ours. Relatedly, unsloth's `UD-Q4_K_M` is not vanilla Q4_K_M — if a blob
   was built with `ollama create --quantize q4_K_M`, published UD numbers do not describe it.
3. **Thresholds should still be self-anchored.** Set the pass band from a first q4_K_M baseline
   on our own hardware and gate on release-over-release drift; use official BF16 numbers as
   direction-only bounds. With the band above, a 2–3 point gap is expected and uninformative,
   while a 20-point gap means preprocessing or token budget — not quantization.
3. **Split and mode mismatches.** Nemotron's DocVQA/ChartQA are **test** split; locally only
   `docvqa_val` scores without an external submission. Gemma 4's numbers are thinking-ON at
   max resolution. Comparing a val-split, q4_K_M, think-off run against a test-split, BF16,
   think-on number will manufacture phantom regressions of several points.
4. **Preprocessing dominates quantization.** Vendors evaluate with their reference vision
   preprocessing; llama.cpp's `clip.cpp` reimplements it. Gemma's own −9.2 resolution ablation
   is larger than any quantization effect measured on any of these models.
5. **Metric traps.** OmniDocBench 1.5 is an edit distance for Google (0.131, lower better) but
   a 0–100 score for Qwen (89.9) — the two are not comparable, and `1 − 0.131 ≠ 0.801`.

---

## 4. Recommended per-release regression battery

Four layers, cheapest first. Layers 1–2 need **no GPU** and run in seconds; they are the ones
that would have caught the failures we have actually had. Layer 3 is the existing suite.
Layer 4 is the new one, and the only layer that produces a number comparable to the outside
world.

### Layer 1 — image-token formulas (no GPU, ~6 ms)

```bash
go test ./llm/ -run TestImageTokensForSize -count=1
```

`TestImageTokensForSize` ([llm/llama_server_test.go:1815](../../llm/llama_server_test.go))
pins the exact per-arch, per-geometry image-token counts that `ImageTokensForSize`
([llm/llama_server.go:1238](../../llm/llama_server.go)) replicates from the C++ preprocessor —
including `nemotron_h_omni` 640×480 → 302, 1920×1080 → 2042, 3000×2000 → 3292, and the
ceiling-exact 2048×1664 → 3330. **This is the tripwire for patch 002.** A silent fall-back to
the 256-token path changes these numbers, and a substring-matching smoke test would never
notice. It costs nothing, so run it on every build.

Its limitation is that it pins the *Go-side replication*, not the *served payload*. Layer 2
closes that.

**Why `measure.py` is not this layer.** Its protocol is right — text-only `prompt_eval_count`
baseline, then a delta per geometry — but it *reports* and never *asserts*: every result is
printed and the script exits 0 whether nemotron returns the dynamic ladder or a flat 256. It
also cannot run from a clean checkout (it reads `testimgs/`, which nothing generates, and its
`open()` sits outside the `try`, so a missing file is an uncaught traceback rather than a
recorded error), and its `image_max_tokens` probe is a *Runner* option, so that one request
forces a full reload of a 25.7 GiB blob. Treat it as the manual protocol it is; use the Go
test as the gate.

**A payload-level fingerprint that costs no inference.** The already-built
`llama-mtmd-debug -p preproc` dumps the preprocessor's output geometry without decoding:
patched, it prints `entry 0 has nx=1568, ny=1568` (tokens = (nx/32)·(ny/32)); unpatched, it
prints `nx=512, ny=512` for *every* input size, because `fixed_size` squashes everything to
512². The same invocation also prints, at load time,
`image_min_pixels: 262144 (custom value)` / `image_max_pixels: 3407872 (custom value)` — lines
that are gated on `> 0` and therefore **do not appear at all** on an unpatched payload, since
that branch never calls `set_limit_image_tokens`. Two independent signatures from one command.
Use `-ngl 0 --no-warmup --no-mmproj-offload` to keep it off the GPU. Caveats: it still loads
the LLM (pass `-m <blob>`), and `-n` only makes square images, so it cannot reproduce the
aspect-ratio rows or see the `+2` marker tokens.

**What no token-count check can catch.** Token counts prove patch hunks 1 and 3 (the budget
and the `dyn_size` dispatch). They say nothing about hunk 2, the
`resize_position_embeddings(GGML_SCALE_MODE_BICUBIC)` call — and since the HIP backend claims
`GGML_OP_UPSCALE` unconditionally, a numerically wrong bicubic would **not** fall back to CPU
and **would not move any token count**. It would silently corrupt position embeddings. That is
precisely why Layer 2's `test-backend-ops` run and Layer 3's quality suite are not optional
extras: they cover a failure mode Layer 1 is structurally blind to.

### Layer 2 — ggml op + engine smoke (no server)

```bash
# a) BICUBIC correctness on gfx1151 — the op patch 002 depends on
./build/bin/test-backend-ops test -o UPSCALE -b ROCm0
```

22 UPSCALE cases, 8 of them BICUBIC, each asserted at NMSE < 1e-7 against the CPU backend.
Build once from a **vanilla** b9888 checkout (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`,
target `test-backend-ops`); neither fork patch touches `ggml/`, so vanilla results transfer
exactly, and the fork build force-sets `LLAMA_BUILD_TESTS=OFF` anyway. Upstream CI never runs
this on ROCm, so it is genuinely new information.

```bash
# b) engine-level vision smoke against our own blobs, bypassing ollama entirely
llama-mtmd-cli -m <blob> --mmproj <blob> --image test-1.jpeg \
  -p "what is the publisher name of the newspaper?" --temp 0 -n 128
```

Using the already-built `/opt/github/MaxusAI/ollama-0321/build/llama-server-rocm_v7_2/bin/llama-mtmd-cli`
(b9888+001+002, gfx1151) with llama.cpp's
[test-1.jpeg](https://raw.githubusercontent.com/ggml-org/llama.cpp/b9888/tools/mtmd/test-1.jpeg)
and its pass rule (`"new york"` or `"men"`+`"walk"`). The compat layer accepts ollama
monolithic blobs, so this runs the *exact weights we serve* through the *exact engine we
ship*, with the HTTP/runner layer removed. When a regression appears, this is what tells you
which side of the boundary it is on. Note the fork binary has `LLAMA_CURL=OFF`, so pass local
blobs — `-hf` will not work.

### Layer 3 — our ground-truth suite (unchanged)

```bash
RESTART_CMD="docker restart <test-container>" \
  MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m" \
  ./run_grid.sh http://127.0.0.1:11435 <release-tag>
```

Still the only layer that scores bboxes in every dialect, runs both think modes, exercises
`format:"json"`, and covers both ollama endpoints. Keep it as the primary gate.

### Layer 4 — one external benchmark slice (new)

```bash
LIMIT=200 THINK=false SLEEP=1 \
  python3 docs/maxusai/vision-suite/extbench.py http://127.0.0.1:11435 <release-tag> \
    nemotron3:33b-q4_K_M ocrbench
```

**OCRBench is the right first external benchmark**: 1,000 items at 0.07 GB, ungated,
judge-free contains-match scoring, and it is the *only* rerunnable benchmark for which one of
our models has published both-think-modes numbers (nemotron3: 88.3 off / 86.6 on). A 200-item
slice takes minutes and yields a number comparable to a model card. Add `countbenchqa` (491
items, 0.02 GB) when you want a counting check, and `refcoco` when you want to cross-validate
the bbox scorer against external ground truth.

### What to skip, and why

| Candidate | Verdict |
|---|---|
| upstream `integration` vision tests | **Optional, `nemotron3:33b` only.** That one tag matches ours exactly and has no flake entry. The other two defaults are different models, the assertions are coarse substrings on cartoons, and `KeepAlive: 10s` evicts campaign models. Low value, non-zero cost. |
| lmms-eval / VLMEvalKit full installs | **Not per-release.** Several GB of wheels to reach benchmarks `extbench.py` already reaches over the same endpoint. Worth installing once, for a *full* OCRBench/RefCOCO comparison against published leaderboard numbers, not per release. |
| lm-evaluation-harness, OpenCompass | **Never** — neither can do vision over a chat-completions API. |
| mistral-evals | **Occasionally.** Its CoT "Final Answer:" prompts are a free prompt-sensitivity A/B on identical data; needs a two-line `/health` patch. |
| llama.cpp `tests.sh` verbatim | **No** — 30–50 GB of models we don't ship. Borrow the assertion, not the script. |

### Shared-GPU protocol (non-negotiable)

Layers 3 and 4 issue inference. Before starting: `ls /sys/class/kfd/kfd/proc/` and
`curl -s :11435/api/ps`; if a campaign is running, wait or use `SLEEP` to yield. Never touch
prod on `:11434`. Write outputs to durable disk, not `/tmp`.

---

## 5. The adapter: `extbench.py`

[vision-suite/extbench.py](vision-suite/extbench.py) runs slices of four external benchmarks
against our endpoints and scores them locally. It is stdlib-only — no `datasets`, no torch, no
HF token — because it pulls rows from the HF **datasets-server REST API**, which serves these
public datasets directly.

```bash
# 50-item OCRBench slice, think off, yielding the GPU between requests
LIMIT=50 THINK=false SLEEP=1 python3 docs/maxusai/vision-suite/extbench.py \
  http://127.0.0.1:11435 <tag> qwen3.6:35b-a3b-q4_k_m ocrbench
```

| benchmark | dataset | items | scoring |
|---|---|---|---|
| `ocrbench` | `echo840/OCRBench` test | 1,000 | contains-match, matching lmms-eval's semantics incl. the whitespace-stripped handwritten-math case |
| `countbenchqa` | `vikhyatk/CountBenchQA` test | 491 | integer match |
| `chartqa` | `lmms-lab-encoder/ChartQA` test | 2,500 | relaxed accuracy (±5 % numeric) |
| `refcoco` | `lmms-lab-encoder/RefCOCO` val | 17.6 k | **dialect-aware IoU** |

It follows the existing suite's conventions: `THINK=on|false`, `ENDPOINT=generate|chat`,
`NUM_PREDICT`, `NUM_CTX`, temperature 0, and a `scores` JSON written beside the script. `LIMIT`
and `OFFSET` slice the dataset; `SLEEP` yields the GPU on a shared host.

The `refcoco` mode is the reason this exists rather than just installing lmms-eval. It searches
{pixel, norm-1000, norm-0-1} × {xyxy, yxyx} per item and reports the **winning dialect and JSON
key alongside the IoU**, so a run doubles as a dialect probe — which is exactly what nemotron
needs, since it has no documented convention and switches space with think mode. Its prompt
deliberately says *"use whatever coordinate convention you were trained on"* instead of forcing
one, because [our own measurements](vision-suite/README.md) show models ignore coordinate
instructions anyway. Compare with the ceilings in [§3.1](#31-the-grounding-trap): this scorer is
a strict superset of both external ones.

**Verification status.** Dataset fetch, image caching, JPEG/PNG size parsing, and every scorer
are unit-tested offline — including round-trip checks that a perfect box expressed as pixel,
norm-1000 xyxy, and norm-1000 yxyx all score IoU ≈ 1.0 and report the correct dialect.
**No live inference run has been made yet**: at the time of writing two campaign runs held the
GPU (`vision_suite.py` against `:11435` and `:11441`, with `nemotron3:33b-q4_K_M` resident), and
per the shared-GPU protocol a benchmark run must not contend with them. The first live run
should be a 20-item OCRBench slice with `SLEEP=1` once the GPU is free.

### Suggested first campaign

| run | purpose |
|---|---|
| `ocrbench`, LIMIT=200, both think modes, nemotron3 | the only both-modes OCR anchor; checks the published 88.3-off / 86.6-on *direction* |
| `chartqa`, LIMIT=200, all three models | the one benchmark comparable to an official test-split number without substitution |
| `refcoco`, LIMIT=100, all three models | cross-validates our bbox scorer against external ground truth, and confirms the dialect map |

Record the dialect histogram from the `refcoco` runs. If nemotron's reported dialect shifts
between payloads, that is a preprocessing change — the canvas-offset behaviour documented in the
suite README is exactly this signal.
