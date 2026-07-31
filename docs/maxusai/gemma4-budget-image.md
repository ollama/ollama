# Building and deploying the gemma4-budget docker image

MaxusAI-fork runbook (this file and `Dockerfile.gemma4budget` are fork-only; they do
not exist upstream). Written 2026-07-17 after the first build + deployment on the
Blackwell workstation (10.8.0.6).

> 🛑 **AMD / gfx1151 is gated at 0.32.1 — do not bump the base for that host.**
> `0.32.5` produced degenerate vision output there and was rolled back on 2026-07-31.
> Blockers: [#17459](https://github.com/ollama/ollama/issues/17459),
> [#17475](https://github.com/ollama/ollama/issues/17475), and the unconditional
> `--direct-io` that upstream now applies to ROCm iGPUs. Read
> **[amd-upgrade-gate.md](amd-upgrade-gate.md)** before changing `OLLAMA_VERSION`, the
> `FROM` tag, or the proof tag for an AMD deployment. CUDA hosts are not covered by the
> dio clause, but #17475 was reported on CUDA and still applies.

## What the image is

An official `ollama/ollama` image with a single file replaced: `/usr/bin/ollama`,
rebuilt from this fork. The base tag tracks whatever upstream tag the fork is
synced to — **`0.32.5`** as of the 2026-07-29 sync.

### Image inventory (10.8.0.6, as of 2026-07-29)

| tag | fork commit | base | image id | status |
|---|---|---|---|---|
| `maxusai/ollama:85ebcb79-gemma4-budget` | `85ebcb79` | `0.32.1` | `6589c114137c` | **deployed** — also carries `:latest` |
| `maxusai/ollama:40f45419-gemma4-budget` | `40f45419` | `0.32.5` | `f83473a3874c` | built + verified, **not deployed** |

Both are local-only; no registry has either. `:latest` deliberately still points at the
deployed 0.32.1 image so it stays usable as a rollback target — the 0.32.5 build was
tagged by sha only. Do not move `:latest` onto an image you have not exercised.

The patch raises the gemma4 vision budget from llama.cpp's hardcoded
`set_limit_image_tokens(40, 280)` (upstream issue ollama/ollama#15626) to
**40..1120**, and exposes `image_min_tokens` / `image_max_tokens` as per-request
`api.Options` (changing them reloads the runner). For modelArch `gemma4`, ollama
passes `--image-min-tokens` / `--image-max-tokens` to llama-server (mtmd).

Patch surface is Go-only: `api/types.go`, `llm/llama_server.go` (+ tests).

> 🛑 **The budget flags are arch-gated.** `visionServerArgs()` switches on `modelArch`, so
> `image_min_tokens` / `image_max_tokens` do nothing for an arch that is not in that switch —
> and for some projectors (notably `nemotron_h_omni`) they are inert even if you add one.
> See [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) before assuming this
> knob applies to any model other than gemma4.

## Why an overlay instead of the repo Dockerfile

The v0.32 Dockerfile bases every stage on `rocm/dev-almalinux-8:7.2.1-complete`
(8.8 GB compressed) and builds CUDA 12.8 + CUDA 13 + Vulkan + MLX — ~45–70 GB of
docker scratch. The overlay needs ~7 GB. It is valid because the fork's C++/CUDA/MLX
payload is untouched — this diff is **empty**, where the tag matches the `FROM` in
`Dockerfile.gemma4budget`:

```bash
git diff HEAD v0.32.5 -- LLAMA_CPP_VERSION llama/ ml/ CMakeLists.txt \
    CMakePresets.json MLX_VERSION MLX_C_VERSION cmake/ x/imagegen/mlx
```

Run it before every rebuild. If it is non-empty the overlay is invalid for that base
and you must re-prove against a newer tag (or build the full Dockerfile).

### Keeping it valid when syncing upstream

**Sync by merging an upstream tag that has a published `ollama/ollama` image, then bump
the tag in the proof command and the `FROM` together.** Merging upstream `main` instead
leaves the tree ahead of every published tag — the payload proof goes non-empty against
`0.32.5` purely because upstream landed post-tag commits, with no fork change involved,
and there is no image to overlay onto. That failure mode is what makes the shortcut look
"dead" when it isn't: the fork's own payload delta has stayed **0 lines** throughout.

CUDA/Blackwell: the payload ships upstream's `cuda_v12` + `cuda_v13` backends.
Verified on the RTX PRO 6000 Blackwell (compute 12.0): selects `cuda_v13`, native
arch `1200` kernels, `BLACKWELL_NATIVE_FP4 = 1`.

## Build

From a clean checkout of the commit you want (tags embed the sha — keep them honest):

```bash
cd /opt/github/MaxusAI/ollama
SHA=$(git rev-parse --short=8 HEAD)
docker build -f Dockerfile.gemma4budget \
  --build-arg OLLAMA_VERSION=0.32.5-gemma4budget-$SHA \
  -t maxusai/ollama:$SHA-gemma4-budget .
```

**Tag by sha only.** The `-t maxusai/ollama:latest` this command used to carry moves
`:latest` onto the image you just built — which silently retires whatever known-good
image it pointed at. Promote `:latest` by hand once the build is exercised, if at all.

Measured: `0.32.1` base → 4.81 GB image / ~44 MB unique layer; `0.32.5` base → 4.8 GB
image / 38.8 MB unique layer, ~5 min wall (dominated by the 3.11 GB base pull and its
79 s extract, not the Go build).

## Verify

```bash
docker run --rm maxusai/ollama:$SHA-gemma4-budget --version
# client version is 0.32.5-gemma4budget-<sha>

# differential patch marker (a stock ollama/ollama image has ZERO occurrences):
docker run --rm --entrypoint sh maxusai/ollama:$SHA-gemma4-budget \
  -c "grep -c -- --image-max-tokens /usr/bin/ollama"
```

Note: rebuilds are **functionally equivalent but not bit-reproducible** — CGO_ENABLED=1
embeds build-id noise, so `sha256sum /usr/bin/ollama` differs across builds of the
same source (tested 2026-07-17: identical 38,570,136-byte size, identical version
string and patch marker, different hash). Verify by the checks above and the
end-to-end budget, never by binary hash.

End-to-end after deployment: the server log's llama-server launch line must contain
`--image-min-tokens 40 --image-max-tokens 1120`, and a gemma4:31b vision request at
the full budget shows `prompt_eval_count` ≈ text tokens + ~1,015 (measured 2,233 for
a ~1,218-token prompt — matching the reference server bit-for-bit).

## Deploy (SyncTechAU/data compose)

`docker/docker-compose.yaml` `ollama` service — pin the tag and never pull (the
image is local-only unless someone pushes it to a registry):

```yaml
image: maxusai/ollama:85ebcb79-gemma4-budget
pull_policy: never
```

then `docker compose up -d --no-deps ollama`. The `ollama_data` models volume
survives the version jump (verified 0.22.1 → 0.32.1, 50 models).

Rollback: `image: ollama/ollama:latest` + `pull_policy: always`, same command. To roll
back to the previous *patched* image instead, pin `85ebcb79-gemma4-budget` — it is kept
on the box for exactly this.

### Promoting the 0.32.5 build — not yet done

`40f45419-gemma4-budget` is built and verified but deliberately undeployed. Before
promoting it, re-measure rather than assuming the numbers below carry over: that build
moves llama.cpp from `b9888` to `b10091` (~200 upstream builds). Gemma's handling in
`tools/mtmd/clip.cpp` is **byte-identical** between those two tags, so the budget
behaviour is expected to hold — but that is read from source, not measured on the binary.

Cheapest check is a side-by-side on a spare port, leaving the live service alone:

```bash
docker run --rm --gpus all -v ollama_data:/root/.ollama -p 11435:11434 \
  maxusai/ollama:40f45419-gemma4-budget
```

Then confirm on `:11435` that the llama-server launch line still contains
`--image-min-tokens 40 --image-max-tokens 1120`, and that a gemma4:31b request at the
full budget still lands near `prompt_eval_count` 2,233 for the reference prompt.

## Measured (2026-07-17, gemma4:31b-it-q4_K_M, 1,218-token prompt + 1 image)

| | stock 0.22.1 local | patched 0.32.1 local | reference 10.8.0.3 |
|---|---|---|---|
| prompt_eval_count | 1,435 (~220 img tok) | **2,233 (~1,015 img tok)** | 2,233 |
| warm wall / call | 46.3 s (low budget!) | **22.4 s** | ~82 s ex-load |

First call after container start ≈ 323 s (139 s model load + llama-server/CUDA
warmup) — warm up before timing anything.

## The budget is a pixel ceiling, not a token count (2026-07-29)

`image_max_tokens` does **not** set a token count directly. The server converts it to
`image_max_pixels`, logged at model load:

> **Precision note (2026-07-30).** The heading is a useful mnemonic but slightly overstates
> the distinction: `image_max_pixels = max_tokens × S²` where `S = patch_size × n_merge` (48
> for gemma4), and token count is `area / S²` — so the pixel ceiling and the token ceiling are
> the *same* constraint in different units, not two different limits. 1120 × 48² = 2,580,480,
> exactly the logged value. The practical advice below is unaffected and correct: a wide image
> runs out of *pixels* long before it reaches the ceiling, so tune by total area. See
> [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md).

```
load_hparams: image_min_pixels:   92160 (custom value)
load_hparams: image_max_pixels: 2580480 (custom value)   # at budget 1120
```

An image is downscaled only if its **total pixel count** exceeds that ceiling. Aspect
ratio therefore decides how much of the budget you can actually spend. Measured against
the deployed image, warm, four generated colour-band images:

| image | pixels | @1120 | @280 |
|---|---|---|---|
| 4032x189 (21:1) | 762,048 | 393 prompt_eval (~334 img tok) | 288 (~229) |
| 8000x100 (80:1) | 800,000 | 391 (~361) | — |
| 189x4032 (tall) | 762,048 | 393 (~336) | — |
| 4000x4000 | 16,000,000 | **1146 (~1089)** | 313 (~256) |

Raising 280 → 1120 buys the 4000x4000 image **+325%** tokens but a panorama only **+46%**:
wide images run out of pixels before they run out of budget, so a high ceiling does almost
nothing for them. The 2,233 figure in the section above came from a normal-aspect image,
which is why it saw the full benefit. **Tune by total pixel count, not by how large one
dimension looks.**

### Vision correctness at 1120 — the open risk did not materialise

`docs/design/gemma4-vision-token-budgets-upstream-rebase.md` flagged a possible unclamped
vision position-embedding lookup at high budgets, and recommended a wide-image smoke test.
Run 2026-07-29: all four images above returned the correct four colours **and** the correct
orientation (including "top-to-bottom" for the tall one) at budget 1120. No crash, no
`GGML_ASSERT`, no garbling, up to 80:1. That risk was in `model/models/gemma4/model_vision.go`,
Go-runner code that no longer exists; nothing equivalent bites on the llama-server path.

## Gotchas

- `pull_policy: always` + a local-only tag = compose tries a registry pull and
  fails. Keep `never` (it also prevents silent image swaps).
- The version string is baked at build time via `--build-arg OLLAMA_VERSION=…`;
  without it you get the ARG default `0.32.5-gemma4budget` (no sha) — fine for
  throwaway builds, wrong for deployed ones.
- To use the image on another host without rebuilding: `docker push` to GHCR
  (`ghcr.io/maxusai/ollama`) — as of 2026-07-17 no registry has it.
- **A low `num_predict` looks exactly like a broken vision path.** gemma4 selects a
  thinking-capable parser (`renderer_parser="[completion vision tools thinking]"`). At
  `num_predict=120` three of the four test images returned `"response": ""` with
  `eval_count` of exactly 120 — the whole allowance went into an unclosed thinking block.
  At 600 they all answered correctly. Check `thinking` as well as `response`, and don't
  diagnose an empty reply as an image failure until `eval_count < num_predict`.
- **Changing the budget forces a runner reload.** That is by design (`image_min_tokens` /
  `image_max_tokens` participate in the scheduler's reload comparison), but it means a
  budget change costs a full model reload. On 10.8.0.6 with concurrent open-webui traffic
  one such request exceeded a 600 s client timeout. Keep the budget fixed per server;
  don't vary it per request on a loaded host.
