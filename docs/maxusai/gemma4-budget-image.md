# Building and deploying the gemma4-budget docker image

MaxusAI-fork runbook (this file and `Dockerfile.gemma4budget` are fork-only; they do
not exist upstream). Written 2026-07-17 after the first build + deployment on the
Blackwell workstation (10.8.0.6).

## What the image is

Official `ollama/ollama:0.32.1` with a single file replaced: `/usr/bin/ollama`,
rebuilt from this fork at `85ebcb79` (merge of PR #2, `feat/gemma4-image-max-tokens`).
The patch raises the gemma4 vision budget from llama.cpp's hardcoded
`set_limit_image_tokens(40, 280)` (upstream issue ollama/ollama#15626) to
**40..1120**, and exposes `image_min_tokens` / `image_max_tokens` as per-request
`api.Options` (changing them reloads the runner). For modelArch `gemma4`, ollama
passes `--image-min-tokens` / `--image-max-tokens` to llama-server (mtmd).

Patch surface is Go-only: `api/types.go`, `llm/llama_server.go` (+ tests).

## Why an overlay instead of the repo Dockerfile

The v0.32 Dockerfile bases every stage on `rocm/dev-almalinux-8:7.2.1-complete`
(8.8 GB compressed) and builds CUDA 12.8 + CUDA 13 + Vulkan + MLX — ~45–70 GB of
docker scratch. The overlay needs ~7 GB. It is valid because the fork's C++/CUDA/MLX
payload is untouched — this diff is **empty** (`4f7786d0` = the fork's merge base,
contained in v0.32.1):

```bash
git diff 4f7786d0 v0.32.1 -- LLAMA_CPP_VERSION llama/ ml/ CMakeLists.txt \
    CMakePresets.json MLX_VERSION MLX_C_VERSION cmake/ x/imagegen/mlx
```

If a future rebase makes that diff non-empty, the overlay shortcut is dead — build
the full Dockerfile (or re-prove equivalence against the new upstream tag).

CUDA/Blackwell: the payload ships upstream's `cuda_v12` + `cuda_v13` backends.
Verified on the RTX PRO 6000 Blackwell (compute 12.0): selects `cuda_v13`, native
arch `1200` kernels, `BLACKWELL_NATIVE_FP4 = 1`.

## Build

From a clean checkout of the commit you want (tags embed the sha — keep them honest):

```bash
cd /opt/github/MaxusAI/ollama
SHA=$(git rev-parse --short=8 HEAD)
docker build -f Dockerfile.gemma4budget \
  --build-arg OLLAMA_VERSION=0.32.1-gemma4budget-$SHA \
  -t maxusai/ollama:$SHA-gemma4-budget -t maxusai/ollama:latest .
```

~4.81 GB image, ~44 MB unique layer over `ollama/ollama:0.32.1`.

## Verify

```bash
docker run --rm maxusai/ollama:$SHA-gemma4-budget --version
# client version is 0.32.1-gemma4budget-<sha>

# differential patch marker (stock 0.32.1 has ZERO occurrences):
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

Rollback: `image: ollama/ollama:latest` + `pull_policy: always`, same command.

## Measured (2026-07-17, gemma4:31b-it-q4_K_M, 1,218-token prompt + 1 image)

| | stock 0.22.1 local | patched 0.32.1 local | reference 10.8.0.3 |
|---|---|---|---|
| prompt_eval_count | 1,435 (~220 img tok) | **2,233 (~1,015 img tok)** | 2,233 |
| warm wall / call | 46.3 s (low budget!) | **22.4 s** | ~82 s ex-load |

First call after container start ≈ 323 s (139 s model load + llama-server/CUDA
warmup) — warm up before timing anything.

## Gotchas

- `pull_policy: always` + a local-only tag = compose tries a registry pull and
  fails. Keep `never` (it also prevents silent image swaps).
- The version string is baked at build time via `--build-arg OLLAMA_VERSION=…`;
  without it you get the ARG default `0.32.1-gemma4budget` (no sha) — fine for
  throwaway builds, wrong for deployed ones.
- To use the image on another host without rebuilding: `docker push` to GHCR
  (`ghcr.io/maxusai/ollama`) — as of 2026-07-17 no registry has it.
