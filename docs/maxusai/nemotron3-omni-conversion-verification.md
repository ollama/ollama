# Nemotron 3 Omni: local conversion verified against the published checkpoint

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-08-01 from a
live conversion on the gfx1151 host, verifying fork PR
[#15](https://github.com/MaxusAI/ollama/pull/15) (`147930aa`, "accept c-radio_v4-h RADIO
version for Nemotron 3 Omni").

> **The one thing to take away:** the fork's converter now converts the real
> `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` checkpoint end-to-end, and its
> output is **tensor-for-tensor identical** to the official `nemotron3:33b` registry
> GGUF. Local `ollama create` from the HF checkpoint (custom quants, fine-tune
> conversions) is a verified-working path — on the fork only. Upstream `ollama/ollama`
> main (checked at `8d8c701d`, 2026-07-31) still rejects the checkpoint outright.

## What was broken

`convert/convert_nemotron_h.go` allowlisted only `""` and `"radio_v2.5-h"` as
`vision_config.version`. The published checkpoint ships `"c-radio_v4-h"`, so metadata
parsing failed before a single tensor was read:

```
nemotron_h_omni: unsupported RADIO version "c-radio_v4-h"
```

Reproduced live against the real `config.json` with the pre-fix fork (`3f6ea735`); the
fixed fork parses the same config and selects the `nemotronHNanoVLModel` converter. The
gap was latent, never a lived failure: every `nemotron3:33b-*` tag on this host was
pulled pre-converted from the registry, so the converter path had never met the real
checkpoint. The gate came in via upstream `87288ced` ("New models (#15861)"), whose test
fixture says `radio_v2.5-h` — evidently written against a pre-release checkpoint.

The two versions are interchangeable for conversion purposes: c-radio_v4-h is the same
ViT-H/16 (32 blocks, hidden 1280, fused qkv, FFN 5120) with the same tensor names; the
version string reflects the C-RADIO distillation recipe, not the layout.

## Method

1. **Checkpoint integrity.** All 17 safetensors shards (66.0 GB) SHA-256-verified
   against the HF LFS manifest; all 26 auxiliary files git-blob-SHA-1-verified against
   the HF tree listing. Everything matched (one file, `video_processing.py`, was missing
   from the local copy and re-fetched; it is not read by the converter).
2. **Before/after metadata parse.** A small Go driver calling
   `convert.LoadModelMetadata` was built twice via `replace` directives: once against
   pre-fix `3f6ea735`, once against fixed main.
3. **Full conversion.** The same driver calling `convert.ConvertModel` over the
   checkpoint, writing an F16 GGUF.
4. **Equivalence check.** `fs/ggml.Decode` over both our GGUF and the registry
   `nemotron3:33b-q4_K_M` model blob
   (`sha256-02025860…`), diffing tensor name sets, shapes, and KV metadata.

## Results

| check | result |
|---|---|
| conversion (`ConvertModel`) | OK, 1m53s, 66.06 GB F16 GGUF |
| architecture | `nemotron_h_omni` |
| tensors | 1608 total; vision 515, audio 686, text-blk 398 |
| text KVs | block_count 52, embedding 2688, experts 128/6 used |
| vision KVs | 32 blocks × 1280, patch 16, image 512, min/max patches 1024/13312 |
| audio KVs | 24 blocks, 128 mel bins |
| vs registry blob: tensor sets | identical, 1608 = 1608, none missing either way |
| vs registry blob: shapes | zero mismatches |
| vs registry blob: KVs | two cosmetic deltas (below) |

The vision count is exactly the predicted arithmetic: 515 = 32 blocks × 16 tensors
(after the fused qkv splits into q/k/v weight+bias) + patch, position, and cls
embeddings — the c-radio_v4-h tower maps 1:1 through the radio_v2.5-h code path.

KV deltas vs the registry blob, both harmless: the registry has
`tokenizer.ggml.add_unknown_token=false` (we omit it), we have
`tokenizer.ggml.padding_token_id=11` (the registry omits it).

**Benign warning:** conversion logs `unknown pretokenizer, using default`
(digest `1d64a9a8…`). The registry blob carries identical tokenizer settings, so the
default fallback reproduces exactly what upstream shipped for this model.

## Where things live

- Verified checkpoint: `/opt/github/MaxusAI/nemotron3-omni-bf16` on the gfx1151 host
  (66 GB, shards checksum-verified). The F16 GGUF was deleted after verification to
  reclaim disk; re-converting takes ~2 minutes from this checkpoint.
- Evidence trail: [PR #15](https://github.com/MaxusAI/ollama/pull/15) and its
  [post-merge verification comment](https://github.com/MaxusAI/ollama/pull/15#issuecomment-5150545805).
- Upstream status: `ollama/ollama` main still carries the old gate; an upstreaming PR
  of the fork fix is in progress.

Related: [vision-token-budget-measurements.md](vision-token-budget-measurements.md)
measured `nemotron3:33b` registry tags on this host — those numbers apply unchanged to
locally converted models, since the artifacts are structurally identical.
