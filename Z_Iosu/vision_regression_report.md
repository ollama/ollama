# Vision Regression Technical Report

Date: 2025-09-01  
Author: Investigation session transcript distilled from troubleshooting steps

## 1. Executive Summary
Recent builds of the Ollama container ("non-working" variant) fail to process image inputs for a vision-capable model that works in an older snapshot container. The user-facing message observed was paraphrased as "unable to view or interpret images". This exact literal does not exist in the repository; instead the internal capability check likely returns an error equivalent to "does not support vision" which an external caller or wrapper rephrased.

Root cause is not yet conclusively pinned down, but evidence points toward a loss (or non-detection) of the `vision` capability arising from missing / altered GGUF metadata keys (e.g. `<arch>.vision.block_count`) or absent projector components (`ProjectorPaths`) in the newer environment. A secondary possibility is that an obsolete model identifier is being used (see `server/routes.go`) forcing a compatibility error disguised as lack of vision support.

## 2. Observed Symptoms
| Aspect | Working Snapshot | Non-Working Build |
| ------ | ---------------- | ----------------- |
| Image completion request | Succeeds (vision tokens generated) | Returns capability style error (translated) |
| Capabilities (expected) | Includes `vision` | Missing `vision` (inferred) |
| Logs (vision encoder) | Shows vision encoder / projector load lines | (Not yet captured) |
| Model metadata | Contains `*.vision.block_count` | Suspect missing / zero (TBD) |

## 3. Capability Detection Mechanism
Implemented in `server/images.go` (`(*Model).Capabilities()`):
1. Opens the primary model GGUF file via `gguf.Open(m.ModelPath)`.
2. Adds `embedding` capability if `pooling_type` key present; otherwise assumes `completion`.
3. Adds `vision` capability if key `vision.block_count` (actually `<arch>.vision.block_count`) exists **OR** if `ProjectorPaths` is non-empty (projector-based multimodal models).
4. Adds `tools`, `insert`, `thinking` based on template variables and thinking tag inference.

If a request needs vision (image input) and `vision` is absent, `CheckCapabilities()` composes an error with base `errCapabilities` + `errCapabilityVision` leading to a higher layer user-friendly phrase.

## 4. Relevant Source Hotspots
| File | Relevance |
| ---- | --------- |
| `server/images.go` | Capability inference and string assembly |
| `server/create.go` (lines ~500+) | Distinguishes standalone vision vs text+vision fusion (standalone if `vision.block_count` present but `block_count` absent) |
| `server/routes.go` | Special-case rejection for obsolete `llama3.2-vision` model variant |
| `fs/gguf/gguf.go` | Key lookup logic auto-prefixes model architecture when key not `general.*` / `tokenizer.*` |
| `fs/ggml/ggml.go` | Vision graph sizing (`VisionGraphSize()`) depends on `vision.block_count` and related keys |
| `convert/convert_*.go` | Writers of `*.vision.block_count` and other metadata during conversion |
| `model/models/*/model_vision.go` | Construction of vision layer structures using `vision.block_count` |

## 5. Metadata Keys of Interest
Primary gating key: `<arch>.vision.block_count` (examples: `llama4.vision.block_count`, `gemma3.vision.block_count`, `mllama.vision.block_count`, `qwen25vl.vision.block_count`). Additional supporting keys: `vision.image_size`, `vision.patch_size`, `vision.num_channels`, `vision.attention.head_count`, `vision.embedding_length`, `vision.max_num_tiles` (architecture-specific), etc.

Loss or absence of `<arch>.vision.block_count` prevents `vision` capability assignment unless a projector chain is detected.

## 6. Historical Change Note (Struct Tag)
Documentation and prior diffs mention a change from struct tags `gguf:"v,vision"` to `gguf:"v"`. While current code uses key prefixing to retrieve values, older conversion logic or downstream tools might have depended on the `vision` alias to name or copy certain tensors or metadata. A model converted with the updated tooling but consumed by mismatched runtime versions (or vice versa) may thus miss required keys.

## 7. Hypotheses (Ranked)
1. **Missing `vision.block_count` key** in problematic model (metadata regression during conversion or pull) → capability not granted.
2. **Obsolete model name** triggers hard-coded incompatibility path (`routes.go`), surfaced to user as inability to process images.
3. **Absent projector files** (`ProjectorPaths` empty) for a model relying on separated projector components (e.g. mllama, qwen25vl) → no vision capability.
4. **Conversion mismatch / tooling version drift**: tag change or quantization step dropping vision-specific metadata.
5. **Malformed client request** (improper multimodal message structure) producing a fallback error interpreted as missing vision (less likely given differential container behavior).

## 8. Evidence Collected So Far
- Grep search: no literal "unable to view or interpret images"; confirms message originates outside repository code or is paraphrased.
- Located capability gating logic and necessary metadata keys.
- Identified special-case obsolescence for `llama3.2-vision` in `routes.go`.
- Confirmed converters write `<arch>.vision.block_count` (see lines in `convert_mistral.go`, `convert_mllama.go`, `convert_qwen25vl.go`, etc.).

## 9. Immediate Verification Steps
Run (PowerShell examples):
```powershell
ollama show <working_model> --json > working.json
ollama show <failing_model> --json > failing.json
Compare-Object (Get-Content working.json) (Get-Content failing.json) | Select-Object -First 40

# Inspect GGUF keys for failing model (adjust path):
strings C:\path\to\failing_model.gguf | findstr /I "vision.block_count"
```
If the above produces no match, we have a confirmed metadata absence.

## 10. Additional Logging (Optional Patch Suggestion)
Add temporary debug in `Capabilities()` after opening the GGUF file:
```go
slog.Debug("vision key check", "model", m.Name, "vision.block_count.valid", f.KeyValue("vision.block_count").Valid())
```
Remove after diagnosis.

## 11. Bisect / Narrowing Strategy
1. List local images chronologically; map digests to internal `ollama version` output.
2. Construct pass/fail table with: Digest | Created | Commit (if label) | Vision OK? | Notes.
3. Identify first failing digest → gather commit range.
4. `git diff <good_commit>..<bad_commit>` focusing on: `server/images.go`, `server/create.go`, `fs/gguf/`, `convert/`, `model/models/*vision*`.

## 12. Mitigations / Workarounds
- Re-pull the affected model: `ollama rm <model>; ollama pull <model>` (forces metadata refresh).
- If projector missing, explicitly include projector GGUF in a Modelfile wrapper.
- Reconvert source weights with current converter ensuring vision metadata is emitted.
- Patch runtime (temporary) to treat alternative markers (e.g. presence of any `vision.*` key set) as enabling vision capability.

## 13. Sample Vision-Capable Modelfile Template
```modelfile
FROM hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL
PARAMETER num_ctx 8192
PARAMETER num_gpu 1
PARAMETER temperature 0.7
SYSTEM You are a multimodal assistant that reasons over images and text.
# If a separate projector exists:
# MODEL projector.gguf
```
Ensure the template (if custom) surfaces images (e.g. `{{ .Images }}`) if required by your model's prompt logic.

## 14. Success Criteria
- `ollama show <model> --json` lists `"vision"` in `capabilities`.
- GGUF contains `<arch>.vision.block_count` and related vision keys.
- Image prompt returns coherent description output (no capability error).

## 15. Open Actions (from Project TODO)
| Task | Status |
| ---- | ------ |
| Compare with official vision model Modelfile | Pending |
| Capture debug logs with image request | Pending |
| Confirm template contains image handling | Pending |
| Propose adjusted Modelfile (optimized params) | Pending |
| Verify GGUF metadata / incompatibility | Pending |
| Diff legacy tag vs HEAD (vision files) | Pending |
| Enumerate commits touching vision path | Pending |
| List images & map digest→commit | Pending |
| Bisect to first failing image | Pending |
| Extract regression diff | Pending |

## 16. Next Recommended Step
Start by confirming metadata: run capability JSON diff and search for `vision.block_count` in failing model. This will immediately confirm or eliminate the leading hypothesis and guide whether to focus on conversion tooling or request formatting.

## 17. Appendix: Key Grep Queries Used
```bash
grep -R "vision.block_count" -n .
grep -R "ProjectorPaths" -n server/
grep -R "llama3.2-vision" -n server/routes.go
```

---
End of report. Update this document once the first failing digest and its commit are identified.
