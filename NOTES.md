# Issue #17113 — glm-ocr repeated results

## 1. Root cause

`glm-ocr` has **two** end-of-generation tokens. From the model's
`generation_config.json` (`zai-org/GLM-OCR`):

```json
"eos_token_id": [59246, 59253]
```

which are `<|endoftext|>` (59246) and `<|user|>` (59253). Like the rest of the
GLM-4 family, the model normally ends its answer by emitting `<|user|>`, i.e. by
starting the next user turn.

The published Ollama blob records both ids, but only in Ollama's own
`tokenizer.ggml.eos_token_ids` **array**. Dumping the KV metadata of the actual
registry blob (`library/glm-ocr:latest`, arch `glmocr`) shows:

```
tokenizer.ggml.eos_token_id  = 59246
tokenizer.ggml.eos_token_ids = [59246, 59253]     <- Ollama-only key
tokenizer.ggml.eot_token_id  = <absent>
tokenizer.ggml.pre           = llama-bpe
```

Since #16031 (CGO runners removed, llama-server used for all GGML models), that
file is loaded by llama.cpp through `llama/compat`. llama.cpp builds its EOG set
from the singular `eos_token_id` / `eot_token_id` / `eom_token_id` keys plus a
hard-coded list of token *names* (`llama-vocab.cpp`, b9888). It never reads
`tokenizer.ggml.eos_token_ids`, and `<|user|>` is not in the name list — it is
only flagged as a control token that "is not marked as EOG".

So under llama-server the model's EOG set is `{<|endoftext|>}` and `<|user|>` is
just another token. When glm-ocr finishes an answer with `<|user|>`, generation
does not stop: the context now looks like the start of a fresh user turn with the
same image still attached, so the model transcribes the page again — which is
exactly what the issue shows (the table repeated inside a ```` ```table ```` fence,
then a run of stray ```` ``` ```` until the token cap, and a much longer runtime).
It is image-dependent because it only bites when the model picks `<|user|>` over
`<|endoftext|>` to close the turn.

Same root cause as #16892 ("glm-ocr infinite loop", where even `hi` loops), and it
matches the report there that v0.23.4 (old Go engine, which honoured
`eos_token_ids`) stopped after 7 tokens while v0.30.10 (llama-server) generated
3975.

## 2. The fix

`llama/compat/llama-ollama-compat.cpp`, `handle_glmocr()`: when a legacy `glmocr`
GGUF has no `tokenizer.ggml.eot_token_id`, look `<|user|>` up in the vocab and
publish it as the EOT token. llama.cpp does add `eot_token_id` to its EOG set, so
generation stops at the end of the turn again.

Why here and why this shape:

* `llama/compat` is the layer whose stated job is normalising already-published
  Ollama blobs whose metadata does not match llama.cpp's loaders. There is direct
  precedent: `handle_llama3()` fixes `eos`/`eot`, `handle_glm4moelite()` calls
  `fix_glm4moelite_eog_token_ids()` for the same Ollama-vs-llama.cpp EOG-key
  mismatch.
* `convert/convert_glmocr.go` already writes `eot_token_id = <|user|>` for *new*
  conversions (`applyGlmOcrTokenizerKV`). The compat change mirrors it, so both
  paths agree; the published blob predates that converter change.
* Guarded on the key being absent, so newly converted files (which also carry
  `general.architecture=glmocr`) are left untouched.
* Scoped to EOG only. `bos`/`unknown` are also absent from the blob but are not
  part of this bug — `add_bos_token` is false and the renderer emits `[gMASK]<sop>`
  itself.

A small `find_token_id()` helper was added next to the existing `token_at_equals()`
util. Looking the token up by name (rather than taking `eos_token_ids[1]`
positionally, as the glm4moelite helper does) keeps the compat layer's definition
of GLM-OCR's turn-ending token identical to the converter's.

## 3. Files changed

* `llama/compat/llama-ollama-compat.cpp` — `find_token_id()` helper; EOT
  normalisation in `handle_glmocr()`.
* `llama/compat/README.md` — the `glmocr` row of the transformations table now
  mentions tokenizer/EOG metadata.
* `convert/convert_glmocr_test.go` — new focused test.

## 4. Risk / uncertainty

* Low blast radius: the new code only runs for `general.architecture=glmocr`
  files, only when `eot_token_id` is missing, and only sets that one key.
* If a future glmocr blob legitimately wanted `<|user|>` *not* to be EOG, this
  would be wrong — but the model's own `generation_config.json` and the current
  converter both say it is EOG, so that is not a real case.
* The uncertainty that remains is empirical, not structural: I could not run
  glm-ocr on a GPU here, so I have not watched the reported image produce a
  single, non-repeated table. The metadata analysis below shows the model can now
  stop where it previously could not; it does not by itself prove every reported
  repetition disappears (a sampler-level repetition loop inside a single turn
  would be a separate problem). The fact that #16892 sees the same runaway on a
  plain `hi` prompt is what makes me confident this is the mechanism, not a
  contributing factor.
* `llama/compat` has no test harness in-tree (`LLAMA_BUILD_TESTS=OFF`, no
  `add_test`), so the C++ change is not covered by a unit test; adding a ctest
  target would have meant new build/CI wiring well beyond this fix.

## 5. How I verified it

* **Ground truth on the real blob.** Range-fetched the GGUF header of
  `registry.ollama.ai/v2/library/glm-ocr` and parsed its KVs: confirmed
  `eos_token_ids = [59246, 59253]`, no `eot_token_id`, and that token 59253 is
  `<|user|>` with token_type 3 (CONTROL).
* **Reproduced llama.cpp's EOG rules** (from `llama-vocab.cpp` at the pinned
  `LLAMA_CPP_VERSION` = b9888) against those exact KVs:

  ```
  BEFORE fix -> EOG: [(59246, '<|endoftext|>')]
  AFTER  fix -> EOG: [(59246, '<|endoftext|>'), (59253, '<|user|>')]
  ```

  i.e. before the change the token the model uses to end its turn is not an EOG
  token; after it, it is.
* **Compiled the compat translation unit** against llama.cpp b9888:
  `g++ -fsyntax-only -std=c++17 -I llama/compat -I <llama.cpp>/src -I <llama.cpp>/include
  -I <llama.cpp>/ggml/include -I <llama.cpp>/ggml/src llama/compat/llama-ollama-compat.cpp`
  → clean.
* **Go tests**: `go test ./convert/ ./model/renderers/ ./model/parsers/` → all pass,
  including the new `TestGlmOcrTokenizerKVMarksUserTokenAsEOT`, which pins the same
  invariant on the conversion path (`<|user|>` is written as EOT).
