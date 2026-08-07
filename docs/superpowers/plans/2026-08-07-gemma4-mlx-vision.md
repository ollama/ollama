# Gemma 4 MLX Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement image input for the gemma4 nvfp4 MLX models (12b unified embedder; shared 27-layer tower for 26b/31b), end to end: wire protocol → budget-fill preprocessor → vision forward passes → embedding injection with bidirectional masks → capability re-allow.

**Architecture:** Media rides the existing `llm.CompletionRequest.Media` into a new field on the MLX wire struct. `Runner.Prepare` (HTTP goroutine, pure Go) splits `[img-N]` markers, preprocesses each image per ADR 0008 budget-fill (ladder snap → sqrt fill → 48px floor → shave), patchifies, and splices `boi + n×image_token + eoi` ids with recorded spans + prefix-cache salts. The pipeline (MLX thread) encodes each image through the model's vision path, merges features into the token embeddings, and prefills in a single chunk with per-layer-type array masks that make image blocks bidirectional (overriding the sliding window, matching the reference). The gemma4 model implements a new optional `base.VisionModel` interface; `batch.Batch` gains `InputsEmbeds` + `BidiSpans`.

**Tech Stack:** Go; `x/mlxrunner/mlx` cgo MLX bindings; `golang.org/x/image/draw` (CatmullRom ≈ PIL bicubic); quantized weights load transparently via `model.LinearFactory` (nvfp4 single/double-scale confirmed in blobs).

**Verified model facts (from config/processor blobs and safetensors headers, 2026-08-07):**
- Tokens (all three): boi=255999, image=258880, eoi=258882. All three text configs: `use_bidirectional_attention: "vision"`, `hidden_size_per_layer_input: 0` (no PLE).
- Preprocessing (both families): convert RGB, bicubic resize (resample 3), rescale 1/255, **no mean/std normalize**. Align = 48 px/soft-token (patch 16 × pool 3).
- 12b unified (`gemma4_unified_vision`): 48px patches → `[L, 6912]`, LayerNorm(6912) → Linear(6912→3840, bias, nvfp4 double-scale) → LayerNorm → +pos (table `[1120, 2, 3840]` bf16; `[x,0]` + `[y,1]`) → LayerNorm. LayerNorm eps 1e-5 (torch default). Projection: weightless RMSNorm(eps 1e-6) → Linear(3840→3840, no bias, nvfp4 double-scale). NO `2x−1` scaling.
- 26b/31b tower (`gemma4_vision`): 16px patches → `[L, 768]`, values `2x−1`; input_proj 768→1152 (26b bf16, 31b nvfp4+global); +pos tables `[2, 10240, 1152]` (`[0,x]` + `[1,y]`); 27 blocks: sandwich RMSNorm (eps 1e-6, weight applied directly, no +1), attn 16 heads × 72 dim with per-head q/k RMSNorm (weighted) and weightless v RMSNorm, 2D RoPE θ=100 (18 freqs per axis, rotate_half within each 36-chan half), SDPA **scale=1.0**, MLP gelu_tanh(gate)·up→down (4304); linear paths carry `.linear.` infix (`self_attn.q_proj.linear`), nvfp4 group 16, no biases; then 3×3 mean-pool (f32) × √1152; `(h − std_bias) · std_scale`; projection: weightless RMSNorm(1e-6) → Linear(1152→2816/5376).
- Patch layout both families: `[C,H,W] → [pH, pW, p, p, C]` flatten, channel fastest. Positions `(x=col, y=row)`, row-major. Our grids are exact 48-multiples ⇒ never any padding, never a pooling mask.
- Bidirectional overlay (reference `language.py`): contiguous image-token runs attend bidirectionally on **both** full and sliding layers, overriding the window. Chunked prefill disabled when images present.

---

### Task 1: Export budget-fill sizing from `llm`

**Files:**
- Modify: `llm/llama_server.go` (~line 1440-1475)
- Test: `llm/llama_server_test.go`

- [ ] **Step 1.1: Write the failing test** — append to `llm/llama_server_test.go`:

```go
func TestBudgetFillSize(t *testing.T) {
	cases := []struct {
		name          string
		w, h, maxTok  int
		wantW, wantH  int
	}{
		// findings §9: 2160×1152 is already the budget-1120 target for its aspect.
		{"native ladder grid", 2160, 1152, 1120, 2160, 1152},
		// findings §9: 1920×1080 fills to 44×25 = 2112×1200 at 1120.
		{"1080p fills to 44x25", 1920, 1080, 1120, 2112, 1200},
		// ADR 0008: budget-fill upscales small images.
		{"640x480 upscales", 640, 480, 1120, 1968, 1488},
		// Off-ladder request snaps down (600 → 560).
		{"snap down", 1920, 1080, 600, 1488, 816},
		// Below the lowest rung clamps up to 70.
		{"clamp up", 1920, 1080, 10, 528, 288},
		// Extreme aspect: short axis floors at 48, long axis shaved to fit.
		{"extreme aspect", 4800, 10, 70, 3360, 48},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, h := BudgetFillSize(tc.w, tc.h, 48, tc.maxTok)
			if w != tc.wantW || h != tc.wantH {
				t.Fatalf("BudgetFillSize(%d,%d,48,%d) = %dx%d, want %dx%d",
					tc.w, tc.h, tc.maxTok, w, h, tc.wantW, tc.wantH)
			}
			if got := budgetFillTokens(tc.w, tc.h, 48, tc.maxTok); got != (w/48)*(h/48) {
				t.Fatalf("budgetFillTokens = %d, want %d", got, (w/48)*(h/48))
			}
		})
	}
}
```

(Expected values must be re-derived by hand from the algorithm before committing; fix the table if arithmetic disagrees — the algorithm is authoritative, ADR grids are spot checks.)

- [ ] **Step 1.2:** `go test ./llm/ -run TestBudgetFillSize` → FAIL (undefined `BudgetFillSize`).
- [ ] **Step 1.3:** Refactor in `llm/llama_server.go`:

```go
// BudgetFillSize returns the Gemma 4 budget-fill target size (ADR 0008; the
// 004 compat patch's calc_size_budget_fill): the requested ceiling snaps down
// to the supported ladder, the image scales (up or down) to fill it, each
// axis floors to align, and extreme aspect ratios shave the long axis back
// under budget. Exported for the MLX runner's preprocessor.
func BudgetFillSize(width, height, align, maxTokens int) (int, int) {
	budget := gemma4SnapBudget(maxTokens)
	pxPerToken := align * align
	factor := math.Sqrt(float64(budget*pxPerToken) / (float64(width) * float64(height)))
	wBar := max(align, int(math.Floor(float64(width)*factor/float64(align)))*align)
	hBar := max(align, int(math.Floor(float64(height)*factor/float64(align)))*align)
	for (wBar/align)*(hBar/align) > budget {
		if wBar >= hBar {
			wBar -= align
		} else {
			hBar -= align
		}
	}
	return wBar, hBar
}

func budgetFillTokens(width, height, align, maxTokens int) int {
	wBar, hBar := BudgetFillSize(width, height, align, maxTokens)
	return (wBar / align) * (hBar / align)
}

// Gemma4ImageBudget exposes the gemma4 min/max token budget resolution for
// the MLX runner. See gemma4ImageTokenBudget.
func Gemma4ImageBudget(opts api.Options) (minTok, maxTok int) {
	return gemma4ImageTokenBudget(opts)
}

// Gemma4ImageAlign is the pixel edge of one gemma4 soft token (patch 16 × pool 3).
const Gemma4ImageAlign = gemma4ImageAlign
```

- [ ] **Step 1.4:** `go test ./llm/ -run 'TestBudgetFillSize|TestImageTokensForSize'` → PASS (existing sizing tests must stay green).
- [ ] **Step 1.5:** Commit: `feat(llm): export gemma4 budget-fill sizing for the MLX runner`

---

### Task 2: Media on the MLX wire; runner-side rejection replaces the client guard

**Files:**
- Modify: `x/mlxrunner/client.go` (struct ~line 100, guard ~line 143)
- Modify: `x/mlxrunner/pipeline.go` (`Prepare`)
- Rewrite: `x/mlxrunner/client_test.go`

- [ ] **Step 2.1: Client test first** — replace `TestCompletionRejectsMedia` with a forwarding test (fake `http.RoundTripper`, pattern from `x/imagegen/server_test.go`):

```go
package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return fn(req) }

func newCompletionTestClient(t *testing.T, handler func(*http.Request) string) *Client {
	t.Helper()
	return &Client{
		port:   11434,
		status: llm.NewStatusWriter(io.Discard),
		client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(handler(req))),
				Request:    req,
			}, nil
		})},
	}
}

// Media must ride the wire to the runner subprocess: the runner owns
// model-specific preprocessing and the does-this-model-support-images check.
func TestCompletionForwardsMedia(t *testing.T) {
	img := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	var got CompletionRequest
	c := newCompletionTestClient(t, func(r *http.Request) string {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		return `{"Content":"ok","Done":true}` + "\n"
	})

	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt: "describe [img-0]",
		Media:  []llm.MediaData{{Data: img, ID: 0, Kind: llm.MediaKindImage}},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Media) != 1 || !bytes.Equal(got.Media[0].Data, img) || got.Media[0].ID != 0 {
		t.Fatalf("media not forwarded: %+v", got.Media)
	}
}
```

Check `llm.NewStatusWriter`'s actual signature/name before using (the imagegen test constructs `status` differently); adapt if needed.

- [ ] **Step 2.2:** `go test ./x/mlxrunner/ -run TestCompletionForwardsMedia` → FAIL (no Media field).
- [ ] **Step 2.3:** In `x/mlxrunner/client.go`: add `Media []llm.MediaData` to `CompletionRequest` (after `Options`); delete the rejection guard block in `Completion`; set `creq.Media = req.Media`. Remove now-unused imports if any (`api`/`http` stay — used elsewhere).
- [ ] **Step 2.4: Runner-side rejection** in `x/mlxrunner/pipeline.go` `Prepare` — the explicit-error property must hold at this commit (no model implements vision yet):

```go
	if len(request.Media) > 0 {
		if _, ok := r.Model.(base.VisionModel); !ok {
			return errors.New("this model does not support image input on the MLX runner")
		}
	}
```

and in `x/mlxrunner/model/base/base.go` define the interface (full form used by later tasks):

```go
// VisionInput is one preprocessed image ready for encoding.
type VisionInput interface {
	// SoftTokens is the number of image placeholder tokens this input expands to.
	SoftTokens() int
}

// VisionModel is implemented by models that accept image input.
type VisionModel interface {
	// VisionTokens returns the ids bracketing one image's soft tokens.
	VisionTokens() (boi, image, eoi int32)
	// NewVisionInput decodes and preprocesses one image. Pure Go — callable
	// off the MLX thread.
	NewVisionInput(data []byte, opts api.Options) (VisionInput, error)
	// EncodeVision embeds a preprocessed image; returns [1, SoftTokens, hidden].
	// MLX thread only.
	EncodeVision(in VisionInput) *mlx.Array
	// MergedEmbeddings returns scaled token embeddings with features spliced
	// over spans (half-open [start,end) positions). MLX thread only.
	MergedEmbeddings(inputIDs *mlx.Array, features []*mlx.Array, spans [][2]int32) *mlx.Array
}
```

(`base` gains imports `github.com/ollama/ollama/api` and the mlx package.)

- [ ] **Step 2.5:** `go test ./x/mlxrunner/... && go build ./...` → PASS. Also run `go test ./server/ -run TestGenerateWithImages` — the groundwork's capability-rejection subtests must stay green (they gate at the routes layer, unaffected).
- [ ] **Step 2.6:** Commit: `feat(mlxrunner): carry media on the wire; reject it in the runner for non-vision models`

---

### Task 3: `batch.Batch` carries embeddings + bidirectional spans; gemma4 consumes them

**Files:**
- Modify: `x/mlxrunner/batch/batch.go`
- Modify: `x/models/gemma4/gemma4.go` (`Forward` head, ~line 987)

- [ ] **Step 3.1:** Add to `Batch` (after `Hidden`):

```go
	// InputsEmbeds, when non-nil, is the precomputed input embedding tensor
	// for this forward pass, shape (B, L, hidden) — already embed-scaled,
	// with any multimodal features spliced in. Models that support it skip
	// their token-embedding lookup; InputIDs still carries the real token
	// ids for masks and bookkeeping.
	InputsEmbeds *mlx.Array

	// BidiSpans lists [start, end) absolute prompt positions that attend
	// bidirectionally (image soft-token blocks). Empty for text-only.
	BidiSpans [][2]int32
```

- [ ] **Step 3.2:** In gemma4 `Forward`:

```go
	h := b.InputsEmbeds
	if h == nil {
		h = m.EmbedTokens.Forward(b.InputIDs)
		h = mlx.MulScalar(h, m.EmbedScale)
	}
```

(PLE stays keyed on `b.InputIDs`; all three nvfp4 checkpoints have `hidden_size_per_layer_input: 0` so `computePLEInputs` never runs for them.)

- [ ] **Step 3.3:** `go build ./... && go test ./x/models/... ./x/mlxrunner/...` → PASS. Commit: `feat(mlxrunner): batch-level input embeddings and bidirectional span plumbing`

---

### Task 4: Vision config parse + weight binding in `x/models/gemma4/vision.go`

**Files:**
- Create: `x/models/gemma4/vision.go`
- Test: `x/models/gemma4/vision_test.go`
- Modify: `x/models/gemma4/gemma4.go` (`Model` struct + `newModel` + `LoadWeights`)

- [ ] **Step 4.1: Config test first** (12b + 26b JSON literals asserting parsed fields incl. tokens; a text-only config parses to nil vision). Key structure:

```go
type VisionConfig struct {
	ModelType string `json:"model_type"`
	// Unified embedder (12b).
	MMEmbedDim     int32 `json:"mm_embed_dim"`
	MMPosembSize   int32 `json:"mm_posemb_size"`
	ModelPatchSize int32 `json:"model_patch_size"`
	OutputProjDims int32 `json:"output_proj_dims"`
	// Full tower (26b/31b).
	HiddenSize            int32 `json:"hidden_size"`
	IntermediateSize      int32 `json:"intermediate_size"`
	NumAttentionHeads     int32 `json:"num_attention_heads"`
	HeadDim               int32 `json:"head_dim"`
	NumHiddenLayers       int32 `json:"num_hidden_layers"`
	PositionEmbeddingSize int32 `json:"position_embedding_size"`
	Standardize           bool  `json:"standardize"`
	RopeParameters        struct {
		RopeTheta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`
	// Shared.
	PatchSize         int32   `json:"patch_size"`
	PoolingKernelSize int32   `json:"pooling_kernel_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
}

type multimodalTokens struct {
	BOI   int32 `json:"boi_token_id"`
	EOI   int32 `json:"eoi_token_id"`
	Image int32 `json:"image_token_id"`
}

// parseVisionConfig reads vision_config and the top-level multimodal token
// ids from the raw config bytes. It must NOT be folded into TextConfig:
// parseTextConfig replaces cfg wholesale with the nested text_config.
func parseVisionConfig(configData []byte) (*VisionConfig, multimodalTokens, error)
```

Defaults applied after parse when zero: `PatchSize=16`, `PoolingKernelSize=3`, `RMSNormEps=1e-6`, `RopeTheta=100`, unified `ModelPatchSize=48`. `IsUnified()` helper: `ModelType == "gemma4_unified_vision"`.

- [ ] **Step 4.2:** run → FAIL; implement parse; run → PASS.
- [ ] **Step 4.3: Weight-binding test** (pattern: `TestLoadFusedExpertsQuantized`). Build fake tensor maps with the exact confirmed names and assert bound/nil fields for both families:
  - unified: `model.vision_embedder.patch_ln1.{weight,bias}`, `model.vision_embedder.patch_dense.{weight,bias}` (+ a quantized variant with `model.vision_embedder.patch_dense.weight_scale` asserting `*nn.QuantizedLinear` with non-nil `Bias`), `patch_ln2.*`, `pos_embedding` `[posemb,2,D]` (assert the precomputed X/Y slices land as `[posemb,D]`), `pos_norm.*`, `model.embed_vision.embedding_projection.weight`.
  - tower: `model.vision_tower.patch_embedder.input_proj.weight`, `...position_embedding_table` `[2,S,D]` → X/Y `[S,D]`, `model.vision_tower.std_bias`/`std_scale`, per-layer `model.vision_tower.encoder.layers.0.self_attn.{q,k,v,o}_proj.linear.weight`, `...self_attn.{q,k}_norm.weight`, four block norms, `mlp.{gate,up,down}_proj.linear.weight`.
  - Missing tensor ⇒ explicit error (assert one case).

Module structs:

```go
// VisionEmbedder is the encoder-free 12b path.
type VisionEmbedder struct {
	PatchLN1   *nn.LayerNorm
	PatchDense nn.LinearLayer
	PatchLN2   *nn.LayerNorm
	PosEmbX    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 0, :]
	PosEmbY    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 1, :]
	PosNorm    *nn.LayerNorm
}

type VisionAttention struct {
	QProj, KProj, VProj, OProj nn.LinearLayer
	QNormWeight, KNormWeight   *mlx.Array // per-head-dim RMSNorm weights
}

type VisionLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm *mlx.Array // RMSNorm weights (applied directly, no +1)
	Attn                                           *VisionAttention
	GateProj, UpProj, DownProj                     nn.LinearLayer
}

// VisionTower is the shared 27-layer encoder (26b/31b).
type VisionTower struct {
	InputProj  nn.LinearLayer
	PosTableX  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[0]
	PosTableY  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[1]
	Layers     []*VisionLayer
	NegStdBias *mlx.Array // -std_bias, precomputed; nil unless Standardize
	StdScale   *mlx.Array
}
```

`Model` gains named fields (NOT embedded — `*TextConfig` is already embedded):

```go
	// Vision (nil for text-only checkpoints).
	VisionCfg       *VisionConfig
	MMTokens        multimodalTokens
	VisionEmbedder  *VisionEmbedder
	VisionTower     *VisionTower
	EmbedVisionProj nn.LinearLayer // embed_vision.embedding_projection
```

Loading: `loadVisionWeights(tensors, linears)` called from `LoadWeights` when `m.VisionCfg != nil`; probe prefixes `{"model.", ""}`; slice+`Contiguous`+material-ize the position tables at load (follow `transposeForGatherMM`'s clone/eval discipline); precompute `NegStdBias = mlx.MulScalar(stdBias, -1)`. Note `pos_embedding` slicing: axis-1 index 0/1 → `[S, D]` (12b table is `[S, 2, D]`; tower table is `[2, S, D]` → axis-0).

- [ ] **Step 4.4:** run binding tests (they need MLX: guard with the existing `useMLXTestThread`/`skipIfNoMLX` helpers) → PASS.
- [ ] **Step 4.5:** Wire `newModel`: after `parseTextConfig`, call `parseVisionConfig(configData)`; store on the model. `go test ./x/models/gemma4/` → PASS.
- [ ] **Step 4.6:** Commit: `feat(gemma4): parse vision config and bind vision weights for both lineages`

---

### Task 5: Vision forward passes

**Files:**
- Modify: `x/models/gemma4/vision.go`
- Test: `x/models/gemma4/vision_test.go`

- [ ] **Step 5.1: Numeric tests first** (all under `useMLXTestThread` + synthetic weights):
  - `TestRope2DMatchesReference`: Go float64 naive implementation of `apply_multidimensional_rope` (per-dim slices, 18 freqs, θ=100, duplicate-halves cos/sin, rotate_half within slice) vs the mlx implementation on random `[1, 6, 4, 72]` with positions from a 3×2 grid; `floatSlicesClose(..., 1e-3)`.
  - `TestVisionPoolMatchesNaive`: random `[1, 36, 8]` over a 6×6 patch grid → pool 3×3 → compare to naive per-block average × √8.
  - `TestVisionEmbedderShapes`: 2×2 soft-token grid → input `[1, 4, 6912]` → out `[1, 4, MMEmbedDim]`; after projection `[1, 4, textHidden]`.
  - `TestVisionTowerShapes`: tiny synthetic config (hidden 8, 2 layers, heads 2×head_dim 4 — head_dim must satisfy `2*(d//4)` slicing, use 8) on a 6×6 patch grid → `[1, 4, textHidden]`.
- [ ] **Step 5.2:** run → FAIL (functions absent).
- [ ] **Step 5.3: Implement.** Core forward code:

```go
// visionRope2D applies the reference multidimensional RoPE to x
// [B, L, H, D]: the head dim splits into two per-axis halves, each rotated
// independently with 100-base frequencies of its own grid coordinate.
// cosX/sinX/cosY/sinY are [1, L, 1, D/2], host-precomputed (duplicated
// halves already concatenated).
func visionRope2D(x, cosX, sinX, cosY, sinY *mlx.Array) *mlx.Array {
	d := int32(x.Dim(3))
	half := d / 2
	rot := func(p *mlx.Array) *mlx.Array {
		lo := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(half/2)))
		hi := p.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(half/2), int(half)))
		return mlx.Concatenate([]*mlx.Array{mlx.MulScalar(hi, -1), lo}, 3)
	}
	xX := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, int(half)))
	xY := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(int(half), int(d)))
	outX := mlx.Add(mlx.Mul(xX, cosX), mlx.Mul(rot(xX), sinX))
	outY := mlx.Add(mlx.Mul(xY, cosY), mlx.Mul(rot(xY), sinY))
	return mlx.Concatenate([]*mlx.Array{outX, outY}, 3)
}

// visionRopeTables builds duplicated-half cos/sin for one axis of the patch
// grid: coords is the per-patch coordinate on that axis, halfDim = D/2 (36),
// theta = 100. Layout [1, L, 1, halfDim], values bf16 to match reference.
func visionRopeTables(coords []int32, halfDim int, theta float64) (cos, sin *mlx.Array) {
	quarter := halfDim / 2 // 18 distinct frequencies
	c := make([]float32, len(coords)*halfDim)
	s := make([]float32, len(coords)*halfDim)
	for i, p := range coords {
		for j := 0; j < quarter; j++ {
			ts := math.Pow(theta, 2.0*float64(j)/float64(halfDim))
			a := float64(p) / ts
			c[i*halfDim+j], c[i*halfDim+quarter+j] = float32(math.Cos(a)), float32(math.Cos(a))
			s[i*halfDim+j], s[i*halfDim+quarter+j] = float32(math.Sin(a)), float32(math.Sin(a))
		}
	}
	cos = mlx.FromValues(c, 1, len(coords), 1, halfDim).AsType(mlx.DTypeBFloat16)
	sin = mlx.FromValues(s, 1, len(coords), 1, halfDim).AsType(mlx.DTypeBFloat16)
	return cos, sin
}
```

Tower block (`(t *VisionTower) Forward(patches *mlx.Array, xs, ys []int32, gridW, gridH int32, cfg *VisionConfig) *mlx.Array`):

```go
	h := t.InputProj.Forward(patches) // [1, L, hidden]
	xi := mlx.FromValues(xs, len(xs))
	yi := mlx.FromValues(ys, len(ys))
	pos := mlx.Add(mlx.Take(t.PosTableX, xi, 0), mlx.Take(t.PosTableY, yi, 0))
	h = mlx.Add(h, mlx.ExpandDims(pos, 0))

	cosX, sinX := visionRopeTables(xs, int(cfg.HeadDim)/2, float64(cfg.RopeParameters.RopeTheta))
	cosY, sinY := visionRopeTables(ys, int(cfg.HeadDim)/2, float64(cfg.RopeParameters.RopeTheta))

	for _, l := range t.Layers {
		n := mlx.RMSNormFn(h, l.InputNorm, cfg.RMSNormEps)
		a := l.Attn.Forward(n, cosX, sinX, cosY, sinY, cfg)
		a = mlx.RMSNormFn(a, l.PostAttnNorm, cfg.RMSNormEps)
		h = mlx.Add(h, a)
		n = mlx.RMSNormFn(h, l.PreFFNorm, cfg.RMSNormEps)
		f := l.DownProj.Forward(mlx.GeGLU(l.GateProj.Forward(n), l.UpProj.Forward(n)))
		f = mlx.RMSNormFn(f, l.PostFFNorm, cfg.RMSNormEps)
		h = mlx.Add(h, f)
	}

	// 3×3 average pool over the exact patch grid, f32 like the reference
	// einsum, then ×√hidden.
	k := cfg.PoolingKernelSize
	rB, cB := gridH/k, gridW/k
	D := cfg.HiddenSize
	p := mlx.Reshape(h.AsType(mlx.DTypeFloat32), 1, rB, k, cB, k, D)
	p = mlx.Mean(mlx.Mean(p, 4, false), 2, false) // [1, rB, cB, D]
	h = mlx.Reshape(p, 1, rB*cB, D).AsType(mlx.DTypeBFloat16)
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(D))))

	if t.NegStdBias != nil {
		h = mlx.Mul(mlx.Add(h, t.NegStdBias), t.StdScale)
	}
	return h
```

Tower attention (`Forward(x, cosX, sinX, cosY, sinY, cfg)`):

```go
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	H, D := cfg.NumAttentionHeads, cfg.HeadDim
	q := mlx.Reshape(a.QProj.Forward(x), B, L, H, D)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, H, D)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, H, D)
	q = mlx.RMSNormFn(q, a.QNormWeight, cfg.RMSNormEps)
	k = mlx.RMSNormFn(k, a.KNormWeight, cfg.RMSNormEps)
	v = mlx.RMSNormFn(v, nil, cfg.RMSNormEps)
	q = visionRope2D(q, cosX, sinX, cosY, sinY)
	k = visionRope2D(k, cosX, sinX, cosY, sinY)
	q, k, v = mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3)
	out := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "", nil) // reference: scale=1.0, no mask (grids are exact)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, H*D)
	if !mlx.MetalIsAvailable() {
		out = mlx.Contiguous(out, false)
	}
	return a.OProj.Forward(out)
```

Embedder (`(e *VisionEmbedder) Forward(patches *mlx.Array, xs, ys []int32) *mlx.Array`):

```go
	h := e.PatchLN1.Forward(patches) // eps 1e-5; [1, L, 6912]
	h = e.PatchDense.Forward(h)
	h = e.PatchLN2.Forward(h)
	xi := mlx.FromValues(xs, len(xs))
	yi := mlx.FromValues(ys, len(ys))
	pos := mlx.Add(mlx.Take(e.PosEmbX, xi, 0), mlx.Take(e.PosEmbY, yi, 0))
	h = mlx.Add(h, mlx.ExpandDims(pos, 0))
	return e.PosNorm.Forward(h)
```

Shared projection (`(m *Model) projectVision(h *mlx.Array) *mlx.Array`): `RMSNormFn(h, nil, eps)` → `m.EmbedVisionProj.Forward(h)` (eps = vision `RMSNormEps`, 1e-6).

- [ ] **Step 5.4:** run tests → PASS. Commit: `feat(gemma4): vision forward passes — unified embedder and 27-layer tower`

---

### Task 6: Preprocessor + `base.VisionModel` implementation

**Files:**
- Modify: `x/models/gemma4/vision.go` (+`gemma4.go` for interface assertion)
- Test: `x/models/gemma4/vision_test.go`

- [ ] **Step 6.1: Tests first:**
  - `TestPatchifyLayouts`: hand-built 96×96 gradient image; assert unified patches `[4, 6912]` channel-fastest `(dy, dx, c)` ordering with values in [0,1]; tower patches `[36, 768]` with values `2x−1`; positions row-major `(x,y)`.
  - `TestNewVisionInputSizing`: a 640×480 PNG at default opts → `SoftTokens() == 1066` (or the Task-1-verified count) and grid `41×26`; parity with `llm.ImageTokensForSize("gemma4", opts, w, h) - imageMarkerTokens`… simpler: assert `SoftTokens() == (tw/48)*(th/48)` for the `llm.BudgetFillSize` target. RGBA input decodes (alpha ignored). Junk bytes → error.
- [ ] **Step 6.2:** run → FAIL. Implement:

```go
type visionInput struct {
	patches      []float32 // [n, patchDim] flattened
	n, patchDim  int32     // n patches at patch granularity (48px unified / 16px tower)
	xs, ys       []int32   // per-patch grid coordinates
	gridW, gridH int32     // patch-level grid dims
	soft         int       // soft tokens = (tw/48)*(th/48)
}

func (v *visionInput) SoftTokens() int { return v.soft }

func (m *Model) VisionTokens() (int32, int32, int32) {
	return m.MMTokens.BOI, m.MMTokens.Image, m.MMTokens.EOI
}

func (m *Model) NewVisionInput(data []byte, opts api.Options) (base.VisionInput, error) {
	if m.VisionCfg == nil {
		return nil, errors.New("model has no vision configuration")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	b := img.Bounds()
	_, maxTok := llm.Gemma4ImageBudget(opts)
	tw, th := llm.BudgetFillSize(b.Dx(), b.Dy(), llm.Gemma4ImageAlign, maxTok)

	dst := image.NewRGBA(image.Rect(0, 0, tw, th))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Src, nil)

	p := int(m.VisionCfg.PatchSize) * int(m.VisionCfg.PoolingKernelSize) // 48
	scale, shift := float32(1)/255, float32(0)
	if !m.VisionCfg.IsUnified() {
		p = int(m.VisionCfg.PatchSize) // 16
		scale, shift = 2.0/255, -1     // tower patchify applies 2x−1
	}
	pW, pH := tw/p, th/p
	patchDim := p * p * 3
	patches := make([]float32, pW*pH*patchDim)
	xs, ys := make([]int32, pW*pH), make([]int32, pW*pH)
	for py := 0; py < pH; py++ {
		for px := 0; px < pW; px++ {
			i := py*pW + px
			xs[i], ys[i] = int32(px), int32(py)
			for dy := 0; dy < p; dy++ {
				o := dst.PixOffset(px*p, py*p+dy)
				row := dst.Pix[o : o+p*4]
				for dx := 0; dx < p; dx++ {
					d := i*patchDim + (dy*p+dx)*3
					patches[d+0] = float32(row[dx*4+0])*scale + shift
					patches[d+1] = float32(row[dx*4+1])*scale + shift
					patches[d+2] = float32(row[dx*4+2])*scale + shift
				}
			}
		}
	}
	return &visionInput{patches: patches, n: int32(pW * pH), patchDim: int32(patchDim),
		xs: xs, ys: ys, gridW: int32(pW), gridH: int32(pH),
		soft: (tw / llm.Gemma4ImageAlign) * (th / llm.Gemma4ImageAlign)}, nil
}

func (m *Model) EncodeVision(in base.VisionInput) *mlx.Array {
	vi := in.(*visionInput)
	x := mlx.FromValues(vi.patches, 1, int(vi.n), int(vi.patchDim)).AsType(mlx.DTypeBFloat16)
	var h *mlx.Array
	if m.VisionEmbedder != nil {
		h = m.VisionEmbedder.Forward(x, vi.xs, vi.ys)
	} else {
		h = m.VisionTower.Forward(x, vi.xs, vi.ys, vi.gridW, vi.gridH, m.VisionCfg)
	}
	return m.projectVision(h)
}

func (m *Model) MergedEmbeddings(inputIDs *mlx.Array, features []*mlx.Array, spans [][2]int32) *mlx.Array {
	h := mlx.MulScalar(m.EmbedTokens.Forward(inputIDs), m.EmbedScale)
	for i, f := range features {
		h = h.SliceUpdate(f.AsType(h.DType()), mlx.Slice(), mlx.Slice(int(spans[i][0]), int(spans[i][1])), mlx.Slice())
	}
	return h
}
```

Imports: `bytes`, `image`, `_ "image/gif"`, `_ "image/jpeg"`, `_ "image/png"`, `golang.org/x/image/draw`, `github.com/ollama/ollama/llm`, `github.com/ollama/ollama/api`, `github.com/ollama/ollama/x/mlxrunner/model/base`. Interface assertion `var _ base.VisionModel = (*Model)(nil)` — but ONLY models with vision configs support it; runner must check `m.VisionCfg != nil` too. Resolution: keep the assertion, and make the runner's capability check `vm, ok := r.Model.(base.VisionModel); ok && vm.SupportsVision()` — add `SupportsVision() bool` to the interface returning `m.VisionCfg != nil && (m.VisionEmbedder != nil || m.VisionTower != nil)`. Update the Task-2 `Prepare` guard accordingly.

- [ ] **Step 6.3:** run tests → PASS. Commit: `feat(gemma4): budget-fill image preprocessing and the VisionModel surface`

---

### Task 7: Runner pipeline — expansion, salts, injection, single-chunk prefill

**Files:**
- Modify: `x/mlxrunner/runner.go` (`Request` fields), `x/mlxrunner/pipeline.go`, `x/mlxrunner/prefix_cache.go`
- Test: `x/mlxrunner/media_test.go` (new), `x/mlxrunner/prefix_cache_test.go`

- [ ] **Step 7.1: Extract a pure expansion helper + test it.** New `x/mlxrunner/media.go`:

```go
var imgMarker = regexp.MustCompile(`\[img-(\d+)\]`)

type expandedPrompt struct {
	Tokens []int32
	Spans  [][2]int32 // [start,end) of each image's soft-token block
	Salts  []uint32   // per-token prefix-cache salt; 0 for text
	Inputs []base.VisionInput
}

// expandMedia tokenizes prompt, replacing each [img-N] marker with
// boi + SoftTokens×image + eoi and preprocessing the matching payload.
// encode is segment tokenization (addBOS applies to the first segment only,
// mirroring the non-media path; special tokens break merges, so per-segment
// encoding matches whole-string encoding at these boundaries).
func expandMedia(prompt string, media []llm.MediaData, vm base.VisionModel, opts api.Options,
	encode func(text string, addBOS bool) []int32, addBOS bool) (*expandedPrompt, error) {

	byID := make(map[int]llm.MediaData, len(media))
	for _, m := range media {
		byID[m.ID] = m
	}
	boi, image, eoi := vm.VisionTokens()
	out := &expandedPrompt{}
	appendText := func(s string) {
		if s == "" {
			return
		}
		toks := encode(s, addBOS && len(out.Tokens) == 0)
		out.Tokens = append(out.Tokens, toks...)
		out.Salts = append(out.Salts, make([]uint32, len(toks))...)
	}
	rest := prompt
	for {
		loc := imgMarker.FindStringSubmatchIndex(rest)
		if loc == nil {
			appendText(rest)
			break
		}
		appendText(rest[:loc[0]])
		id, _ := strconv.Atoi(rest[loc[2]:loc[3]])
		m, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("prompt references [img-%d] but no media with that id was provided", id)
		}
		if m.Kind == llm.MediaKindAudio {
			return nil, errors.New("audio input is not supported on the MLX runner")
		}
		in, err := vm.NewVisionInput(m.Data, opts)
		if err != nil {
			return nil, err
		}
		n := in.SoftTokens()
		start := int32(len(out.Tokens) + 1) // after boi
		out.Tokens = append(out.Tokens, boi)
		out.Salts = append(out.Salts, 0)
		sum := sha256.Sum256(m.Data)
		w0, w1 := binary.BigEndian.Uint32(sum[0:4]), binary.BigEndian.Uint32(sum[4:8])
		for i := 0; i < n; i++ {
			out.Tokens = append(out.Tokens, image)
			// Position-mixed digest words: identical images (full trie reuse)
			// diverge from different images at the first soft token.
			out.Salts = append(out.Salts, (w0^uint32(i)*0x9E3779B9)|1)
			_ = w1
		}
		out.Tokens = append(out.Tokens, eoi)
		out.Salts = append(out.Salts, 0)
		out.Spans = append(out.Spans, [2]int32{start, start + int32(n)})
		out.Inputs = append(out.Inputs, in)
		rest = rest[loc[1]:]
	}
	return out, nil
}
```

(Refine the salt mix during implementation — the requirements are: nonzero at image positions, deterministic per image content, position-dependent; use w1 in the mix or drop it.) Test with a fake `VisionModel` + fake `encode` (returns one token per rune, say): marker replacement, span/salt alignment, multiple images, unknown id error, audio rejection, text-only passthrough (nil result unchanged path).

- [ ] **Step 7.2: Prefix-cache salting.** Change `begin(inputs []int32)` → `begin(inputs []int32, salts []uint32)`; in `key()`, before packing: `t = int32(uint32(t) ^ salt[i])` when salts non-nil (apply to both packed halves in the `draftLookahead` case). Update existing callers/tests with `nil`. New test: two begins with same tokens but different salts share no trie prefix beyond the text prefix; same tokens+salts hit fully.
- [ ] **Step 7.3: Prepare wiring** (`pipeline.go`), replacing the Task-2 guard body:

```go
	if len(request.Media) > 0 {
		vm, ok := r.Model.(base.VisionModel)
		if !ok || !vm.SupportsVision() {
			return errors.New("this model does not support image input on the MLX runner")
		}
		exp, err := expandMedia(request.Prompt, request.Media, vm, request.Options,
			func(s string, bos bool) []int32 { return r.Tokenizer.Encode(s, bos) },
			r.Tokenizer.AddBOS())
		if err != nil {
			return err
		}
		tokens = exp.Tokens
		request.VisionInputs, request.VisionSpans, request.CacheSalts = exp.Inputs, exp.Spans, exp.Salts
	}
```

with `Request` gaining `VisionInputs []base.VisionInput`, `VisionSpans [][2]int32`, `CacheSalts []uint32`. The existing empty/context-length/NumPredict logic then operates on the expanded tokens unchanged (verify the `len(tokens) == 0` check moves after expansion).

- [ ] **Step 7.4: Pipeline injection.** In `TextGenerationPipeline`: `session := r.cache.begin(inputs, request.CacheSalts)`. Compute features + merged embeddings on the MLX thread before prefill:

```go
	var embeds *mlx.Array
	if len(request.VisionInputs) > 0 {
		vm := r.Model.(base.VisionModel)
		feats := make([]*mlx.Array, len(request.VisionInputs))
		for i, vi := range request.VisionInputs {
			feats[i] = vm.EncodeVision(vi)
		}
		ids := mlx.FromValues(inputs, 1, len(inputs))
		embeds = vm.MergedEmbeddings(ids, feats, request.VisionSpans)
		mlx.Pin(embeds)
		defer mlx.Unpin(embeds)
		mlx.Eval(embeds) // settle tower memory before prefill
	}
```

Thread `embeds` + `request.VisionSpans` into `prefill`: media requests prefill in one chunk (`chunk = total-1`, mirroring the reference's `no_chunked_prefill`); each prefill `Batch` gets `InputsEmbeds: embeds.Slice(mlx.Slice(), mlx.Slice(processed, processed+n), mlx.Slice())` and `BidiSpans: request.VisionSpans`. Note the prefix cache may resume mid-prompt (same image ⇒ same salted keys): `SeqOffsets` handles the offset; slice embeds by absolute positions `[position, position+n)`. Decode batches stay untouched (nil embeds, nil spans).

- [ ] **Step 7.5:** `go test ./x/mlxrunner/...` → PASS. Commit: `feat(mlxrunner): expand image markers, salt the prefix cache, and inject vision embeddings`

---

### Task 8: Bidirectional prefill masks in gemma4

**Files:**
- Modify: `x/models/gemma4/gemma4.go` (`Attention.Forward` mask site) + new helper in `vision.go`
- Test: `x/models/gemma4/vision_test.go`

Reference semantics (`language.py _apply_blockwise_bidirectional_overlay`): base causal (windowed on sliding layers) OR same-image-block — the block **overrides** the window. Additive `Intersect` can't express that, so build an explicit array mask when spans are present.

- [ ] **Step 8.1: Test first** — `TestVisionPrefillMask`: L=6, K=6, offset 0, span [1,4), window 2: assert (q=1,k=3) allowed (bidi future), (q=3,k=1) allowed (bidi past beyond window), (q=5,k=0) blocked (window), (q=0,k=5) blocked (causal, outside span), diagonal allowed everywhere. Also offset>0 case (K=offset+L).
- [ ] **Step 8.2: Implement** in `vision.go`:

```go
// visionPrefillMask materializes the additive attention mask for a media
// prefill chunk: causal, optionally sliding-windowed, with each image block
// bidirectional — including across the window, matching the reference
// overlay. Queries are chunk-relative (absolute position = offset + qi).
func visionPrefillMask(L, K, offset, window int, spans [][2]int32, dtype mlx.DType) *mlx.Array {
	neg := float32(math.Inf(-1))
	data := make([]float32, L*K)
	inSpan := func(p int) int {
		for si, s := range spans {
			if p >= int(s[0]) && p < int(s[1]) {
				return si
			}
		}
		return -1
	}
	for qi := 0; qi < L; qi++ {
		q := offset + qi
		qs := inSpan(q)
		for k := 0; k < K; k++ {
			ok := k <= q && (window <= 0 || q-k < window)
			if !ok && qs >= 0 && inSpan(k) == qs {
				ok = true
			}
			if !ok {
				data[qi*K+k] = neg
			}
		}
	}
	return mlx.FromValues(data, 1, 1, L, K).AsType(dtype)
}
```

- [ ] **Step 8.3: Wire into `Attention.Forward`:** where `mask := nn.CausalMask()` is currently built, when `len(b.BidiSpans) > 0 && L > 1` replace the mask with `nn.ArrayMask(...)` memoized per window size via `b.Memo` (key on `window`; full-attention layers use `window = 0`). K = key length for this forward = `offset + L` (empty-history case) or the history's total length — read the exact K from the same source the existing mask/SDPA path uses (`kv.history` length or `k.Dim(2)`); keep the sliding `kv.mask` un-intersected in this branch (the array already encodes the window). Verify how `nn.ScaledDotProductAttention` treats `ArrayMask` with `WithKVHistory` — if the history applier offsets/pads K, align the mask to the applier's K layout (single-chunk media prefill from a salted-cache prefix is the only path that hits this; when in doubt, force `matched=0` for media requests instead and simplify to offset=0 — correctness first, reuse later; document the choice).
- [ ] **Step 8.4:** run vision tests + full `go test ./x/models/... ./x/mlxrunner/...` → PASS. Commit: `feat(gemma4): bidirectional image-block masks for media prefill`

---

### Task 9: End-to-end smoke against a real model

**Files:**
- Create: `x/mlxrunner/vision_e2e_test.go`
- Fixture: `x/models/gemma4/testdata/vision_fixture.png` (generate: 384×384 PNG, solid red circle on white — unambiguous content)

- [ ] **Step 9.1:** Test gated on model availability (`t.Skip` unless the manifest for `OLLAMA_VISION_E2E_MODEL` (default `gemma4:12b-nvfp4`) exists under the models-mlx root and MLX is available): load Runner, Prepare a request with `Prompt: "<bos><|turn>user\n[img-0] What shape is in this image? Answer with one word.<turn|>\n<|turn>model\n"`, Media = fixture bytes, run the pipeline, assert the streamed content mentions "circle" (case-insensitive). Budget opts default. This validates tokens→tower→injection→mask→decode as a whole.
- [ ] **Step 9.2:** Run for 12b (unified) and — hardware permitting — `gemma4:26b-nvfp4` and `31b` via the env var. Record results in the PR/commit message. Commit: `test(mlxrunner): gemma4 vision end-to-end smoke`

---

### Task 10: Golden-vector parity vs mlx-vlm

**Files:**
- Create: `x/models/gemma4/testdata/gen_vision_goldens.py` (uv script), `x/models/gemma4/vision_golden_test.go`, `x/models/gemma4/testdata/vision_goldens_{12b,26b,31b}.json`

Pattern follows `model/renderers/gemma4_reference_test.go`: committed expected values; a documented uv command regenerates them; the Go test skips when the local model manifest is absent.

- [ ] **Step 10.1:** Python generator: loads the **ollama blobs directly** (config.json + vision tensor safetensors from `~/.ollama/models-mlx`), instantiates mlx-vlm's `gemma4.vision.VisionModel` / `gemma4_unified` `VisionEmbedder` + `MultimodalEmbedder` (pip `mlx-vlm @ git+…@main`; PyPI 0.31.3 lacks gemma4_unified), dequantizing/loading nvfp4 tensors via `mx.quantized_matmul`-compatible layers or explicit `mx.dequantize`, runs the fixture image through the **reference** preprocessing pinned to the 004 sizing (resize with PIL BICUBIC to `BudgetFillSize` — do NOT trust mlx-vlm's own sizing; the divergence is documented in `docs/maxusai/upstream-gemma4-sizing-issue.md`), and writes JSON: target size, grid, and the projected features' shape, per-position L2 norms for the first 8 positions, mean/std, and the full first and last feature rows.
- [ ] **Step 10.2:** Go test: same fixture through `NewVisionInput` + `EncodeVision` on real weights; compare stats within bf16-appropriate tolerance (rel ~2e-2 on norms; document the achieved deltas). Skips cleanly when goldens or models are absent.
- [ ] **Step 10.3:** Attempt golden generation on this machine (`uv run --with 'mlx>=0.30' --with 'git+https://github.com/Blaizzy/mlx-vlm@main' testdata/gen_vision_goldens.py`). If the environment can't run it (network/package gaps), commit the script + test with a README note and mark the goldens TODO — the e2e smoke (Task 9) remains the behavioral gate.
- [ ] **Step 10.4:** Commit: `test(gemma4): golden-vector parity harness vs mlx-vlm`

---

### Task 11: Re-allow the vision capability + memory update

**Files:**
- Modify: `server/images.go` (~458), `server/model_list_cache.go` (~394), `server/images_test.go` (~518-549)
- Memory: `~/.claude/projects/-opt-github-MaxusAI-ollama/memory/mlx-vision-image-drop.md`

- [ ] **Step 11.1: Tests first:** update the three `images_test.go` cases ("gemma4 … safetensors suppresses vision and audio" → "… suppresses audio"), `expectedCaps: []model.Capability{model.CapabilityVision}` (keep completion etc. as the cases have them). Run → FAIL.
- [ ] **Step 11.2:** Delete the gemma4-vision `DeleteFunc` block in `filterUnsupportedCapabilities` (keep `suppressAudioCapability` — audio remains unimplemented). Narrow `model_list_cache.go` to `c == model.CapabilityAudio` only.
- [ ] **Step 11.3:** `go test ./server/...` → PASS (including `TestGenerateWithImages` groundwork subtests: novision rejection unchanged; audio rejection unchanged).
- [ ] **Step 11.4:** Update the memory file: images to gemma4 MLX models now work end-to-end (date, branch); audio still 400s; ladder-reachability now enforced by the Go preprocessor (`llm.BudgetFillSize`). Update `MEMORY.md` hook line.
- [ ] **Step 11.5:** Commit: `feat(server): advertise vision for gemma4 MLX models`

---

### Self-review checklist (run after writing, before executing)

1. Spec coverage: user items (1)–(6) map to Tasks 2/7 (wire+injection), 1/6 (preprocessor), 4/5 (forwards), 9/10 (parity), 11 (capability+memory). Bidirectional masks (implied by "vision forward … ported from mlx_vlm") = Task 8.
2. Known deliberate deviations from the user's brief: the "media-rejection guard" lived uncommitted in the cool-williams worktree — applied here as the base commit; `routes_generate_test.go` rejection subtests are capability-layer and stay valid rather than being removed.
3. Risks to re-verify during implementation: `nn.ScaledDotProductAttention` + `ArrayMask` + `WithKVHistory` interaction (Task 8.3 fallback documented); `llm.NewStatusWriter` name; `mlx.Mean` axis semantics after the first reduction; `Memo` API shape; exact `FromValues` int32 overloads.
