package gemma4

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/testutil"
)

// Forward-pass validation against a PyTorch reference.
//
// Set GEMMA4_MODEL_DIR to point at a directory containing gemma-4-e2b base
// weights (config.json + safetensors). The test skips when the directory is
// not present.
//
// To regenerate the reference (one-time, then cached on disk):
//
//	.venv/bin/python3 x/models/scripts/dump_activations.py \
//	    --model $GEMMA4_MODEL_DIR \
//	    --model-class Gemma4ForConditionalGeneration \
//	    [--transformers-path .tmp/transformers-5.5/transformers/src]
//
// Output: /tmp/ollama_ref/gemma-4-e2b/activations.safetensors

const defaultModelDir = "models/gemma-4-e2b"

func loadRefAndModel(t *testing.T) (ref map[string]*mlx.Array, m *Model) {
	t.Helper()

	// Load model FIRST: base.Weights() calls mlx.Sweep() which would
	// destroy any unpinned arrays, including reference tensors.
	modelDir := testutil.ModelDir(t, "GEMMA4_MODEL_DIR", defaultModelDir)
	bm := testutil.LoadModelFromDir(t, modelDir)

	var ok bool
	m, ok = bm.(*Model)
	if !ok {
		t.Fatalf("expected *gemma4.Model, got %T", bm)
	}

	refPath := filepath.Join(testutil.DefaultRefDir("gemma-4-e2b"), "activations.safetensors")
	ref = testutil.LoadReference(t, refPath)
	return ref, m
}

// refKey returns the activation key used by the Python dumper. The hooks see
// the full module path inside the multimodal wrapper, so layer outputs live
// under "model.language_model.<suffix>".
func refKey(suffix string) string {
	return "model.language_model." + suffix
}

// TestForwardFinalHidden runs the full Forward() and asserts cosine
// similarity (>= 0.999) against Python's final-norm output. Catches model
// bugs that survive an early-layer check but are introduced later in the
// stack.
func TestForwardFinalHidden(t *testing.T) {
	testutil.SkipIfNoMLX(t)
	ref, m := loadRefAndModel(t)

	tokens := ref["input_ids"].AsType(mlx.DTypeInt32)
	mlx.Eval(tokens)

	caches := m.NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()
	h := m.Forward(tokens, caches)
	mlx.Eval(h)

	want, ok := ref[refKey("norm")]
	if !ok {
		t.Skip("reference is missing final norm output")
	}
	testutil.CompareArraysCosineSim(t, "final_hidden", h, want, 0.999)
}

// TestForwardEmbedding compares the (scaled) embedding output against the
// reference. Gemma4TextScaledWordEmbedding always multiplies the lookup by
// sqrt(hidden_size), so the Python hook sees the scaled values.
//
// This is a single-op test (embedding lookup + scalar multiply), so tight
// bf16 element-wise tolerance is appropriate.
func TestForwardEmbedding(t *testing.T) {
	testutil.SkipIfNoMLX(t)
	ref, m := loadRefAndModel(t)

	tokens := ref["input_ids"].AsType(mlx.DTypeInt32)
	mlx.Eval(tokens)

	h := m.EmbedTokens.Forward(tokens)
	hScaled := mlx.MulScalar(h, m.EmbedScale)
	mlx.Eval(hScaled)

	want, ok := ref[refKey("embed_tokens")]
	if !ok {
		t.Skip("reference is missing embed_tokens")
	}
	testutil.CompareArrays(t, "embed_scaled", hScaled, want, testutil.BFloat16Tol())
}

// TestForwardLayersPerPosition replicates Forward() layer by layer and
// asserts cosine similarity (>= 0.999) against the reference at each
// layer. Logs a per-position drift report to localize a failure.
func TestForwardLayersPerPosition(t *testing.T) {
	testutil.SkipIfNoMLX(t)
	ref, m := loadRefAndModel(t)

	tokens := ref["input_ids"].AsType(mlx.DTypeInt32)
	mlx.Eval(tokens)
	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	h := m.EmbedTokens.Forward(tokens)
	h = mlx.MulScalar(h, m.EmbedScale)

	var perLayerInputs *mlx.Array
	if m.HiddenSizePerLayer > 0 && m.EmbedTokensPerLayer != nil {
		perLayerInputs = m.computePLEInputs(tokens, h)
	}

	caches := m.NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()

	var sharedKV map[int32]sharedKVEntry
	if len(m.KVShareMap) > 0 {
		sharedKV = make(map[int32]sharedKVEntry)
	}

	var smc *slidingMaskCache
	if L > 1 && m.SlidingWindow > 0 {
		smc = &slidingMaskCache{}
	}

	got := make(map[string]*mlx.Array)
	want := make(map[string]*mlx.Array)

	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}

		var pleInput *mlx.Array
		if perLayerInputs != nil {
			pleInput = sliceLayerDim(perLayerInputs, int32(i), B, L, m.HiddenSizePerLayer)
		}

		var donorEntry *sharedKVEntry
		if layer.KVShareDonor >= 0 && sharedKV != nil {
			if entry, ok := sharedKV[layer.KVShareDonor]; ok {
				donorEntry = &entry
			}
		}

		var donorKV *sharedKVEntry
		h, donorKV = layer.Forward(h, c, B, L, m.TextConfig, pleInput, donorEntry, smc)

		if layer.IsDonor && donorKV != nil && sharedKV != nil {
			sharedKV[layer.LayerIdx] = *donorKV
		}

		key := fmt.Sprintf("layers.%02d", i)
		mlx.Eval(h)
		mlx.Pin(h)
		got[key] = h

		refName := fmt.Sprintf("layers.%d", i)
		if arr, ok := ref[refKey(refName)]; ok {
			want[key] = arr
		}
	}
	defer func() {
		for _, a := range got {
			mlx.Unpin(a)
		}
	}()

	// Strict cosine-similarity assertion per layer.
	const minCos = float32(0.999)
	for i := range m.Layers {
		key := fmt.Sprintf("layers.%02d", i)
		w, ok := want[key]
		if !ok {
			continue
		}
		testutil.CompareArraysCosineSim(t, key, got[key], w, minCos)
	}

	// Per-position diagnostic — log only, used when an assertion above fails
	// to localize where things went wrong.
	loose := testutil.WithTolerance(0.5, 0.05)
	report := testutil.CompareLayersPerPosition(t, got, want, 1, loose)
	report.Summary(t)
	testutil.LogDriftRanks(t, "\nTop 10 drifts (absolute):", report.TopDrifts(10))
	if layerIdx, pos := report.EarliestOutlierPosition(5); layerIdx >= 0 {
		layer := report.Layers[layerIdx]
		t.Logf("\nEARLIEST OUTLIER: layer %s, position %d (rel_to_median ≥ 5×)", layer.Name, pos)
	}
}
