package gemma4

import (
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/testutil"
)

// TestLongSeqLayerDriftPerPosition is the regression fixture for the
// long-sequence sliding-window forward pass. It loads a Python activation
// reference produced from a prompt longer than the sliding window and walks
// the MLX forward pass layer by layer, using `CompareLayersPerPosition` to
// find the first (layer, position) where MLX drifts from the reference.
//
// This is the fixture that originally caught the SWA boundary bug fixed by
// the gemma4 implementation commit: prefill scoring of positions past the
// sliding window was attending to the entire prefix instead of only the last
// `window` tokens, which produced a clean drift onset at position
// `sliding_window` of layer 0 that compounded through every subsequent layer.
//
// Set GEMMA4_MODEL_DIR to point at a directory containing gemma-4-e2b base
// weights. Override the reference dump path via GEMMA4_LONG_REFERENCE; the
// test skips when either is unavailable.
//
// To regenerate the reference (typically a 6 KB slice of wiki.test.raw,
// long enough to overflow the 512-token sliding window):
//
//	.venv/bin/python3 x/models/scripts/dump_activations.py \
//	    --model $GEMMA4_MODEL_DIR \
//	    --model-class Gemma4ForConditionalGeneration \
//	    --prompt "$(head -c 6000 /tmp/ollama-bench-data/wiki.test.raw)" \
//	    --skip-logits \
//	    --output /tmp/ollama_ref/gemma-4-e2b/long_activations.safetensors
func TestLongSeqLayerDriftPerPosition(t *testing.T) {
	testutil.SkipIfNoMLX(t)

	refPath := os.Getenv("GEMMA4_LONG_REFERENCE")
	if refPath == "" {
		refPath = "/tmp/ollama_ref/gemma-4-e2b/long_activations.safetensors"
	}
	if _, err := os.Stat(refPath); err != nil {
		t.Skipf("long-seq reference not available at %s; see test docstring for how to regenerate", refPath)
	}

	// Load the model FIRST so base.Weights() pins all model tensors before
	// we load the reference (otherwise the reference tensors would be
	// Sweep'd away by the model load).
	modelDir := testutil.ModelDir(t, "GEMMA4_MODEL_DIR", "models/gemma-4-e2b")
	bm := testutil.LoadModelFromDir(t, modelDir)
	m, ok := bm.(*Model)
	if !ok {
		t.Fatalf("expected *gemma4.Model, got %T", bm)
	}

	// Filter the dump down to just the tensors we compare against (input_ids,
	// embed_tokens, final norm, and one tensor per decoder layer). The full
	// dump captures every submodule — for a >sliding_window prompt that's
	// gigabytes of pinned activations we don't need.
	keep := map[string]bool{
		"input_ids":                         true,
		"model.language_model.embed_tokens": true,
		"model.language_model.norm":         true,
	}
	for i := range m.NumLayers() {
		keep[fmt.Sprintf("model.language_model.layers.%d", i)] = true
	}
	ref := testutil.LoadReferenceFiltered(t, refPath, keep)
	t.Logf("loaded %d reference tensors (filtered from full dump)", len(ref))

	inputIDs := ref["input_ids"]
	if inputIDs == nil {
		t.Fatal("reference is missing input_ids")
	}
	tokens := inputIDs.AsType(mlx.DTypeInt32)
	mlx.Eval(tokens)

	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	t.Logf("input length: %d tokens (sliding window: %d)", L, m.SlidingWindow)

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

		key := fmt.Sprintf("layers.%02d", i) // zero-padded so sort matches layer order
		mlx.Eval(h)
		mlx.Pin(h)
		got[key] = h

		refKey := fmt.Sprintf("model.language_model.layers.%d", i)
		if arr, ok := ref[refKey]; ok {
			want[key] = arr
		} else {
			t.Logf("note: reference missing %q; layer %d will be skipped in compare", refKey, i)
		}
	}
	defer func() {
		for _, a := range got {
			mlx.Unpin(a)
		}
	}()

	// Per-layer cosine-similarity assertion. Threshold is 0.99 rather than
	// the short-prompt 0.999 because long-prompt bf16 accumulation across
	// 35 layers is noisier; 0.99 still catches catastrophic divergence
	// (the original SWA boundary bug drove these layers to ~0.6-0.9).
	const minCos = float32(0.99)
	for i := range m.Layers {
		key := fmt.Sprintf("layers.%02d", i)
		w, ok := want[key]
		if !ok {
			continue
		}
		testutil.CompareArraysCosineSim(t, key, got[key], w, minCos)
	}

	// Per-position drift diagnostic — log only, used to localize a failure
	// from the assertions above.
	loose := testutil.WithTolerance(0.5, 0.05)
	report := testutil.CompareLayersPerPosition(t, got, want, 1, loose)
	report.Summary(t)
	testutil.LogDriftRanks(t, "\nTop 15 drifts (absolute):", report.TopDrifts(15))
	testutil.LogDriftRanks(t, "\nTop 15 drifts (relative to layer median):", report.TopDriftsByRelative(15))
	if layerIdx, pos := report.EarliestOutlierPosition(5); layerIdx >= 0 {
		layer := report.Layers[layerIdx]
		t.Logf("\nEARLIEST OUTLIER: layer %s, position %d (rel_to_median ≥ 5×)", layer.Name, pos)
	}
}
