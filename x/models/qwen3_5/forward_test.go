package qwen3_5

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/testutil"
)

// TestForwardLayersPerPosition validates the Qwen 3.5 MLX forward pass against
// a Python activation reference produced by `x/models/scripts/dump_activations.py`.
// It walks the model layer by layer, captures each layer's output, and asks
// `testutil.CompareLayersPerPosition` to find the first (layer, position)
// where MLX drifts from the reference. This is the second-model validation
// for the MLX porting tooling — Qwen 3.5 has a hybrid attention pattern
// (mix of full and Mamba-style linear attention layers) that exercises a
// different code path from the Gemma 4 fixture.
//
// Setup: set QWEN35_MODEL_DIR to a directory containing Qwen3.5-4B-Base
// safetensors weights. The test skips when the directory or activation
// reference is not present.
//
// To regenerate the reference:
//
//	.venv/bin/python3 x/models/scripts/dump_activations.py \
//	    --model $QWEN35_MODEL_DIR \
//	    --model-class Qwen3_5ForConditionalGeneration \
//	    --prompt "The quick brown fox jumps over the lazy dog. A bright sunny day in the meadow." \
//	    --skip-logits
func TestForwardLayersPerPosition(t *testing.T) {
	testutil.SkipIfNoMLX(t)

	refPath := filepath.Join(testutil.DefaultRefDir("Qwen3.5-4B-Base"), "activations.safetensors")
	if _, err := os.Stat(refPath); err != nil {
		t.Skipf("Qwen3.5 reference not available at %s; see test docstring", refPath)
	}

	modelDir := testutil.ModelDir(t, "QWEN35_MODEL_DIR", "models/Qwen3.5-4B-Base")
	bm := testutil.LoadModelFromDir(t, modelDir)
	m, ok := bm.(*Model)
	if !ok {
		t.Fatalf("expected *qwen3_5.Model, got %T", bm)
	}

	keep := map[string]bool{
		"input_ids":                         true,
		"model.language_model.embed_tokens": true,
		"model.language_model.norm":         true,
	}
	for i := range m.NumLayers() {
		keep[fmt.Sprintf("model.language_model.layers.%d", i)] = true
	}
	ref := testutil.LoadReferenceFiltered(t, refPath, keep)
	t.Logf("loaded %d reference tensors", len(ref))

	inputIDs := ref["input_ids"]
	if inputIDs == nil {
		t.Fatal("reference is missing input_ids")
	}
	tokens := inputIDs.AsType(mlx.DTypeInt32)
	mlx.Eval(tokens)

	dims := tokens.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	t.Logf("input length: %d tokens", L)

	h := m.EmbedTokens.Forward(tokens)
	caches := m.NewCaches()
	defer func() {
		for _, c := range caches {
			if c != nil {
				c.Free()
			}
		}
	}()

	got := make(map[string]*mlx.Array)
	want := make(map[string]*mlx.Array)

	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, c, B, L, m.Config)

		key := fmt.Sprintf("layers.%02d", i)
		mlx.Eval(h)
		mlx.Pin(h)
		got[key] = h

		refKey := fmt.Sprintf("model.language_model.layers.%d", i)
		if arr, ok := ref[refKey]; ok {
			want[key] = arr
		} else {
			t.Logf("note: reference missing %q; layer %d will be skipped", refKey, i)
		}
	}
	defer func() {
		for _, a := range got {
			mlx.Unpin(a)
		}
	}()

	// Strict cosine-similarity assertion per layer. See
	// testutil.CompareArraysCosineSim for why cosine is the right primary
	// check here rather than element-wise tolerance.
	const minCos = float32(0.999)
	for i := range m.Layers {
		key := fmt.Sprintf("layers.%02d", i)
		w, ok := want[key]
		if !ok {
			continue
		}
		testutil.CompareArraysCosineSim(t, key, got[key], w, minCos)
	}

	// Per-position diagnostic — log only, used to localize a failure above.
	report := testutil.CompareLayersPerPosition(t, got, want, 1, testutil.WithTolerance(0.5, 0.05))
	report.Summary(t)
	testutil.LogDriftRanks(t, "\nTop 10 drifts (absolute):", report.TopDrifts(10))
	testutil.LogDriftRanks(t, "\nTop 10 drifts (relative to layer median):", report.TopDriftsByRelative(10))
}
