package gemma4

import (
	"bytes"
	"image"
	"image/png"
	"testing"
)

// TestProcessImageHonorsDeclaredBudget pins the resize contract: a
// declared soft-token budget fixes the grid (the embedding reference
// processor, max_soft_tokens=280 for embeddinggemma2, resizes every image
// to the budget's grid), while budget 0 keeps the dynamic per-image
// selection the generative path uses (#18603). The dynamic sweep
// regressed embeddings to the closest-fit grid — a 320x240 photo went
// from the reference's 280 soft tokens to 63, costing 0.02-0.04 cosine
// vs sentence-transformers.
func TestProcessImageHonorsDeclaredBudget(t *testing.T) {
	cfg := &VisionConfig{
		ModelType:         "gemma4_vision",
		PatchSize:         2,
		PoolingKernelSize: 1,
		RMSNormEps:        1e-6,
	}
	cfg.positionEmbeddingSize = 10240

	// 64x64 discriminates the two policies: the declared 70-budget grid
	// is 8x8 patches (64 soft tokens); dynamic selection upscales to the
	// largest covering grid (33x33 patches, 1089 soft tokens).
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 64, 64))); err != nil {
		t.Fatal(err)
	}

	_, _, fixed, err := ProcessImage(buf.Bytes(), cfg, 70)
	if err != nil {
		t.Fatalf("ProcessImage budget 70: %v", err)
	}
	if fixed.NumSoftTokens != 64 {
		t.Errorf("budget 70: soft tokens = %d, want 64 (the declared grid)", fixed.NumSoftTokens)
	}

	_, _, dyn, err := ProcessImage(buf.Bytes(), cfg, 0)
	if err != nil {
		t.Fatalf("ProcessImage budget 0: %v", err)
	}
	if dyn.NumSoftTokens != 1089 {
		t.Errorf("budget 0 (dynamic): soft tokens = %d, want 1089 (largest covering grid)", dyn.NumSoftTokens)
	}
}
