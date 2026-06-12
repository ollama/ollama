//go:build mlx

package qwen_image

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// TestTransformerConfig tests configuration invariants.
func TestTransformerConfig(t *testing.T) {
	cfg := defaultTransformerConfig()

	// Property: hidden_dim = n_heads * head_dim
	if cfg.HiddenDim != cfg.NHeads*cfg.HeadDim {
		t.Errorf("hidden_dim != n_heads * head_dim: %d != %d * %d",
			cfg.HiddenDim, cfg.NHeads, cfg.HeadDim)
	}

	// Property: axes_dims_rope sums to head_dim
	var ropeSum int32
	for _, d := range cfg.AxesDimsRope {
		ropeSum += d
	}
	if ropeSum != cfg.HeadDim {
		t.Errorf("axes_dims_rope sum != head_dim: %d != %d", ropeSum, cfg.HeadDim)
	}

	// Property: in_channels = out_channels * patch_size^2
	expectedIn := cfg.OutChannels * cfg.PatchSize * cfg.PatchSize
	if cfg.InChannels != expectedIn {
		t.Errorf("in_channels != out_channels * patch_size^2: %d != %d", cfg.InChannels, expectedIn)
	}
}

// TestTransformerRoPE tests RoPE frequency computation produces valid values.
func TestTransformerRoPE(t *testing.T) {
	cfg := defaultTransformerConfig()

	// Test with small image dimensions
	imgH, imgW := int32(4), int32(4) // 4x4 latent = 16 patches
	txtLen := int32(5)

	ropeCache := PrepareRoPE(imgH, imgW, txtLen, cfg.AxesDimsRope)
	mlx.Eval(ropeCache.ImgFreqs, ropeCache.TxtFreqs)

	// Verify shapes: [seq_len, head_dim]
	imgSeqLen := imgH * imgW
	if ropeCache.ImgFreqs.Shape()[0] != imgSeqLen {
		t.Errorf("ImgFreqs seq_len: got %d, want %d", ropeCache.ImgFreqs.Shape()[0], imgSeqLen)
	}
	if ropeCache.ImgFreqs.Shape()[1] != cfg.HeadDim {
		t.Errorf("ImgFreqs head_dim: got %d, want %d", ropeCache.ImgFreqs.Shape()[1], cfg.HeadDim)
	}

	if ropeCache.TxtFreqs.Shape()[0] != txtLen {
		t.Errorf("TxtFreqs seq_len: got %d, want %d", ropeCache.TxtFreqs.Shape()[0], txtLen)
	}

	// Verify values are finite
	imgData := ropeCache.ImgFreqs.Data()
	for i := 0; i < min(100, len(imgData)); i++ {
		if math.IsNaN(float64(imgData[i])) || math.IsInf(float64(imgData[i]), 0) {
			t.Errorf("ImgFreqs[%d] not finite: %v", i, imgData[i])
			break
		}
	}
}

// TestTransformerForward tests full forward pass (integration test).
// Skips if model weights are not available.
func TestTransformerForward(t *testing.T) {
	weightsPath := "../../../weights/Qwen-Image-2512/transformer"
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skip("Skipping: model weights not found at " + weightsPath)
	}

	transformer := &Transformer{}
	if err := transformer.Load(weightsPath); err != nil {
		t.Fatalf("Failed to load transformer: %v", err)
	}
	mlx.Keep(mlx.Collect(transformer)...)
	cfg := transformer.Config

	// Small test inputs
	batchSize := int32(1)
	imgH, imgW := int32(4), int32(4)
	imgSeqLen := imgH * imgW
	txtSeqLen := int32(5)

	hiddenStates := mlx.RandomNormal([]int32{batchSize, imgSeqLen, cfg.InChannels}, 0)
	encoderHiddenStates := mlx.RandomNormal([]int32{batchSize, txtSeqLen, cfg.JointAttentionDim}, 0)
	timestep := mlx.NewArray([]float32{0.5}, []int32{batchSize})

	ropeCache := PrepareRoPE(imgH, imgW, txtSeqLen, cfg.AxesDimsRope)

	// Forward pass
	out := transformer.Forward(hiddenStates, encoderHiddenStates, timestep, ropeCache.ImgFreqs, ropeCache.TxtFreqs)
	mlx.Eval(out)

	// Verify output shape: [batch, img_seq_len, in_channels]
	wantShape := []int32{batchSize, imgSeqLen, cfg.InChannels}
	gotShape := out.Shape()
	if gotShape[0] != wantShape[0] || gotShape[1] != wantShape[1] || gotShape[2] != wantShape[2] {
		t.Errorf("output shape: got %v, want %v", gotShape, wantShape)
	}

	// Verify output is finite
	outData := out.Data()
	for i := 0; i < min(100, len(outData)); i++ {
		if math.IsNaN(float64(outData[i])) || math.IsInf(float64(outData[i]), 0) {
			t.Errorf("output[%d] not finite: %v", i, outData[i])
			break
		}
	}
}
