//go:build mlx

package qwen_image

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// TestVAEConfig tests configuration invariants.
func TestVAEConfig(t *testing.T) {
	cfg := defaultVAEConfig()

	// Property: latents_mean and latents_std have z_dim elements
	if int32(len(cfg.LatentsMean)) != cfg.ZDim {
		t.Errorf("latents_mean length != z_dim: %d != %d", len(cfg.LatentsMean), cfg.ZDim)
	}
	if int32(len(cfg.LatentsStd)) != cfg.ZDim {
		t.Errorf("latents_std length != z_dim: %d != %d", len(cfg.LatentsStd), cfg.ZDim)
	}

	// Property: dim_mult defines 4 stages
	if len(cfg.DimMult) != 4 {
		t.Errorf("dim_mult should have 4 stages: got %d", len(cfg.DimMult))
	}

	// Property: temperal_downsample has 3 elements (for 3 transitions)
	if len(cfg.TemperalDownsample) != 3 {
		t.Errorf("temperal_downsample should have 3 elements: got %d", len(cfg.TemperalDownsample))
	}
}

// TestVAELatentsNormalization tests the latent denormalization values.
func TestVAELatentsNormalization(t *testing.T) {
	cfg := defaultVAEConfig()

	// Verify latents_std values are all positive
	for i, std := range cfg.LatentsStd {
		if std <= 0 {
			t.Errorf("latents_std[%d] should be positive: %v", i, std)
		}
	}

	// Verify values are in reasonable range (from actual model)
	for i, mean := range cfg.LatentsMean {
		if math.Abs(float64(mean)) > 5 {
			t.Errorf("latents_mean[%d] seems too large: %v", i, mean)
		}
	}
	for i, std := range cfg.LatentsStd {
		if std > 10 {
			t.Errorf("latents_std[%d] seems too large: %v", i, std)
		}
	}
}

// TestVAEDecoderForward tests full forward pass (integration test).
// Skips if model weights are not available.
func TestVAEDecoderForward(t *testing.T) {
	weightsPath := "../../../weights/Qwen-Image-2512/vae"
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skip("Skipping: model weights not found at " + weightsPath)
	}

	vae := &VAEDecoder{}
	if err := vae.Load(weightsPath); err != nil {
		t.Fatalf("Failed to load VAE decoder: %v", err)
	}
	mlx.Keep(mlx.Collect(vae)...)

	// Small test input: [B, C, T, H, W]
	// After 4 upsampling stages (2x each), H/W multiply by 16
	batchSize := int32(1)
	channels := int32(16)
	frames := int32(1)
	latentH := int32(4)
	latentW := int32(4)

	latents := mlx.RandomNormal([]int32{batchSize, channels, frames, latentH, latentW}, 0)

	// Decode
	out := vae.Decode(latents)
	mlx.Eval(out)

	// Verify output shape: [B, 3, T, H*16, W*16]
	outShape := out.Shape()
	if outShape[0] != batchSize {
		t.Errorf("batch size: got %d, want %d", outShape[0], batchSize)
	}
	if outShape[1] != 3 {
		t.Errorf("channels: got %d, want 3", outShape[1])
	}
	if outShape[2] != frames {
		t.Errorf("frames: got %d, want %d", outShape[2], frames)
	}
	expectedH := latentH * 16 // 4 stages of 2x upsampling
	expectedW := latentW * 16
	if outShape[3] != expectedH || outShape[4] != expectedW {
		t.Errorf("spatial dims: got [%d, %d], want [%d, %d]",
			outShape[3], outShape[4], expectedH, expectedW)
	}

	// Verify output is in valid range (should be clamped to [0, 1] by decode)
	outData := out.Data()
	for i := 0; i < min(100, len(outData)); i++ {
		if math.IsNaN(float64(outData[i])) || math.IsInf(float64(outData[i]), 0) {
			t.Errorf("output[%d] not finite: %v", i, outData[i])
			break
		}
	}
}
