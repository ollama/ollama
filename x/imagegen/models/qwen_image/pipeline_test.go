//go:build mlx

package qwen_image

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// TestMain initializes MLX before running tests.
// If MLX libraries are not available, tests are skipped.
func TestMain(m *testing.M) {
	// Change to repo root so ./build/lib/ollama/ path works
	_, thisFile, _, _ := runtime.Caller(0)
	repoRoot := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "..")
	if err := os.Chdir(repoRoot); err != nil {
		fmt.Printf("Failed to change to repo root: %v\n", err)
		os.Exit(1)
	}

	if err := mlx.InitMLX(); err != nil {
		fmt.Printf("Skipping qwen_image tests: %v\n", err)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestPipelineOutput runs the full pipeline (integration test).
// Skips if model weights not found. Requires ~50GB VRAM.
func TestPipelineOutput(t *testing.T) {
	modelPath := "../../../weights/Qwen-Image-2512"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping: model weights not found at " + modelPath)
	}

	// Load model
	pm, err := LoadPersistent(modelPath)
	if err != nil {
		t.Skipf("Skipping: failed to load model: %v", err)
	}

	// Run 2-step pipeline (minimum for stable scheduler)
	cfg := &GenerateConfig{
		Prompt: "a cat",
		Width:  256,
		Height: 256,
		Steps:  2,
		Seed:   42,
	}

	output, err := pm.GenerateFromConfig(cfg)
	if err != nil {
		t.Fatalf("Pipeline failed: %v", err)
	}
	mlx.Eval(output)

	// Verify output shape [1, C, H, W]
	shape := output.Shape()
	if len(shape) != 4 {
		t.Errorf("Expected 4D output, got %v", shape)
	}
	if shape[0] != 1 || shape[1] != 3 || shape[2] != cfg.Height || shape[3] != cfg.Width {
		t.Errorf("Shape mismatch: got %v, expected [1, 3, %d, %d]", shape, cfg.Height, cfg.Width)
	}

	// Verify values in expected range [0, 1]
	data := output.Data()
	minVal, maxVal := float32(1.0), float32(0.0)
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	t.Logf("Output range: [%.4f, %.4f]", minVal, maxVal)

	if minVal < -0.1 || maxVal > 1.1 {
		t.Errorf("Output values out of range: [%.4f, %.4f]", minVal, maxVal)
	}
}
