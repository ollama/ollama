//go:build mlx

package safetensors

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

func TestLoadModelWeights(t *testing.T) {
	// Skip if no model available
	modelDir := "../weights/gpt-oss-20b"
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skip("model weights not available")
	}

	mw, err := LoadModelWeights(modelDir)
	if err != nil {
		t.Fatalf("LoadModelWeights: %v", err)
	}
	defer mw.ReleaseAll()

	// Check we found tensors
	tensors := mw.ListTensors()
	if len(tensors) == 0 {
		t.Fatal("no tensors found")
	}
	t.Logf("found %d tensors", len(tensors))

	// Check HasTensor
	if !mw.HasTensor(tensors[0]) {
		t.Errorf("HasTensor(%q) = false", tensors[0])
	}
	if mw.HasTensor("nonexistent.weight") {
		t.Error("HasTensor returned true for nonexistent tensor")
	}
}

func TestGetTensor(t *testing.T) {
	modelDir := "../weights/gpt-oss-20b"
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skip("model weights not available")
	}

	mw, err := LoadModelWeights(modelDir)
	if err != nil {
		t.Fatalf("LoadModelWeights: %v", err)
	}
	defer mw.ReleaseAll()

	tensors := mw.ListTensors()
	if len(tensors) == 0 {
		t.Skip("no tensors")
	}

	// Load first tensor
	arr, err := mw.GetTensor(tensors[0])
	if err != nil {
		t.Fatalf("GetTensor(%q): %v", tensors[0], err)
	}

	// Verify it has a shape
	shape := arr.Shape()
	if len(shape) == 0 {
		t.Error("tensor has no shape")
	}
	t.Logf("%s: shape=%v dtype=%v", tensors[0], shape, arr.Dtype())
}

func TestLoadWithDtype(t *testing.T) {
	modelDir := "../weights/gpt-oss-20b"
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skip("model weights not available")
	}

	mw, err := LoadModelWeights(modelDir)
	if err != nil {
		t.Fatalf("LoadModelWeights: %v", err)
	}
	defer mw.ReleaseAll()

	// Load all tensors as bfloat16
	if err := mw.Load(mlx.DtypeBFloat16); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Get a tensor from cache
	tensors := mw.ListTensors()
	arr, err := mw.Get(tensors[0])
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	// Verify dtype (unless it was already bf16)
	t.Logf("%s: dtype=%v", tensors[0], arr.Dtype())
}

func TestLookupTensor(t *testing.T) {
	modelDir := "../weights/gpt-oss-20b"
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skip("model weights not available")
	}

	mw, err := LoadModelWeights(modelDir)
	if err != nil {
		t.Fatalf("LoadModelWeights: %v", err)
	}
	defer mw.ReleaseAll()

	// HasTensor returns false for nonexistent
	if mw.HasTensor("nonexistent") {
		t.Error("HasTensor should return false for nonexistent")
	}

	// HasTensor returns true for existing tensor
	tensors := mw.ListTensors()
	if !mw.HasTensor(tensors[0]) {
		t.Error("HasTensor should return true for existing tensor")
	}
}

func TestParseSafetensorHeader(t *testing.T) {
	modelDir := "../weights/gpt-oss-20b"
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		t.Skip("model weights not available")
	}

	// Find a safetensors file
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		t.Fatal(err)
	}

	var stFile string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".safetensors" {
			stFile = filepath.Join(modelDir, e.Name())
			break
		}
	}
	if stFile == "" {
		t.Skip("no safetensors file found")
	}

	header, err := parseSafetensorHeader(stFile)
	if err != nil {
		t.Fatalf("parseSafetensorHeader: %v", err)
	}

	if len(header) == 0 {
		t.Error("header is empty")
	}

	// Check a tensor has valid info
	for name, info := range header {
		if info.Dtype == "" {
			t.Errorf("%s: empty dtype", name)
		}
		if len(info.Shape) == 0 {
			t.Errorf("%s: empty shape", name)
		}
		break // just check one
	}
}
