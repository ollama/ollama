package model_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"

	_ "github.com/ollama/ollama/model/models" // Import the model package to ensure it is registered
	"github.com/ollama/ollama/model/models/qwen25vl"
)

/*
import torch
import numpy as np
import json
import os
from typing import Optional, Union, Tuple

def save_tensor_for_go(tensor: torch.Tensor,
                       filename: str,
                       create_dir: bool = True,
                       transpose_dims: Optional[Tuple[int, ...]] = None) -> None:
    """
    Save a PyTorch tensor to a JSON file in a format easily readable by Go.

    Args:
        tensor: The PyTorch tensor to save
        filename: Path where the JSON file will be saved
        create_dir: Whether to create the directory if it doesn't exist
        transpose_dims: Optional dimension ordering for transposing before saving
                        (useful when Go expects a different dimension order)
    """
    # Ensure tensor is on CPU and detached from computation graph
    if isinstance(tensor, torch.Tensor):
        tensor_data = tensor.detach().cpu()
    else:
        # Handle numpy arrays or convert other types to tensor
        tensor_data = torch.tensor(tensor)

    # Apply transpose if specified
    if transpose_dims is not None:
        tensor_data = tensor_data.permute(*transpose_dims)

    # Convert to numpy for serialization
    numpy_data = tensor_data.numpy()

    # Create output directory if needed
    if create_dir:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Build the data structure
    data = {
        "shape": list(numpy_data.shape),
        "data": numpy_data.flatten().tolist()
    }

    # Write to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved tensor with shape {numpy_data.shape} to {filename}")
*/

// TensorData represents tensor data in a serializable format
type TensorData struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// TestConfig holds configuration for tensor comparison tests
type TestConfig struct {
	RefFileName string        // Reference file name
	Generator   TensorGenFunc // Function to generate tensor for comparison
	avgDiff     float32       // Custom average difference threshold
	maxDiff     float32       // Custom max difference threshold
}

// TensorGenFunc is a function type that generates a tensor for testing
type TensorGenFunc func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error)

var (
	modelUnderTest = "qwen25vl"
	testImagePath  = "/Users/bruce/Desktop/libertine_menu1.jpg" // TODO: use a URL for this?

	maxVisualizeItems = 10 // Maximum items to visualize when difference exceeds threshold
)

// setup prepares the test environment and returns the context and model
func setup(t *testing.T) (ml.Context, *qwen25vl.Model) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	models := filepath.Join(home, ".ollama", "models")

	m, err := model.New(context.TODO(), filepath.Join(models, "blobs", "sha256-819ff989b3412d10f06bea15f5cb9f8f08b07a0da74a2925fa934b42312bf29f"), ml.BackendParams{NumGPULayers: 99})
	if err != nil {
		t.Fatal(err)
	}

	return m.Backend().NewContext().Input(), m.(*qwen25vl.Model)
}

// LoadTensorFromJSON loads a tensor from a JSON file
func LoadTensorFromJSON(ctx ml.Context, filename string) (ml.Tensor, error) {
	// Read the file
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Unmarshal JSON
	var tensorData TensorData
	if err := json.Unmarshal(jsonData, &tensorData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Create tensor from data
	tensor, err := ctx.FromFloatSlice(tensorData.Data, tensorData.Shape...)
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor from data: %w", err)
	}

	return tensor, nil
}

// ExtractTensorData extracts tensor data as a float32 slice
func ExtractTensorData(ctx ml.Context, t ml.Tensor) ([]float32, error) {
	// Make sure tensor data is computed
	if t.Bytes() == nil {
		ctx.Forward(t).Compute(t)
	}

	var result []float32

	// Handle different tensor data types
	switch t.DType() {
	case ml.DTypeF32:
		// For float32, we can read directly
		result = make([]float32, mul(t.Shape()...))
		if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &result); err != nil {
			return nil, fmt.Errorf("failed to read float32 tensor data: %w", err)
		}

	case ml.DTypeF16, ml.DTypeQ80, ml.DTypeQ40:
		// For other floating point types, convert to float32 first
		f32 := ctx.Input().Empty(ml.DTypeF32, t.Shape()...)
		f32 = t.Copy(ctx, f32)
		return ExtractTensorData(ctx, f32)

	case ml.DTypeI32:
		// For int32, read then convert to float32
		i32Data := make([]int32, mul(t.Shape()...))
		if err := binary.Read(bytes.NewBuffer(t.Bytes()), binary.LittleEndian, &i32Data); err != nil {
			return nil, fmt.Errorf("failed to read int32 tensor data: %w", err)
		}

		// Convert int32 to float32
		result = make([]float32, len(i32Data))
		for i, v := range i32Data {
			result[i] = float32(v)
		}

	default:
		return nil, fmt.Errorf("unsupported tensor data type: %v", t.DType())
	}

	return result, nil
}

// Helper function to multiply all elements in a slice
func mul(values ...int) int {
	result := 1
	for _, v := range values {
		result *= v
	}
	return result
}

// CompareTensors compares two tensors and returns statistics about their differences
func CompareTensors(ctx ml.Context, a, b ml.Tensor) (maxDiff, meanDiff float32, err error) {
	// Extract data from both tensors
	aData, err := ExtractTensorData(ctx, a)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to extract data from tensor a: %w", err)
	}

	bData, err := ExtractTensorData(ctx, b)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to extract data from tensor b: %w", err)
	}

	// Check that tensors have the same number of elements
	if len(aData) != len(bData) {
		return 0, 0, fmt.Errorf("tensors have different number of elements: %d vs %d",
			len(aData), len(bData))
	}

	// Calculate differences
	maxDiff = 0
	sumDiff := float32(0)

	for i := range aData {
		diff := float32(math.Abs(float64(aData[i] - bData[i])))
		sumDiff += diff

		if diff > maxDiff {
			maxDiff = diff
		}
	}

	meanDiff = sumDiff / float32(len(aData))

	return maxDiff, meanDiff, nil
}

// VisualizeTensorDiff shows differences between tensors that exceed a threshold
func VisualizeTensorDiff(t *testing.T, a, b ml.Tensor, threshold float32, maxItems int) {
	aData, err := ExtractTensorData(nil, a)
	if err != nil {
		t.Logf("Failed to extract data from tensor a: %v", err)
		return
	}

	bData, err := ExtractTensorData(nil, b)
	if err != nil {
		t.Logf("Failed to extract data from tensor b: %v", err)
		return
	}

	if len(aData) != len(bData) {
		t.Logf("Tensors have different sizes: %d vs %d", len(aData), len(bData))
		return
	}

	diffCount := 0
	for i := range aData {
		diff := float32(math.Abs(float64(aData[i] - bData[i])))
		if diff > threshold {
			t.Logf("Diff at index %d: %.8f vs %.8f (diff: %.8f)",
				i, aData[i], bData[i], diff)
			diffCount++

			if diffCount >= maxItems {
				t.Logf("... and more %d elements with diff > %.8f)", len(aData), threshold)
				break
			}
		}
	}

	if diffCount == 0 {
		t.Logf("No differences found above threshold %.8f", threshold)
	}
}

// runTensorTest runs a generic tensor comparison test
func runTensorTest(t *testing.T, config TestConfig) {
	ctx, m := setup(t)
	defer ctx.Close()

	// Generate tensor using the provided function
	goTensor, err := config.Generator(ctx, m)
	if err != nil {
		t.Fatalf("%s: Failed to generate tensor: %v", t.Name(), err)
	}
	if goTensor == nil {
		t.Fatalf("%s: Generated tensor is nil", t.Name())
	}

	// Load the Python-generated tensor
	pythonTensor, err := LoadTensorFromJSON(ctx, filepath.Join("testdata", "forward", modelUnderTest, config.RefFileName))
	if err != nil {
		t.Fatalf("%s: Failed to load Python reference tensor: %v", t.Name(), err)
	}

	// TODO: Check shape information

	// Check total number of elements
	goElements := mul(goTensor.Shape()...)
	pyElements := mul(pythonTensor.Shape()...)
	if goElements != pyElements {
		t.Fatalf("%s: Tensor total element count mismatch: Go %d vs Python %d",
			t.Name(), goElements, pyElements)
	}

	// Compare tensors
	maxDiff, meanDiff, err := CompareTensors(ctx, goTensor, pythonTensor)
	if err != nil {
		t.Fatalf("%s: Failed to compare tensors: %v", t.Name(), err)
	}

	t.Logf("%s: Comparison results: max diff=%.8f, mean diff=%.8f",
		t.Name(), maxDiff, meanDiff)

	// If differences are too large, visualize them
	if meanDiff > config.avgDiff || maxDiff > config.maxDiff {
		t.Logf("%s: Differences exceed thresholds (mean: %.8f vs %.8f, max: %.8f vs %.8f), showing details:",
			t.Name(), meanDiff, config.avgDiff, maxDiff, config.maxDiff)
		VisualizeTensorDiff(t, goTensor, pythonTensor, config.avgDiff, maxVisualizeItems)
		t.Fail() // Mark test as failed
	} else {
		t.Logf("%s: Implementation matches Python reference within thresholds (mean: %.8f, max: %.8f)",
			t.Name(), config.avgDiff, config.maxDiff)
	}
}

// getTestGrid returns a standard grid for testing
func getTestGrid() *qwen25vl.Grid {
	return &qwen25vl.Grid{
		Temporal: 1,
		Height:   118,
		Width:    58,
	}
}

// loadTestImage loads the test image data
func loadTestImage(t *testing.T) []byte {
	imageData, err := os.ReadFile(testImagePath)
	if err != nil {
		t.Fatalf("Failed to read image file: %v", err)
	}
	return imageData
}

func TestPositionalEmbedding(t *testing.T) {
	runTensorTest(t, TestConfig{
		RefFileName: "positional_embeddings.json",
		Generator: func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error) {
			return m.PositionalEmbedding(ctx, getTestGrid()), nil
		},
		avgDiff: float32(1e-2),
		maxDiff: float32(1e-3),
	})
}

func TestWindowIndex(t *testing.T) {
	runTensorTest(t, TestConfig{
		RefFileName: "window_index.json",
		Generator: func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error) {
			wi, _ := m.WindowIndex(ctx, getTestGrid())
			return wi, nil
		},
		avgDiff: float32(1e-2),
		maxDiff: float32(1e-3),
	})
}

func TestPixelValues(t *testing.T) {
	runTensorTest(t, TestConfig{
		RefFileName: "pixel_values.json",
		Generator: func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error) {
			imageData := loadTestImage(t)
			pixels, _, err := m.PixelValues(ctx, imageData)
			if err != nil {
				return nil, fmt.Errorf("failed to get pixel values: %v", err)
			}
			return pixels, nil
		},
		avgDiff: float32(0.1),
		maxDiff: float32(0.5),
	})
}

func TestPatchEmbedding(t *testing.T) {
	runTensorTest(t, TestConfig{
		RefFileName: "patch_embeddings.json",
		Generator: func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error) {
			imageData := loadTestImage(t)
			pixels, _, err := m.PixelValues(ctx, imageData)
			if err != nil {
				return nil, fmt.Errorf("failed to get pixel values: %v", err)
			}
			return m.PatchEmbedding.Forward(ctx, pixels, m.VisionModelOptions), nil
		},
		avgDiff: float32(0.1),
		maxDiff: float32(5),
	})
}

func TestEncodeMultimodal(t *testing.T) {
	runTensorTest(t, TestConfig{
		RefFileName: "encode_multimodal.json",
		Generator: func(ctx ml.Context, m *qwen25vl.Model) (ml.Tensor, error) {
			imageData := loadTestImage(t)
			enc, err := m.EncodeMultimodal(ctx, imageData)
			if err != nil {
				return nil, fmt.Errorf("failed to encode multimodal data: %v", err)
			}

			// Convert to tensor
			encTensor, ok := enc.(ml.Tensor)
			if !ok {
				return nil, fmt.Errorf("encoded multimodal data is not a tensor")
			}
			return encTensor, nil
		},
		avgDiff: float32(0.05),
		maxDiff: float32(16.5),
	})
}
