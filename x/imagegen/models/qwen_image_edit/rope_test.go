//go:build mlx

package qwen_image_edit

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/qwen_image"
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
		fmt.Printf("Skipping qwen_image_edit tests: %v\n", err)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestComputeAxisFreqs verifies frequency computation matches Python reference
func TestComputeAxisFreqs(t *testing.T) {
	theta := float64(10000)

	// Expected values from Python:
	// freqs = 1.0 / (theta ** (np.arange(0, half_dim) / half_dim))
	expectedFreqsT := []float64{
		1.000000000000000, 0.316227766016838, 0.100000000000000, 0.031622776601684,
		0.010000000000000, 0.003162277660168, 0.001000000000000, 0.000316227766017,
	}

	expectedFreqsH_first4 := []float64{
		1.000000000000000, 0.719685673001152, 0.517947467923121, 0.372759372031494,
	}

	expectedFreqsH_last4 := []float64{
		0.000372759372031, 0.000268269579528, 0.000193069772888, 0.000138949549437,
	}

	// Test temporal frequencies (dim=16)
	freqsT := qwen_image.ComputeAxisFreqs(16, theta)
	if len(freqsT) != 8 {
		t.Fatalf("expected 8 temporal frequencies, got %d", len(freqsT))
	}
	for i, expected := range expectedFreqsT {
		if diff := math.Abs(freqsT[i] - expected); diff > 1e-10 {
			t.Errorf("freqsT[%d]: expected %.15f, got %.15f, diff %.2e", i, expected, freqsT[i], diff)
		}
	}

	// Test height/width frequencies (dim=56)
	freqsH := qwen_image.ComputeAxisFreqs(56, theta)
	if len(freqsH) != 28 {
		t.Fatalf("expected 28 height frequencies, got %d", len(freqsH))
	}
	for i, expected := range expectedFreqsH_first4 {
		if diff := math.Abs(freqsH[i] - expected); diff > 1e-10 {
			t.Errorf("freqsH[%d]: expected %.15f, got %.15f, diff %.2e", i, expected, freqsH[i], diff)
		}
	}
	for i, expected := range expectedFreqsH_last4 {
		idx := 24 + i // last 4 of 28
		if diff := math.Abs(freqsH[idx] - expected); diff > 1e-10 {
			t.Errorf("freqsH[%d]: expected %.15f, got %.15f, diff %.2e", idx, expected, freqsH[idx], diff)
		}
	}
}

// TestMakeFreqTable verifies the frequency lookup table for both positive and negative positions
func TestMakeFreqTable(t *testing.T) {
	theta := float64(10000)
	freqsT := qwen_image.ComputeAxisFreqs(16, theta)
	maxIdx := int32(4096)

	// Test positive table
	posTable := qwen_image.MakeFreqTable(maxIdx, freqsT, false)

	// Position 0 should give cos=1, sin=0 for all frequencies
	for i := 0; i < len(freqsT)*2; i += 2 {
		if posTable[0][i] != 1.0 {
			t.Errorf("posTable[0][%d] (cos): expected 1.0, got %f", i, posTable[0][i])
		}
		if posTable[0][i+1] != 0.0 {
			t.Errorf("posTable[0][%d] (sin): expected 0.0, got %f", i+1, posTable[0][i+1])
		}
	}

	// Position 1, first frequency (1.0): angle = 1*1 = 1
	// cos(1) = 0.5403, sin(1) = 0.8415
	if diff := math.Abs(float64(posTable[1][0]) - 0.5403023058681398); diff > 1e-6 {
		t.Errorf("posTable[1][0] (cos): expected 0.5403, got %f", posTable[1][0])
	}
	if diff := math.Abs(float64(posTable[1][1]) - 0.8414709848078965); diff > 1e-6 {
		t.Errorf("posTable[1][1] (sin): expected 0.8415, got %f", posTable[1][1])
	}

	// Test negative table
	negTable := qwen_image.MakeFreqTable(maxIdx, freqsT, true)

	// negTable[4095] corresponds to position -1
	// cos(-1) = cos(1), sin(-1) = -sin(1)
	if diff := math.Abs(float64(negTable[4095][0]) - 0.5403023058681398); diff > 1e-6 {
		t.Errorf("negTable[4095][0] (cos(-1)): expected 0.5403, got %f", negTable[4095][0])
	}
	if diff := math.Abs(float64(negTable[4095][1]) - (-0.8414709848078965)); diff > 1e-6 {
		t.Errorf("negTable[4095][1] (sin(-1)): expected -0.8415, got %f", negTable[4095][1])
	}

	// negTable[4094] corresponds to position -2
	// cos(-2) = cos(2), sin(-2) = -sin(2)
	cos2 := math.Cos(2.0)
	sin2 := math.Sin(2.0)
	if diff := math.Abs(float64(negTable[4094][0]) - cos2); diff > 1e-6 {
		t.Errorf("negTable[4094][0] (cos(-2)): expected %f, got %f", cos2, negTable[4094][0])
	}
	if diff := math.Abs(float64(negTable[4094][1]) - (-sin2)); diff > 1e-6 {
		t.Errorf("negTable[4094][1] (sin(-2)): expected %f, got %f", -sin2, negTable[4094][1])
	}
}

// TestPrepareRoPE_QwenImage verifies qwen_image.PrepareRoPE for single-segment case
func TestPrepareRoPE_QwenImage(t *testing.T) {
	if !mlx.GPUIsAvailable() {
		t.Skip("GPU not available")
	}

	mlx.SetDefaultDeviceCPU()

	// 4x4 patch grid, single image
	imgH, imgW := int32(4), int32(4)
	txtLen := int32(5)
	axesDims := []int32{16, 56, 56}

	cache := qwen_image.PrepareRoPE(imgH, imgW, txtLen, axesDims)
	mlx.Eval(cache.ImgFreqs, cache.TxtFreqs)

	// Check shapes
	imgShape := cache.ImgFreqs.Shape()
	if imgShape[0] != 16 { // 4*4 patches
		t.Errorf("ImgFreqs seq len: expected 16, got %d", imgShape[0])
	}

	// For single image (frame=0), all temporal values should be cos=1, sin=0
	imgFreqsCPU := mlx.AsType(cache.ImgFreqs, mlx.DtypeFloat32)
	mlx.Eval(imgFreqsCPU)
	imgData := imgFreqsCPU.Data()

	// Check first 16 values of patch 0 (temporal cos/sin pairs)
	for i := 0; i < 16; i += 2 {
		cosVal := imgData[i]
		sinVal := imgData[i+1]
		if diff := math.Abs(float64(cosVal - 1.0)); diff > 1e-5 {
			t.Errorf("ImgFreqs[0][%d] (cos): expected 1.0, got %f", i, cosVal)
		}
		if diff := math.Abs(float64(sinVal - 0.0)); diff > 1e-5 {
			t.Errorf("ImgFreqs[0][%d] (sin): expected 0.0, got %f", i+1, sinVal)
		}
	}

	cache.ImgFreqs.Free()
	cache.TxtFreqs.Free()
}

// TestScaleRopePositions verifies the centered position calculation for scale_rope=True
func TestScaleRopePositions(t *testing.T) {
	// For a 4x4 grid with scale_rope=True:
	// hHalf = 2, wHalf = 2
	// hNegCount = 4 - 2 = 2 (positions 0,1 are negative)
	// wNegCount = 4 - 2 = 2 (positions 0,1 are negative)
	//
	// Height positions:
	//   y=0: -(4-2) + 0 = -2
	//   y=1: -(4-2) + 1 = -1
	//   y=2: 2 - 2 = 0
	//   y=3: 3 - 2 = 1
	//
	// Same for width

	pH, pW := int32(4), int32(4)
	hHalf := pH / 2
	wHalf := pW / 2
	hNegCount := pH - hHalf
	wNegCount := pW - wHalf

	expectedH := []int32{-2, -1, 0, 1}
	expectedW := []int32{-2, -1, 0, 1}

	for y := int32(0); y < pH; y++ {
		var hPos int32
		if y < hNegCount {
			hPos = -(pH - hHalf) + y
		} else {
			hPos = y - hNegCount
		}
		if hPos != expectedH[y] {
			t.Errorf("y=%d: expected h_pos=%d, got %d", y, expectedH[y], hPos)
		}
	}

	for x := int32(0); x < pW; x++ {
		var wPos int32
		if x < wNegCount {
			wPos = -(pW - wHalf) + x
		} else {
			wPos = x - wNegCount
		}
		if wPos != expectedW[x] {
			t.Errorf("x=%d: expected w_pos=%d, got %d", x, expectedW[x], wPos)
		}
	}
}

// TestRoPEHeadDimensions verifies the head dimension breakdown
func TestRoPEHeadDimensions(t *testing.T) {
	// axes_dims_rope = [16, 56, 56]
	// Each dimension uses half the values for frequencies
	// So we get: 8 + 28 + 28 = 64 frequency values
	// Each frequency produces cos + sin, so: 64 * 2 = 128 total values per position

	axesDims := []int32{16, 56, 56}
	expectedFreqs := (axesDims[0]/2 + axesDims[1]/2 + axesDims[2]/2)
	expectedHeadDim := expectedFreqs * 2

	if expectedFreqs != 64 {
		t.Errorf("expected 64 frequency values, got %d", expectedFreqs)
	}
	if expectedHeadDim != 128 {
		t.Errorf("expected head_dim=128, got %d", expectedHeadDim)
	}

	// This should match the transformer's attention head dimension
	// hidden_size = 3072, num_heads = 24
	// head_dim = 3072 / 24 = 128
}

