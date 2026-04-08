package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func approxEqual(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) < tol
}

// TestLayerNormNoBias verifies LayerNorm without bias against manual computation.
func TestLayerNormNoBias(t *testing.T) {
	skipIfNoMLX(t)

	// Input: [1, 4] — single row, 4 features
	x := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := mlx.FromValues([]float32{1, 1, 1, 1}, 4)
	mlx.Eval(x, weight)

	ln := &LayerNorm{Weight: weight, Eps: 1e-5}
	out := ln.Forward(x)
	mlx.Eval(out)

	data := out.Floats()
	if len(data) != 4 {
		t.Fatalf("expected 4 values, got %d", len(data))
	}

	// Manual LayerNorm: mean=2.5, var=1.25, std=sqrt(1.25+1e-5)
	// normalized = (x - mean) / std
	mean := float32(2.5)
	variance := float32(1.25)
	std := float32(math.Sqrt(float64(variance + 1e-5)))
	for i, v := range []float32{1, 2, 3, 4} {
		expected := (v - mean) / std
		if !approxEqual(data[i], expected, 1e-4) {
			t.Errorf("index %d: expected %.6f, got %.6f", i, expected, data[i])
		}
	}
}

// TestLayerNormWithBias verifies LayerNorm with weight and bias.
func TestLayerNormWithBias(t *testing.T) {
	skipIfNoMLX(t)

	x := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := mlx.FromValues([]float32{2, 2, 2, 2}, 4)
	bias := mlx.FromValues([]float32{10, 20, 30, 40}, 4)
	mlx.Eval(x, weight, bias)

	ln := &LayerNorm{Weight: weight, Bias: bias, Eps: 1e-5}
	out := ln.Forward(x)
	mlx.Eval(out)

	data := out.Floats()
	if len(data) != 4 {
		t.Fatalf("expected 4 values, got %d", len(data))
	}

	mean := float32(2.5)
	variance := float32(1.25)
	std := float32(math.Sqrt(float64(variance + 1e-5)))
	biases := []float32{10, 20, 30, 40}
	for i, v := range []float32{1, 2, 3, 4} {
		expected := ((v-mean)/std)*2 + biases[i]
		if !approxEqual(data[i], expected, 1e-4) {
			t.Errorf("index %d: expected %.6f, got %.6f", i, expected, data[i])
		}
	}
}

// TestLayerNormBatched verifies LayerNorm normalizes each row independently.
func TestLayerNormBatched(t *testing.T) {
	skipIfNoMLX(t)

	// Input: [2, 3] — two rows
	x := mlx.FromValues([]float32{
		1, 2, 3,
		10, 20, 30,
	}, 2, 3)
	weight := mlx.FromValues([]float32{1, 1, 1}, 3)
	mlx.Eval(x, weight)

	ln := &LayerNorm{Weight: weight, Eps: 1e-5}
	out := ln.Forward(x)
	mlx.Eval(out)

	data := out.Floats()
	if len(data) != 6 {
		t.Fatalf("expected 6 values, got %d", len(data))
	}

	// Each row should be independently normalized.
	// Row 0: [1,2,3] mean=2, var=2/3
	// Row 1: [10,20,30] mean=20, var=200/3
	// After normalization both rows should have the same pattern
	// since [10,20,30] = 10*[1,2,3], the normalized values are identical.
	for i := range 3 {
		if !approxEqual(data[i], data[i+3], 1e-4) {
			t.Errorf("row 0 elem %d (%.6f) != row 1 elem %d (%.6f); expected identical normalized values",
				i, data[i], i, data[i+3])
		}
	}

	// Verify the normalized values sum to ~0 (mean-centered)
	sum := data[0] + data[1] + data[2]
	if !approxEqual(sum, 0, 1e-4) {
		t.Errorf("normalized row sum should be ~0, got %.6f", sum)
	}
}

// TestLayerNormDefaultEps verifies the default epsilon of 1e-5 is used when Eps is 0.
func TestLayerNormDefaultEps(t *testing.T) {
	skipIfNoMLX(t)

	x := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 4)
	weight := mlx.FromValues([]float32{1, 1, 1, 1}, 4)
	mlx.Eval(x, weight)

	// Eps=0 should use default 1e-5
	ln0 := &LayerNorm{Weight: weight, Eps: 0}
	out0 := ln0.Forward(x)
	mlx.Eval(out0)

	lnExplicit := &LayerNorm{Weight: weight, Eps: 1e-5}
	outExplicit := lnExplicit.Forward(x)
	mlx.Eval(outExplicit)

	d0 := out0.Floats()
	dE := outExplicit.Floats()
	for i := range d0 {
		if !approxEqual(d0[i], dE[i], 1e-6) {
			t.Errorf("index %d: Eps=0 gave %.6f, Eps=1e-5 gave %.6f", i, d0[i], dE[i])
		}
	}
}

func TestQuantizedLinearMXFP4MatchesDequantizedWeight(t *testing.T) {
	skipIfNoMLX(t)

	weightVals := make([]float32, 3*32)
	for i := range weightVals {
		weightVals[i] = float32((i%11)-5) / 7
	}
	inputVals := make([]float32, 2*32)
	for i := range inputVals {
		inputVals[i] = float32((i%7)-3) / 5
	}

	weight := mlx.FromValues(weightVals, 3, 32).AsType(mlx.DTypeBFloat16)
	input := mlx.FromValues(inputVals, 2, 32).AsType(mlx.DTypeBFloat16)
	mlx.Eval(weight, input)

	ql := NewQuantizedLinear(weight, nil, 32, 4, "mxfp4")
	if ql.QBiases != nil {
		t.Fatalf("mxfp4 qbiases = %v, want nil", ql.QBiases)
	}

	dequantizedWeight := mlx.Dequantize(ql.Weight, ql.Scales, ql.QBiases, 32, 4, "mxfp4")
	mlx.Eval(dequantizedWeight)

	qOut := ql.Forward(input)
	dOut := NewLinear(dequantizedWeight, nil).Forward(input)
	mlx.Eval(qOut, dOut)

	got := qOut.Floats()
	want := dOut.Floats()
	if len(got) != len(want) {
		t.Fatalf("output length = %d, want %d", len(got), len(want))
	}

	for i := range got {
		if !approxEqual(got[i], want[i], 1e-3) {
			t.Fatalf("output[%d] = %.6f, want %.6f", i, got[i], want[i])
		}
	}
}
