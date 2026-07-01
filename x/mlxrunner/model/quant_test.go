package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Pure shape-math tests below run everywhere, including CI runners without
// the MLX library.

func TestAffineParamsMatchCols(t *testing.T) {
	tests := []struct {
		name                   string
		weightCols, scalesCols int
		groupSize, bits        int
		want                   bool
	}{
		// inDim=128: weightCols = inDim*bits/32, scalesCols = inDim/groupSize.
		{name: "4bit gs64", weightCols: 16, scalesCols: 2, groupSize: 64, bits: 4, want: true},
		{name: "8bit gs32", weightCols: 32, scalesCols: 4, groupSize: 32, bits: 8, want: true},
		{name: "6bit gs64", weightCols: 24, scalesCols: 2, groupSize: 64, bits: 6, want: true},
		{name: "2bit gs64", weightCols: 8, scalesCols: 2, groupSize: 64, bits: 2, want: true},
		{name: "ambiguous ratio matches gs64/4bit", weightCols: 16, scalesCols: 2, groupSize: 64, bits: 4, want: true},
		{name: "ambiguous ratio matches gs32/8bit", weightCols: 16, scalesCols: 2, groupSize: 32, bits: 8, want: true},
		{name: "6bit tensor against 4bit params", weightCols: 24, scalesCols: 2, groupSize: 64, bits: 4, want: false},
		{name: "zero group size", weightCols: 16, scalesCols: 2, groupSize: 0, bits: 4, want: false},
		{name: "zero bits", weightCols: 16, scalesCols: 2, groupSize: 64, bits: 0, want: false},
		{name: "zero weight cols", weightCols: 0, scalesCols: 2, groupSize: 64, bits: 4, want: false},
		{name: "zero scales cols", weightCols: 16, scalesCols: 0, groupSize: 64, bits: 4, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := affineParamsMatchCols(tt.weightCols, tt.scalesCols, tt.groupSize, tt.bits); got != tt.want {
				t.Fatalf("affineParamsMatchCols(%d, %d, %d, %d) = %v, want %v",
					tt.weightCols, tt.scalesCols, tt.groupSize, tt.bits, got, tt.want)
			}
		})
	}
}

func TestAffineBitsForGroupSize(t *testing.T) {
	tests := []struct {
		name                   string
		weightCols, scalesCols int
		groupSize              int
		wantBits               int
		wantOK                 bool
	}{
		// inDim=128 throughout.
		{name: "6bit gs64", weightCols: 24, scalesCols: 2, groupSize: 64, wantBits: 6, wantOK: true},
		{name: "3bit gs64", weightCols: 12, scalesCols: 2, groupSize: 64, wantBits: 3, wantOK: true},
		{name: "5bit gs64", weightCols: 20, scalesCols: 2, groupSize: 64, wantBits: 5, wantOK: true},
		{name: "2bit gs64", weightCols: 8, scalesCols: 2, groupSize: 64, wantBits: 2, wantOK: true},
		{name: "ratio not divisible by group size", weightCols: 24, scalesCols: 2, groupSize: 80, wantOK: false},
		{name: "unsupported bit width", weightCols: 28, scalesCols: 2, groupSize: 64, wantOK: false}, // ratio 448 -> 7 bits
		{name: "zero group size", weightCols: 24, scalesCols: 2, groupSize: 0, wantOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bits, ok := affineBitsForGroupSize(tt.weightCols, tt.scalesCols, tt.groupSize)
			if ok != tt.wantOK || (ok && bits != tt.wantBits) {
				t.Fatalf("affineBitsForGroupSize(%d, %d, %d) = (%d, %v), want (%d, %v)",
					tt.weightCols, tt.scalesCols, tt.groupSize, bits, ok, tt.wantBits, tt.wantOK)
			}
		})
	}
}

func TestInferAffineQuantParamsFromCols(t *testing.T) {
	tests := []struct {
		name                    string
		weightCols, scalesCols  int
		hintGroupSize, hintBits int
		wantGroupSize, wantBits int
		wantOK                  bool
	}{
		// inDim=128 throughout.
		{name: "4bit gs32", weightCols: 16, scalesCols: 4, wantGroupSize: 32, wantBits: 4, wantOK: true},
		{name: "8bit gs64", weightCols: 32, scalesCols: 2, wantGroupSize: 64, wantBits: 8, wantOK: true},
		{name: "ambiguous ratio, 4bit hint", weightCols: 16, scalesCols: 2, hintBits: 4, wantGroupSize: 64, wantBits: 4, wantOK: true},
		{name: "ambiguous ratio, 8bit hint", weightCols: 16, scalesCols: 2, hintBits: 8, wantGroupSize: 32, wantBits: 8, wantOK: true},
		{name: "6bit via group size hint", weightCols: 24, scalesCols: 2, hintGroupSize: 64, hintBits: 4, wantGroupSize: 64, wantBits: 6, wantOK: true},
		{name: "no hints, uninferable ratio", weightCols: 24, scalesCols: 2, wantOK: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			groupSize, bits, ok := inferAffineQuantParamsFromCols(tt.weightCols, tt.scalesCols, tt.hintGroupSize, tt.hintBits)
			if ok != tt.wantOK || groupSize != tt.wantGroupSize || bits != tt.wantBits {
				t.Fatalf("inferAffineQuantParamsFromCols(%d, %d, %d, %d) = (%d, %d, %v), want (%d, %d, %v)",
					tt.weightCols, tt.scalesCols, tt.hintGroupSize, tt.hintBits,
					groupSize, bits, ok, tt.wantGroupSize, tt.wantBits, tt.wantOK)
			}
		})
	}
}

func TestSupportsGatherQMM(t *testing.T) {
	tests := []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 1, want: false},
		{mode: "affine", bits: 2, want: true},
		{mode: "affine", bits: 3, want: true},
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 5, want: true},
		{mode: "affine", bits: 6, want: true},
		{mode: "affine", bits: 7, want: false},
		{mode: "affine", bits: 8, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "nvfp4", bits: 8, want: false},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 8, want: false},
		{mode: "", bits: 4, want: false},
		{mode: "unknown", bits: 4, want: false},
	}

	for _, tt := range tests {
		if got := SupportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("SupportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

func TestResolveLinearQuantParamsNonAffinePassthrough(t *testing.T) {
	// Shape validation only applies to affine mode; other modes must pass
	// through untouched (no MLX needed: the arrays are never inspected).
	tests := []struct {
		mode      string
		groupSize int
		bits      int
	}{
		{mode: "nvfp4", groupSize: 16, bits: 4},
		{mode: "mxfp4", groupSize: 32, bits: 4},
		{mode: "mxfp8", groupSize: 32, bits: 8},
	}

	for _, tt := range tests {
		groupSize, bits, mode := ResolveLinearQuantParams(
			tt.groupSize, tt.bits, tt.mode,
			nil,
			"w", nil, nil,
		)
		if groupSize != tt.groupSize || bits != tt.bits || mode != tt.mode {
			t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (%d, %d, %q)",
				groupSize, bits, mode, tt.groupSize, tt.bits, tt.mode)
		}
	}
}

// MLX-backed tests below quantize real tensors and skip when the MLX library
// is not available.

func quantizeTestWeight(t *testing.T, rows, cols, groupSize, bits int) (qw, scales *mlx.Array) {
	t.Helper()

	vals := make([]float32, rows*cols)
	for i := range vals {
		vals[i] = float32(i%19)/7 - 1
	}
	dense := mlx.FromValues(vals, rows, cols).AsType(mlx.DTypeBFloat16)
	qw, scales, qbiases := mlx.Quantize(dense, groupSize, bits, "affine")
	mlx.Eval(qw, scales, qbiases)
	return qw, scales
}

func TestResolveLinearQuantParamsConsistentMetadata(t *testing.T) {
	skipIfNoMLX(t)

	qw, scales := quantizeTestWeight(t, 8, 128, 64, 4)

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 4, "affine",
		map[string]*TensorQuantInfo{"w": {QuantType: "INT4", GroupSize: 64}},
		"w", qw, scales,
	)
	if groupSize != 64 || bits != 4 || mode != "affine" {
		t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (64, 4, affine)", groupSize, bits, mode)
	}
}

func TestResolveLinearQuantParamsMismatchedMetadataBits(t *testing.T) {
	skipIfNoMLX(t)

	// Mixed-precision checkpoint: tensor quantized at 6 bits, but metadata
	// carries the model-level 4-bit params.
	qw, scales := quantizeTestWeight(t, 8, 128, 64, 6)

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 4, "affine",
		map[string]*TensorQuantInfo{"w": {QuantType: "INT4", GroupSize: 64}},
		"w", qw, scales,
	)
	if groupSize != 64 || bits != 6 || mode != "affine" {
		t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (64, 6, affine)", groupSize, bits, mode)
	}
}

func TestResolveLinearQuantParamsMismatchedMetadataBits2(t *testing.T) {
	skipIfNoMLX(t)

	// A 2-bit packed ratio collides with common 4/8-bit pairs, so the legacy
	// heuristics alone would mis-infer it; the recorded per-tensor group size
	// must be consulted first.
	qw, scales := quantizeTestWeight(t, 8, 128, 64, 2)

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 4, "affine",
		map[string]*TensorQuantInfo{"w": {QuantType: "INT4", GroupSize: 64}},
		"w", qw, scales,
	)
	if groupSize != 64 || bits != 2 || mode != "affine" {
		t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (64, 2, affine)", groupSize, bits, mode)
	}
}

func TestResolveLinearQuantParamsAmbiguousRatioKeepsMetadata(t *testing.T) {
	skipIfNoMLX(t)

	// gs=32/8-bit and gs=64/4-bit have the same packed ratio, so recorded
	// metadata that is shape-consistent is trusted even when it is wrong.
	// This pins the documented limitation: shape validation cannot
	// distinguish parameter pairs with identical packed ratios, and it must
	// never override metadata that passes the shape check.
	qw, scales := quantizeTestWeight(t, 8, 128, 32, 8)

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 4, "affine",
		map[string]*TensorQuantInfo{"w": {QuantType: "INT4", GroupSize: 64}},
		"w", qw, scales,
	)
	if groupSize != 64 || bits != 4 || mode != "affine" {
		t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (64, 4, affine): ambiguous ratios must keep recorded metadata", groupSize, bits, mode)
	}
}

func TestResolveLinearQuantParamsNoMetadataFallsBackToShapes(t *testing.T) {
	skipIfNoMLX(t)

	qw, scales := quantizeTestWeight(t, 8, 128, 32, 4)

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 8, "affine",
		nil,
		"w", qw, scales,
	)
	if groupSize != 32 || bits != 4 || mode != "affine" {
		t.Fatalf("resolved (gs=%d, bits=%d, mode=%q), want (32, 4, affine)", groupSize, bits, mode)
	}
}
