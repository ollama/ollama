package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func shapedArray(t *testing.T, dims ...int) *mlx.Array {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	return mlx.FromValues(make([]float32, n), dims...)
}

// affineShapeConsistent uses packed_cols*32 == scale_cols*groupSize*bits, so the
// last-axis sizes are all that matter. Small proportional shapes exercise the
// same arithmetic as full-size weights.
func TestAffineShapeConsistent(t *testing.T) {
	skipIfNoMLX(t)

	tests := []struct {
		name                  string
		weightCols, scaleCols int
		groupSize, bits       int
		want                  bool
	}{
		{"4bit matches shapes", 16, 2, 64, 4, true}, // 16*32 == 2*64*4
		{"8bit matches shapes", 32, 2, 64, 8, true}, // 32*32 == 2*64*8
		{"4bit metadata on 8bit weight", 32, 2, 64, 4, false},
		{"8bit metadata on 4bit weight", 16, 2, 64, 8, false},
		{"zero group size", 16, 2, 0, 4, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w := shapedArray(t, 4, tc.weightCols)
			s := shapedArray(t, 4, tc.scaleCols)
			if got := affineShapeConsistent(w, s, tc.groupSize, tc.bits); got != tc.want {
				t.Fatalf("affineShapeConsistent(wCols=%d,sCols=%d,g=%d,b=%d) = %v, want %v",
					tc.weightCols, tc.scaleCols, tc.groupSize, tc.bits, got, tc.want)
			}
		})
	}
}

// TestResolveLinearQuantParamsMixedPrecision mirrors a checkpoint whose container
// declares a single global 4-bit quant type even though some layers are packed at
// 8 bits (e.g. gemma-4 QAT: 4-bit attention, 8-bit MLP). The packed shapes are
// ground truth, so the 8-bit layers must be detected as 8-bit despite the
// metadata, while genuinely 4-bit layers keep their metadata params.
func TestResolveLinearQuantParamsMixedPrecision(t *testing.T) {
	skipIfNoMLX(t)

	q4 := map[string]*TensorQuantInfo{
		"attn.weight": {QuantType: "Q4", GroupSize: 64},
		"mlp.weight":  {QuantType: "Q4", GroupSize: 64},
	}

	// 4-bit attention weight: 16*32 == 2*64*4, consistent with the metadata.
	attnW := shapedArray(t, 4, 16)
	attnS := shapedArray(t, 4, 2)
	if g, b, mode := ResolveLinearQuantParams(64, 4, "affine", q4, "attn.weight", attnW, attnS); g != 64 || b != 4 || mode != "affine" {
		t.Fatalf("attention resolved to g=%d b=%d mode=%q, want 64/4/affine", g, b, mode)
	}

	// 8-bit MLP weight: same scale groups, twice the packed columns (32*32 ==
	// 2*64*8). Metadata still claims 4-bit; shape inference must override it.
	mlpW := shapedArray(t, 4, 32)
	mlpS := shapedArray(t, 4, 2)
	if g, b, mode := ResolveLinearQuantParams(64, 4, "affine", q4, "mlp.weight", mlpW, mlpS); g != 64 || b != 8 || mode != "affine" {
		t.Fatalf("MLP resolved to g=%d b=%d mode=%q, want 64/8/affine", g, b, mode)
	}
}
