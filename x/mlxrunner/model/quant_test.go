package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestQuantizationParams_Types(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		wantGS   int
		wantBits int
		wantMode string
	}{
		{"nvfp4", "NVFP4", 16, 4, "nvfp4"},
		{"mxfp4", "MXFP4", 32, 4, "mxfp4"},
		{"fp4", "FP4", 64, 4, "affine"},
		{"q4", "Q4", 64, 4, "affine"},
		{"int4", "INT4", 64, 4, "affine"},
		{"mxfp8", "MXFP8", 32, 8, "mxfp8"},
		{"fp8", "FP8", 64, 8, "affine"},
		{"q8", "Q8", 64, 8, "affine"},
		{"int8", "INT8", 64, 8, "affine"},
		{"empty", "", 0, 0, ""},
		{"unknown", "something_unknown", 32, 8, "affine"},
		{"lowercase_nvfp4", "nvfp4", 16, 4, "nvfp4"},
		{"lowercase_int4", "int4", 64, 4, "affine"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gs, bits, mode := QuantizationParams(tc.input)
			if gs != tc.wantGS || bits != tc.wantBits || mode != tc.wantMode {
				t.Errorf("got (%d, %d, %q), want (%d, %d, %q)", gs, bits, mode, tc.wantGS, tc.wantBits, tc.wantMode)
			}
		})
	}
}

func TestTensorQuantParams_Fallback(t *testing.T) {
	t.Run("nil_map", func(t *testing.T) {
		gs, bits, mode, fromTensor := TensorQuantParams(32, 4, "nvfp4", nil, "foo.weight")
		if gs != 32 || bits != 4 || mode != "nvfp4" || fromTensor {
			t.Errorf("got (%d, %d, %q, %v), want (32, 4, nvfp4, false)", gs, bits, mode, fromTensor)
		}
	})
	t.Run("absent_key", func(t *testing.T) {
		gs, bits, mode, fromTensor := TensorQuantParams(32, 4, "nvfp4", map[string]*TensorQuantInfo{"bar.weight": {QuantType: "INT4"}}, "foo.weight")
		if gs != 32 || bits != 4 || mode != "nvfp4" || fromTensor {
			t.Errorf("got (%d, %d, %q, %v), want (32, 4, nvfp4, false)", gs, bits, mode, fromTensor)
		}
	})
	t.Run("empty_quant_type", func(t *testing.T) {
		gs, bits, mode, fromTensor := TensorQuantParams(32, 4, "nvfp4", map[string]*TensorQuantInfo{"foo.weight": {}}, "foo.weight")
		if gs != 0 || bits != 0 || mode != "" || !fromTensor {
			t.Errorf("got (%d, %d, %q, %v), want (0, 0, \"\", true)", gs, bits, mode, fromTensor)
		}
	})
}

func TestTensorQuantParams_UsesQuantTypeWhenPresent(t *testing.T) {
	t.Run("int4_with_group_size_override", func(t *testing.T) {
		gs, bits, mode, fromTensor := TensorQuantParams(0, 0, "", map[string]*TensorQuantInfo{"foo.weight": {QuantType: "INT4", GroupSize: 32}}, "foo.weight")
		if gs != 32 || bits != 4 || mode != "affine" || !fromTensor {
			t.Errorf("got (%d, %d, %q, %v), want (32, 4, affine, true)", gs, bits, mode, fromTensor)
		}
	})
	t.Run("nvfp4_no_group_size_override", func(t *testing.T) {
		gs, bits, mode, fromTensor := TensorQuantParams(0, 0, "", map[string]*TensorQuantInfo{"foo.weight": {QuantType: "NVFP4"}}, "foo.weight")
		if gs != 16 || bits != 4 || mode != "nvfp4" || !fromTensor {
			t.Errorf("got (%d, %d, %q, %v), want (16, 4, nvfp4, true)", gs, bits, mode, fromTensor)
		}
	})
}

// TestResolveLinearQuantParams_PerTensorHit confirms non-affine mode skips
// shape inference entirely.
func TestResolveLinearQuantParams_PerTensorHit(t *testing.T) {
	tq := map[string]*TensorQuantInfo{"foo.weight": {QuantType: "NVFP4"}}
	gs, bits, mode := ResolveLinearQuantParams(0, 0, "", tq, "foo.weight", nil, nil)
	if gs != 16 || bits != 4 || mode != "nvfp4" {
		t.Errorf("got (%d, %d, %q), want (16, 4, nvfp4)", gs, bits, mode)
	}
}

// TestResolveLinearQuantParams_AffineShapeInference exercises the shape-based
// fallback for affine mode: weight [4, 16] + scales [4, 2] hits the ambiguous
// `groupSize4==64 && groupSize8==32` case (gs4=128/2=64, gs8=64/2=32).
// hintBits=4 resolves it to (64, 4, "affine").
func TestResolveLinearQuantParams_AffineShapeInference(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*16), 4, 16)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)

	gs, bits, mode := ResolveLinearQuantParams(0, 4, "affine", nil, "foo.weight", w, scales)
	if gs != 64 || bits != 4 || mode != "affine" {
		t.Errorf("got (%d, %d, %q), want (64, 4, affine)", gs, bits, mode)
	}
}

func TestInferAffineQuantParamsFromShapes(t *testing.T) {
	skipIfNoMLX(t)
	makeArray := func(shape ...int) *mlx.Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		a := mlx.FromValues(make([]uint32, n), shape...)
		mlx.Eval(a)
		return a
	}

	cases := []struct {
		name        string
		weightShape []int
		scaleShape  []int
		hintBits    int
		wantGS      int
		wantBits    int
		wantOK      bool
	}{
		{"int4_groupSize32", []int{4, 8}, []int{4, 2}, 0, 32, 4, true},
		// Triggers the `groupSize8 == 64` switch arm: weight cols 16 / scale cols 1
		// gives groupSize4=128 (not 32) and groupSize8=64.
		{"int8_groupSize64", []int{4, 16}, []int{4, 1}, 0, 64, 8, true},
		{"int4_groupSize16_via_isCommon", []int{4, 4}, []int{4, 2}, 0, 16, 4, true},
		{"int8_groupSize128_via_isCommon", []int{4, 32}, []int{4, 1}, 0, 128, 8, true},
		{"ambiguous_hint_4", []int{4, 16}, []int{4, 2}, 4, 64, 4, true},
		{"ambiguous_hint_8", []int{4, 16}, []int{4, 2}, 8, 32, 8, true},
		{"ambiguous_hint_0", []int{4, 16}, []int{4, 2}, 0, 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := makeArray(tc.weightShape...)
			scales := makeArray(tc.scaleShape...)
			gs, bits, ok := InferAffineQuantParamsFromShapes(w, scales, tc.hintBits)
			if gs != tc.wantGS || bits != tc.wantBits || ok != tc.wantOK {
				t.Errorf("got (%d, %d, %v), want (%d, %d, %v)", gs, bits, ok, tc.wantGS, tc.wantBits, tc.wantOK)
			}
		})
	}

	t.Run("nil_weight", func(t *testing.T) {
		scales := makeArray(4, 2)
		if gs, bits, ok := InferAffineQuantParamsFromShapes(nil, scales, 0); gs != 0 || bits != 0 || ok {
			t.Errorf("nil weight should return zero")
		}
	})
	t.Run("nil_scales", func(t *testing.T) {
		w := makeArray(4, 8)
		if gs, bits, ok := InferAffineQuantParamsFromShapes(w, nil, 0); gs != 0 || bits != 0 || ok {
			t.Errorf("nil scales should return zero")
		}
	})
}
