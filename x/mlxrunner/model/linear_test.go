package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestMakeLinearLayer_DenseFallback_NoWeight(t *testing.T) {
	if MakeLinearLayer(map[string]*mlx.Array{}, "foo", 0, 0, "", nil) != nil {
		t.Error("empty map should return nil")
	}
	if MakeLinearLayer(map[string]*mlx.Array{"foo.bias": {}}, "foo", 0, 0, "", nil) != nil {
		t.Error("bias-only map should return nil")
	}
}

func TestMakeLinearLayer_DenseLinear_NoScales(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	mlx.Eval(w)
	got := MakeLinearLayer(map[string]*mlx.Array{"foo.weight": w}, "foo", 0, 0, "", nil)
	if _, ok := got.(*nn.Linear); !ok {
		t.Errorf("type = %T, want *nn.Linear", got)
	}
}

func TestMakeLinearLayer_DenseLinear_WithBias(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := mlx.FromValues([]float32{0.5, 0.5}, 2)
	mlx.Eval(w, b)
	got := MakeLinearLayer(map[string]*mlx.Array{"foo.weight": w, "foo.bias": b}, "foo", 0, 0, "", nil)
	lin, ok := got.(*nn.Linear)
	if !ok {
		t.Fatalf("type = %T, want *nn.Linear", got)
	}
	if lin.Bias == nil {
		t.Error("Bias not wired")
	}
}

func TestMakeLinearLayer_OllamaNativeQuantized(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)
	got := MakeLinearLayer(map[string]*mlx.Array{
		"foo.weight":       w,
		"foo.weight_scale": scales,
	}, "foo", 32, 8, "affine", nil)
	ql, ok := got.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("type = %T, want *nn.QuantizedLinear", got)
	}
	if ql.Weight != w || ql.Scales != scales {
		t.Error("weight/scales not wired")
	}
	if ql.GroupSize != 32 || ql.Bits != 8 || ql.Mode != "affine" {
		t.Errorf("quant params = (%d, %d, %q), want (32, 8, affine)", ql.GroupSize, ql.Bits, ql.Mode)
	}
}

func TestMakeLinearLayer_OllamaNativeQuantized_WithQBiasAndBias(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	qbias := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	bias := mlx.FromValues([]float32{0.1, 0.2, 0.3, 0.4}, 4)
	mlx.Eval(w, scales, qbias, bias)
	got := MakeLinearLayer(map[string]*mlx.Array{
		"foo.weight":       w,
		"foo.weight_scale": scales,
		"foo.weight_qbias": qbias,
		"foo.bias":         bias,
	}, "foo", 32, 8, "affine", nil)
	ql, ok := got.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("type = %T, want *nn.QuantizedLinear", got)
	}
	if ql.QBiases != qbias || ql.Bias != bias {
		t.Error("qbias/bias not wired")
	}
}

// TestMakeLinearLayer_MLXLMSiblingQuantized confirms that MakeLinearLayer
// accepts the mlx-lm sibling-plural aux naming ("<path>.scales" /
// "<path>.biases") and builds a QuantizedLinear with those tensors wired in.
//
// Without the plural fallback, the layer would silently fall through to a
// dense *nn.Linear and load the U32-packed weight as raw float data.
func TestMakeLinearLayer_MLXLMSiblingQuantized(t *testing.T) {
	skipIfNoMLX(t)

	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	biases := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales, biases)

	tensors := map[string]*mlx.Array{
		"foo.weight": w,
		"foo.scales": scales,
		"foo.biases": biases,
	}

	got := MakeLinearLayer(tensors, "foo", 16, 4, "nvfp4", nil)
	ql, ok := got.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("layer type = %T, want *nn.QuantizedLinear", got)
	}
	if ql.Weight != w {
		t.Error("Weight tensor not wired from tensor map")
	}
	if ql.Scales != scales {
		t.Error("Scales tensor not sourced from sibling-plural key")
	}
	if ql.QBiases != biases {
		t.Error("QBiases tensor not sourced from sibling-plural key")
	}
}

func TestMakeLinearLayer_PerTensorQuantOverride(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)
	tq := map[string]*TensorQuantInfo{"foo.weight": {QuantType: "INT4", GroupSize: 32}}
	got := MakeLinearLayer(map[string]*mlx.Array{
		"foo.weight":       w,
		"foo.weight_scale": scales,
	}, "foo", 16, 4, "nvfp4", tq)
	ql, ok := got.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("type = %T, want *nn.QuantizedLinear", got)
	}
	if ql.GroupSize != 32 || ql.Bits != 4 || ql.Mode != "affine" {
		t.Errorf("quant params = (%d, %d, %q), want (32, 4, affine)", ql.GroupSize, ql.Bits, ql.Mode)
	}
}

func TestMakeLinearLayer_NilTensorQuant_UsesDefaults(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues(make([]uint32, 4*8), 4, 8)
	scales := mlx.FromValues(make([]uint8, 4*2), 4, 2)
	mlx.Eval(w, scales)
	got := MakeLinearLayer(map[string]*mlx.Array{
		"foo.weight":       w,
		"foo.weight_scale": scales,
	}, "foo", 16, 4, "nvfp4", nil)
	ql, ok := got.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("type = %T, want *nn.QuantizedLinear", got)
	}
	if ql.GroupSize != 16 || ql.Bits != 4 || ql.Mode != "nvfp4" {
		t.Errorf("quant params = (%d, %d, %q), want (16, 4, nvfp4)", ql.GroupSize, ql.Bits, ql.Mode)
	}
}

func TestLinearFactory_Make_ProducesLayer(t *testing.T) {
	skipIfNoMLX(t)
	w := mlx.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	mlx.Eval(w)
	tensors := map[string]*mlx.Array{"bar.weight": w}
	factory := NewLinearFactory(tensors, 0, 0, "", nil)
	if factory.Make("bar") == nil {
		t.Error("factory.Make returned nil")
	}
	if MakeLinearLayer(tensors, "bar", 0, 0, "", nil) == nil {
		t.Error("direct MakeLinearLayer returned nil")
	}
}
