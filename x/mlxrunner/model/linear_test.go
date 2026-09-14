package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

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
