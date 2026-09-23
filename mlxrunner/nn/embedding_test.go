package nn

import (
	"testing"

	"github.com/ollama/ollama/mlx"
)

func TestQuantizedEmbeddingAsLinearPreservesGlobalScale(t *testing.T) {
	weight := &mlx.Array{}
	scales := &mlx.Array{}
	qbiases := &mlx.Array{}
	globalScale := &mlx.Array{}

	embedding := &QuantizedEmbedding{
		Weight:      weight,
		Scales:      scales,
		QBiases:     qbiases,
		GlobalScale: globalScale,
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}

	linear, ok := embedding.AsLinear().(*QuantizedLinear)
	if !ok {
		t.Fatalf("AsLinear type = %T, want *QuantizedLinear", embedding.AsLinear())
	}
	if linear.Weight != weight {
		t.Fatalf("AsLinear Weight = %p, want %p", linear.Weight, weight)
	}
	if linear.Scales != scales {
		t.Fatalf("AsLinear Scales = %p, want %p", linear.Scales, scales)
	}
	if linear.QBiases != qbiases {
		t.Fatalf("AsLinear QBiases = %p, want %p", linear.QBiases, qbiases)
	}
	if linear.GlobalScale != globalScale {
		t.Fatalf("AsLinear GlobalScale = %p, want %p", linear.GlobalScale, globalScale)
	}
	if linear.GroupSize != 16 || linear.Bits != 4 || linear.Mode != "nvfp4" {
		t.Fatalf("AsLinear quant params = (%d, %d, %q), want (16, 4, %q)", linear.GroupSize, linear.Bits, linear.Mode, "nvfp4")
	}
}
