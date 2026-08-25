package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func denseTestWeight(rows, cols int) *mlx.Array {
	out := make([]float32, rows*cols)
	for i := range out {
		out[i] = float32(i%17) / 8
	}
	return mlx.FromValues(out, rows, cols).AsType(mlx.DTypeBFloat16)
}

func TestResolveLinearQuantParamsTrustsConsistentMetadata(t *testing.T) {
	skipIfNoMLX(t)

	qw, scales, _ := mlx.Quantize(denseTestWeight(8, 64), 64, 4, "affine")

	tensorQuant := map[string]*TensorQuantInfo{
		"router.weight": {QuantType: "int4", GroupSize: 64},
	}

	groupSize, bits, mode := ResolveLinearQuantParams(64, 4, "affine", tensorQuant, "router.weight", qw, scales)
	if groupSize != 64 || bits != 4 || mode != "affine" {
		t.Fatalf("got (%d, %d, %q), want (64, 4, affine)", groupSize, bits, mode)
	}
}

// TestResolveLinearQuantParamsOverridesInconsistentMetadata covers the
// GraniteMoe case that broke inference: `ollama create` writes one
// model-wide quant_type/group_size into every tensor's blob metadata, so a
// router kept at a higher bit-width than the rest of the model (a common
// mlx_lm mixed-precision pattern) inherits the wrong (groupSize, bits) from
// that blind default. Calling quantized_matmul with a declared bit-width
// that does not match how the tensor was actually packed produces an
// invalid array (a shape mismatch MLX cannot reconcile), so the actual
// packed shape must win over untrustworthy metadata.
func TestResolveLinearQuantParamsOverridesInconsistentMetadata(t *testing.T) {
	skipIfNoMLX(t)

	// Actually packed at 8 bits, but the blob metadata (from the model's
	// global 4-bit default) wrongly claims 4 bits.
	qw, scales, _ := mlx.Quantize(denseTestWeight(8, 64), 64, 8, "affine")

	tensorQuant := map[string]*TensorQuantInfo{
		"router.weight": {QuantType: "int4", GroupSize: 64},
	}

	groupSize, bits, mode := ResolveLinearQuantParams(64, 4, "affine", tensorQuant, "router.weight", qw, scales)
	if groupSize != 64 || bits != 8 || mode != "affine" {
		t.Fatalf("got (%d, %d, %q), want (64, 8, affine)", groupSize, bits, mode)
	}
}
