package create

import (
	"encoding/json"
	"testing"
)

func TestGraniteImportTransformRegistration(t *testing.T) {
	for _, arch := range []string{"GraniteForCausalLM", "GraniteMoeForCausalLM"} {
		t.Run(arch, func(t *testing.T) {
			inv := Inventory{
				Config:    sourceModelConfig{Architectures: []string{arch}},
				RawConfig: json.RawMessage(`{}`),
			}

			policy, err := newTensorImportTransform(inv)
			if err != nil {
				t.Fatalf("newTensorImportTransform() error = %v", err)
			}

			_, ok := policy.(graniteImportTransform)
			if !ok {
				t.Fatalf("newTensorImportTransform() = %T, want graniteImportTransform", policy)
			}
		})
	}
}

func TestGraniteImportTransformOProjPromotion(t *testing.T) {
	policy, err := newGraniteImportTransform(nil)
	if err != nil {
		t.Fatal(err)
	}

	oProj := "model.layers.0.self_attn.o_proj.weight"
	shape := []int32{2560, 2560}

	tests := []struct {
		name string
		quant string
		want string
	}{
		{"nvfp4 promotes o_proj to mxfp8", "nvfp4", "mxfp8"},
		{"mxfp4 promotes o_proj to mxfp8", "mxfp4", "mxfp8"},
		// int4 is not affected by the granite fp4 underflow fix; the transform
		// falls through to GetTensorQuantization which promotes v/k/down to 8-bit
		// but leaves o_proj at base int4 (since o_proj is not v/k/down).
		{"int4 falls through to generic policy", "int4", "int4"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := policy.quantizationType(oProj, shape, tt.quant)
			if got != tt.want {
				t.Errorf("quantizationType(%q, %v, %q) = %q, want %q", oProj, shape, tt.quant, got, tt.want)
			}
		})
	}
}

func TestGraniteImportTransformOtherTensors(t *testing.T) {
	policy, err := newGraniteImportTransform(nil)
	if err != nil {
		t.Fatal(err)
	}

	// Non-o_proj tensors should fall through to the generic policy unchanged.
	genericTensors := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
	}

	for _, name := range genericTensors {
		t.Run(name, func(t *testing.T) {
			shape := []int32{2560, 2560}
			// For nvfp4, the generic policy promotes v/k/down but not q/gate
			got := policy.quantizationType(name, shape, "nvfp4")
			genericGot := GetTensorQuantization(name, shape, "nvfp4")
			if got != genericGot {
				t.Errorf("quantizationType(%q, %v, nvfp4) = %q, want generic %q", name, shape, got, genericGot)
			}
		})
	}
}

func TestIsStackedExpertWeightGraniteMoe(t *testing.T) {
	tests := []struct {
		name string
		tensor string
		want bool
	}{
		{"block_sparse_moe.input_linear", "model.layers.0.block_sparse_moe.input_linear.weight", true},
		{"block_sparse_moe.output_linear", "model.layers.0.block_sparse_moe.output_linear.weight", true},
		{"block_sparse_moe.switch_mlp.gate_proj", "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight", true},
		{"block_sparse_moe.switch_mlp.up_proj", "model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight", true},
		{"block_sparse_moe.switch_mlp.down_proj", "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight", true},
		{"input_linear with bias suffix", "model.layers.0.block_sparse_moe.input_linear.bias", false},
		{"output_linear with scale suffix", "model.layers.0.block_sparse_moe.output_linear.scale", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isStackedExpertWeight(tt.tensor)
			if got != tt.want {
				t.Errorf("isStackedExpertWeight(%q) = %v, want %v", tt.tensor, got, tt.want)
			}
		})
	}
}

func TestIsRoutingGateGraniteMoe(t *testing.T) {
	tests := []struct {
		name string
		tensor string
		want bool
	}{
		{"block_sparse_moe router", "model.layers.0.block_sparse_moe.router.layer.weight", true},
		{"q_proj is not a routing gate", "model.layers.0.self_attn.q_proj.weight", false},
		{"o_proj is not a routing gate", "model.layers.0.self_attn.o_proj.weight", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRoutingGate(tt.tensor)
			if got != tt.want {
				t.Errorf("isRoutingGate(%q) = %v, want %v", tt.tensor, got, tt.want)
			}
		})
	}
}
