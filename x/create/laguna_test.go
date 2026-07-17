package create

import (
	"encoding/json"
	"strconv"
	"testing"
)

func TestLagunaImportTransformRegistration(t *testing.T) {
	inv := Inventory{
		Config:    sourceModelConfig{Architectures: []string{"LagunaForCausalLM"}},
		RawConfig: json.RawMessage(`{"num_hidden_layers":40,"mlp_layer_types":["dense","sparse","sparse"]}`),
	}

	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}

	transform, ok := policy.(lagunaImportTransform)
	if !ok {
		t.Fatalf("newTensorImportTransform() = %T, want lagunaImportTransform", policy)
	}
	if !transform.denseMLPLayers[0] || transform.denseMLPLayers[1] {
		t.Fatalf("denseMLPLayers = %v, want only layer 0", transform.denseMLPLayers)
	}
	if transform.numLayers != 40 {
		t.Fatalf("numLayers = %d, want 40", transform.numLayers)
	}
}

func TestLagunaImportTransformSameRecipeAcrossConfigs(t *testing.T) {
	configs := map[string]json.RawMessage{
		"laguna xs.2": json.RawMessage(`{
			"num_hidden_layers": 40,
			"mlp_layer_types": ["dense", "sparse", "sparse"]
		}`),
		"laguna xs 2.1": json.RawMessage(`{
			"num_hidden_layers": 40,
			"mlp_only_layers": [0],
			"gating_types": ["per_head"]
		}`),
	}

	for name, rawConfig := range configs {
		t.Run(name, func(t *testing.T) {
			policy, err := newLagunaImportTransform(rawConfig)
			if err != nil {
				t.Fatal(err)
			}
			testLagunaQuantizationRecipe(t, policy)
		})
	}
}

func TestLagunaPlanExpertGroupUsesStackedDownProjectionPolicy(t *testing.T) {
	policy, err := newLagunaImportTransform(json.RawMessage(`{
		"num_hidden_layers": 40,
		"mlp_layer_types": ["dense", "sparse"]
	}`))
	if err != nil {
		t.Fatal(err)
	}

	inv := Inventory{Tensors: map[string]SourceTensor{}}
	for _, layer := range []int{1, 5} {
		for expert := 0; expert < 2; expert++ {
			for _, projection := range []string{"gate_proj", "down_proj"} {
				name := "model.layers." + strconv.Itoa(layer) + ".mlp.experts." + strconv.Itoa(expert) + "." + projection + ".weight"
				inv.Tensors[name] = SourceTensor{
					Name:  name,
					Dtype: "BF16",
					Shape: []int32{2048, 512},
				}
			}
		}
	}

	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "mxfp8"}, policy)
	if err != nil {
		t.Fatal(err)
	}

	tests := map[string]string{
		"model.layers.1.mlp.experts.down_proj.weight": "",
		"model.layers.1.mlp.experts.gate_proj.weight": "mxfp8",
		"model.layers.5.mlp.experts.down_proj.weight": "mxfp8",
	}
	for tensor, want := range tests {
		if got := quantizeForPlannedTensor(specs, tensor); got != want {
			t.Fatalf("planned quantization for %s = %q, want %q", tensor, got, want)
		}
	}
}

func quantizeForPlannedTensor(specs []BlobSpec, name string) string {
	for _, spec := range specs {
		for _, ts := range spec.Tensors {
			if ts.Name == name {
				return ts.Quantize
			}
		}
	}
	return "<missing>"
}

func testLagunaQuantizationRecipe(t *testing.T, policy quantizePolicy) {
	t.Helper()

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{
			name:     "attention q projection uses requested fp4",
			tensor:   "model.layers.1.self_attn.q_proj.weight",
			shape:    []int32{8192, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention v projection uses requested fp4 before promotion layer",
			tensor:   "model.layers.0.self_attn.v_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention v projection uses requested fp4 on layer 4",
			tensor:   "model.layers.4.self_attn.v_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention v projection uses requested fp4 after promotion layer",
			tensor:   "model.layers.5.self_attn.v_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention k projection uses requested fp4 on input layer",
			tensor:   "model.layers.0.self_attn.k_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention k projection uses requested fp4 past layer 0",
			tensor:   "model.layers.4.self_attn.k_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "attention k projection stays source precision for mxfp8",
			tensor:   "model.layers.4.self_attn.k_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "attention v projection stays source precision for mxfp8",
			tensor:   "model.layers.4.self_attn.v_proj.weight",
			shape:    []int32{1024, 2048},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "attention q projection stays source precision for mxfp8",
			tensor:   "model.layers.4.self_attn.q_proj.weight",
			shape:    []int32{8192, 2048},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "attention o projection stays source precision for mxfp8",
			tensor:   "model.layers.4.self_attn.o_proj.weight",
			shape:    []int32{2048, 8192},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "attention gate projection stays source precision for mxfp8",
			tensor:   "model.layers.4.self_attn.g_proj.weight",
			shape:    []int32{64, 2048},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "attention gate projection uses requested fp4",
			tensor:   "model.layers.1.self_attn.g_proj.weight",
			shape:    []int32{64, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "dense gate projection uses requested fp4",
			tensor:   "model.layers.0.mlp.gate_proj.weight",
			shape:    []int32{8192, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "dense down projection uses requested fp4",
			tensor:   "model.layers.0.mlp.down_proj.weight",
			shape:    []int32{2048, 8192},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "unsupported dense projection in sparse layer stays source precision",
			tensor:   "model.layers.1.mlp.down_proj.weight",
			shape:    []int32{2048, 8192},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "routed expert gate uses requested fp4",
			tensor:   "model.layers.1.mlp.experts.gate_proj.weight",
			shape:    []int32{256, 512, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "routed expert down uses requested fp4 on cadence layer",
			tensor:   "model.layers.1.mlp.experts.down_proj.weight",
			shape:    []int32{256, 2048, 512},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "routed expert down uses requested fp4 on later layer",
			tensor:   "model.layers.5.mlp.experts.down_proj.weight",
			shape:    []int32{256, 2048, 512},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "shared expert gate uses requested fp4",
			tensor:   "model.layers.1.mlp.shared_expert.gate_proj.weight",
			shape:    []int32{512, 2048},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "shared expert down promotes to mxfp8 on input-side layer",
			tensor:   "model.layers.1.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "shared expert down uses requested fp4 before middle cadence",
			tensor:   "model.layers.5.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "shared expert down promotes to mxfp8 on first selected middle layer",
			tensor:   "model.layers.7.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "shared expert down promotes to mxfp8 on early middle layer",
			tensor:   "model.layers.10.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "shared expert down promotes to mxfp8 on last selected middle layer",
			tensor:   "model.layers.13.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "shared expert down uses requested fp4 after selected middle layers",
			tensor:   "model.layers.16.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "shared expert down uses requested fp4 on late middle layer",
			tensor:   "model.layers.19.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "shared expert down promotes to mxfp8 on final layers",
			tensor:   "model.layers.39.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "shared expert down stays source precision for mxfp8 on selected layer",
			tensor:   "model.layers.1.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "shared expert down stays source precision for mxfp8 off selected layers",
			tensor:   "model.layers.5.mlp.shared_expert.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "routed expert down stays source precision for mxfp8 on selected layer",
			tensor:   "model.layers.1.mlp.experts.down_proj.weight",
			shape:    []int32{256, 2048, 512},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "per-expert routed down stays source precision for mxfp8 on selected layer",
			tensor:   "model.layers.1.mlp.experts.0.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "mxfp8",
			want:     "",
		},
		{
			name:     "routed expert down uses mxfp8 off selected layers",
			tensor:   "model.layers.5.mlp.experts.down_proj.weight",
			shape:    []int32{256, 2048, 512},
			quantize: "mxfp8",
			want:     "mxfp8",
		},
		{
			name:     "per-expert routed down uses mxfp8 off selected layers",
			tensor:   "model.layers.5.mlp.experts.0.down_proj.weight",
			shape:    []int32{2048, 512},
			quantize: "mxfp8",
			want:     "mxfp8",
		},
		{
			name:     "router gate stays source precision",
			tensor:   "model.layers.1.mlp.gate.weight",
			shape:    []int32{256, 2048},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "embedding stays source precision",
			tensor:   "model.embed_tokens.weight",
			shape:    []int32{100352, 2048},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "lm head stays source precision",
			tensor:   "lm_head.weight",
			shape:    []int32{100352, 2048},
			quantize: "nvfp4",
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := policy.quantizationType(tt.tensor, tt.shape, tt.quantize); got != tt.want {
				t.Fatalf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}
