package create

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestNemotronHImportTransformRegistration(t *testing.T) {
	inv := Inventory{
		Config:    sourceModelConfig{Architectures: []string{"NemotronH_Nano_Omni_Reasoning_V3"}},
		RawConfig: json.RawMessage(`{"architectures":["NemotronH_Nano_Omni_Reasoning_V3"],"llm_config":{"num_hidden_layers":52}}`),
	}

	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatalf("newTensorImportTransform() error = %v", err)
	}
	transform, ok := policy.(nemotronHImportTransform)
	if !ok {
		t.Fatalf("newTensorImportTransform() = %T, want nemotronHImportTransform", policy)
	}
	if transform.numLayers != 52 {
		t.Fatalf("numLayers = %d, want 52", transform.numLayers)
	}
}

func TestNemotronHPlanKeepsUnsupportedModalitiesAtSourcePrecision(t *testing.T) {
	transform := nemotronHImportTransform{numLayers: 52}
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"language_model.backbone.embeddings.weight":                  {Name: "language_model.backbone.embeddings.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"language_model.backbone.layers.0.mixer.in_proj.weight":      {Name: "language_model.backbone.layers.0.mixer.in_proj.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"vision_model.radio_model.model.patch_generator.proj.weight": {Name: "vision_model.radio_model.model.patch_generator.proj.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"mlp1.0.weight": {Name: "mlp1.0.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"sound_encoder.encoder.layers.0.self_attn.q_proj.weight": {Name: "sound_encoder.encoder.layers.0.self_attn.q_proj.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"sound_projection.adapter.weight":                        {Name: "sound_projection.adapter.weight", Dtype: "BF16", Shape: []int32{128, 128}},
	}}

	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "nvfp4"}, transform)
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	got := specNames(specs)
	want := []string{
		"language_model.backbone.embeddings.weight",
		"language_model.backbone.layers.0.mixer.in_proj.weight",
		"mlp1.0.weight",
		"sound_encoder.encoder.layers.0.self_attn.q_proj.weight",
		"sound_projection.adapter.weight",
		"vision_model.radio_model.model.patch_generator.proj.weight",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("spec names = %v, want %v", got, want)
	}

	for _, name := range []string{
		"mlp1.0.weight",
		"sound_encoder.encoder.layers.0.self_attn.q_proj.weight",
		"sound_projection.adapter.weight",
		"vision_model.radio_model.model.patch_generator.proj.weight",
	} {
		spec, ok := specByName(specs, name)
		if !ok {
			t.Fatalf("missing spec %s", name)
		}
		if got := spec.Tensors[0].Quantize; got != "" {
			t.Fatalf("%s quantize = %q, want source precision", name, got)
		}
	}
}

func TestNemotronHImportTransformQuantizationPolicy(t *testing.T) {
	transform := nemotronHImportTransform{numLayers: 52}

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{"mamba in_proj nvfp4", "language_model.backbone.layers.0.mixer.in_proj.weight", []int32{96, 32}, "nvfp4", "nvfp4"},
		{"mamba out_proj promoted", "language_model.backbone.layers.0.mixer.out_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"attention v_proj promoted", "language_model.backbone.layers.0.mixer.v_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"attention o_proj promoted", "language_model.backbone.layers.0.mixer.o_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"attention o_proj mxfp8", "language_model.backbone.layers.1.mixer.o_proj.weight", []int32{32, 32}, "mxfp8", "mxfp8"},
		{"expert up_proj nvfp4", "language_model.backbone.layers.2.mixer.experts.1.up_proj.weight", []int32{64, 32}, "nvfp4", "nvfp4"},
		{"expert down_proj promoted", "language_model.backbone.layers.2.mixer.experts.1.down_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"shared expert down_proj promoted", "language_model.backbone.layers.2.mixer.shared_experts.down_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"shared expert down_proj mxfp8", "language_model.backbone.layers.2.mixer.shared_experts.down_proj.weight", []int32{32, 64}, "mxfp8", "mxfp8"},
		{"late dense down_proj promoted", "language_model.backbone.layers.51.mixer.down_proj.weight", []int32{32, 32}, "nvfp4", "mxfp8"},
		{"middle dense down_proj stays nvfp4", "language_model.backbone.layers.16.mixer.down_proj.weight", []int32{32, 32}, "nvfp4", "nvfp4"},
		{"router gate kept bf16", "language_model.backbone.layers.2.mixer.gate.weight", []int32{8, 32}, "nvfp4", ""},
		{"conv kept bf16", "language_model.backbone.layers.0.mixer.conv1d.weight", []int32{96, 32}, "nvfp4", ""},
		{"embedding kept bf16", "language_model.backbone.embeddings.weight", []int32{4096, 32}, "mxfp8", ""},
		{"lm head kept bf16", "language_model.lm_head.weight", []int32{4096, 32}, "nvfp4", ""},
		{"vision kept bf16", "vision_model.radio_model.model.patch_generator.proj.weight", []int32{32, 32}, "nvfp4", ""},
		{"vision projector kept bf16", "mlp1.0.weight", []int32{32, 32}, "nvfp4", ""},
		{"sound encoder kept bf16", "sound_encoder.encoder.layers.0.self_attn.q_proj.weight", []int32{32, 32}, "nvfp4", ""},
		{"sound projector kept bf16", "sound_projection.adapter.weight", []int32{32, 32}, "nvfp4", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := transform.quantizationType(tt.tensor, tt.shape, tt.quantize); got != tt.want {
				t.Fatalf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}
