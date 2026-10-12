package create

import (
	"encoding/json"
	"testing"
)

func TestGlimmerImportTransformPreservesMultimodalTensors(t *testing.T) {
	transform := glimmerImportTransform{numLayers: 52}
	shape := []int32{256, 256}

	for _, name := range []string{
		"model.vision_tower.layers.0.attn.q_proj.weight",
		"model.vision_adapter.fc1.weight",
		"model.vision_projection.weight",
	} {
		if got := transform.quantizationType(name, shape, "int4"); got != "" {
			t.Errorf("quantizationType(%q) = %q, want source precision", name, got)
		}
	}

	if got := transform.quantizationType("model.language_model.layers.10.self_attn.q_proj.weight", shape, "int4"); got == "" {
		t.Fatal("text decoder projection unexpectedly kept at source precision")
	}
}

func TestGlimmerPlanKeepsMultimodalTensors(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"MuseGlimmerForConditionalGeneration"}}, map[string]string{
		"model.language_model.layers.0.self_attn.q_proj.weight": "BF16",
		"model.vision_tower.layers.0.attn.q_proj.weight":        "BF16",
		"model.vision_adapter.fc1.weight":                       "BF16",
		"model.vision_projection.weight":                        "BF16",
	})
	inv.RawConfig = json.RawMessage(`{"text_config":{"num_hidden_layers":52}}`)

	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "nvfp4"}, policy)
	if err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{
		"model.vision_tower.layers.0.attn.q_proj.weight",
		"model.vision_adapter.fc1.weight",
		"model.vision_projection.weight",
	} {
		spec, ok := specByName(specs, name)
		if !ok {
			t.Fatalf("missing multimodal tensor %q in plan", name)
		}
		if len(spec.Tensors) != 1 || spec.Tensors[0].Quantize != "" {
			t.Fatalf("planned tensor %q = %+v, want source precision", name, spec.Tensors)
		}
	}

	text, ok := specByName(specs, "model.language_model.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatal("missing text decoder tensor in plan")
	}
	if len(text.Tensors) != 1 || text.Tensors[0].Quantize == "" {
		t.Fatalf("text decoder tensor = %+v, want quantized", text.Tensors)
	}
}

func TestGlimmerQuantizationType(t *testing.T) {
	transform := glimmerImportTransform{numLayers: 52}
	large := []int32{6656, 6656}
	lmHead := []int32{202048, 6656}
	ffnDown := []int32{6656, 19968}

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{"embed_tokens nvfp4 promotes", "model.language_model.embed_tokens.weight", lmHead, "nvfp4", "mxfp8"},
		{"embed_tokens mxfp8 stays", "model.language_model.embed_tokens.weight", lmHead, "mxfp8", "mxfp8"},
		{"lm_head nvfp4 promotes", "lm_head.weight", lmHead, "nvfp4", "mxfp8"},
		{"lm_head mxfp8 stays", "lm_head.weight", lmHead, "mxfp8", "mxfp8"},
		{"lm_head int4 quantizes", "lm_head.weight", lmHead, "int4", "int8"},

		{"q_proj nvfp4 promotes", "model.language_model.layers.8.self_attn.q_proj.weight", large, "nvfp4", "mxfp8"},
		{"q_proj nvfp4 always promotes", "model.language_model.layers.6.self_attn.q_proj.weight", large, "nvfp4", "mxfp8"},
		{"o_proj nvfp4 promotes", "model.language_model.layers.8.self_attn.o_proj.weight", large, "nvfp4", "mxfp8"},
		{"o_proj nvfp4 always promotes", "model.language_model.layers.6.self_attn.o_proj.weight", large, "nvfp4", "mxfp8"},
		{"k_proj nvfp4 promotes", "model.language_model.layers.8.self_attn.k_proj.weight", []int32{256, 6656}, "nvfp4", "mxfp8"},
		{"v_proj nvfp4 promotes", "model.language_model.layers.8.self_attn.v_proj.weight", []int32{256, 6656}, "nvfp4", "mxfp8"},

		{"down_proj nvfp4 first layer promotes", "model.language_model.layers.0.mlp.down_proj.weight", ffnDown, "nvfp4", "mxfp8"},
		{"down_proj nvfp4 non-promoted layer", "model.language_model.layers.6.mlp.down_proj.weight", ffnDown, "nvfp4", "nvfp4"},
		{"down_proj nvfp4 periodic layer promotes", "model.language_model.layers.8.mlp.down_proj.weight", ffnDown, "nvfp4", "mxfp8"},
		{"down_proj mxfp8 stays", "model.language_model.layers.6.mlp.down_proj.weight", ffnDown, "mxfp8", "mxfp8"},
		{"output_gate nvfp4 promoted layer", "model.language_model.layers.0.self_attn.gate_proj.weight", large, "nvfp4", "mxfp8"},
		{"output_gate nvfp4 non-promoted layer", "model.language_model.layers.6.self_attn.gate_proj.weight", large, "nvfp4", "nvfp4"},

		{"vision projection preserved", "model.vision_projection.weight", large, "nvfp4", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := transform.quantizationType(tt.tensor, tt.shape, tt.quantize)
			if got != tt.want {
				t.Fatalf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

func TestGlimmerImportTransformRegistered(t *testing.T) {
	transform, err := newTensorImportTransform(Inventory{
		Config:    sourceModelConfig{Architectures: []string{"MuseGlimmerForConditionalGeneration"}},
		RawConfig: json.RawMessage(`{"text_config":{"num_hidden_layers":52}}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := transform.(glimmerImportTransform); !ok {
		t.Fatalf("newTensorImportTransform() = %T, want glimmerImportTransform", transform)
	}
}
