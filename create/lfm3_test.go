package create

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestLFM3VLImportKeepsVisionAndProjector(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"Lfm3VlForConditionalGeneration"}}, map[string]string{})
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	shape := []int32{2048, 2048}
	for _, name := range []string{
		"model.multi_modal_projector.linear_1.weight",
		"model.multi_modal_projector.linear_2.weight",
		"model.vision_tower.layers.0.self_attn.q_proj.weight",
	} {
		if got := policy.quantizationType(name, shape, "mxfp8"); got != "" {
			t.Errorf("quantizationType(%q) = %q, want source precision", name, got)
		}
	}
	if got := policy.quantizationType("model.language_model.layers.0.feed_forward.w1.weight", shape, "mxfp8"); got != "mxfp8" {
		t.Errorf("language linear quantization = %q, want mxfp8", got)
	}
	var cfg model.ConfigV2
	if err := policy.(interface{ augmentConfig(*model.ConfigV2) error }).augmentConfig(&cfg); err != nil || cfg.Voices != nil {
		t.Errorf("VL voices %v (%v), want none", cfg.Voices, err)
	}
}

func TestLFM3SpeechImportQuantizesOnlyBackbone(t *testing.T) {
	shape := []int32{2048, 2048}
	for arch, cases := range map[string]map[string]bool{
		"Lfm3OmniForConditionalGeneration": {
			"model.language_model.layers.0.feed_forward.w1.weight":       true,
			"model.audio_encoder.lfm.layers.0.feed_forward.w1.weight":    false,
			"model.audio_projector.linear_1.weight":                      false,
			"model.depth_decoder.model.layers.0.self_attn.q_proj.weight": false,
			"model.multi_modal_projector.linear_1.weight":                false,
		},
		"Lfm3AudioForConditionalGeneration": {
			"lfm.model.layers.0.feed_forward.w1.weight":          true,
			"encoder.lfm.lfm.layers.0.feed_forward.w1.weight":    false,
			"mlp.model.1.weight":                                 false,
			"depthformer.model.layers.0.self_attn.q_proj.weight": false,
		},
	} {
		inv := newInventory(sourceModelConfig{Architectures: []string{arch}}, map[string]string{})
		inv.RawConfig = []byte(`{"lfm":{"hidden_size":2048,"max_position_embeddings":128000}}`)
		policy, err := newTensorImportTransform(inv)
		if err != nil {
			t.Fatal(err)
		}
		var cfg model.ConfigV2
		if err := policy.(interface{ augmentConfig(*model.ConfigV2) error }).augmentConfig(&cfg); err != nil {
			t.Fatal(err)
		}
		if !slices.Contains(cfg.Voices, "female-japanese") {
			t.Errorf("%s voices %v, want the LFM3 speech voices", arch, cfg.Voices)
		}
		for name, want := range cases {
			if got := policy.quantizationType(name, shape, "mxfp8") != ""; got != want {
				t.Errorf("%s %s quantized = %v, want %v", arch, name, got, want)
			}
		}
	}
}

func TestD1ImportQuantizesOnlyBackbone(t *testing.T) {
	shape := []int32{2048, 2048}
	policy, err := newTensorImportTransform(newInventory(sourceModelConfig{Architectures: []string{"D1Model"}}, map[string]string{}))
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]bool{
		"model.language_model.layers.0.feed_forward.w1.weight":       true,
		"model.vision_tower.layers.0.mlp.fc1.weight":                 false,
		"model.audio_encoder.lfm.layers.0.feed_forward.w1.weight":    false,
		"model.depth_decoder.model.layers.0.self_attn.q_proj.weight": false,
		"model.multi_modal_projector.linear_1.weight":                false,
	} {
		if got := policy.quantizationType(name, shape, "mxfp8") != ""; got != want {
			t.Errorf("%s quantized = %v, want %v", name, got, want)
		}
	}
}
