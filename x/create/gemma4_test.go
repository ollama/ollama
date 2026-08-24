package create

import (
	"encoding/json"
	"testing"
)

func TestGemma4UnifiedImportTransformRegistration(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		cfg        sourceModelConfig
		wantErr    bool
		wantLayers int
	}{
		{
			name:       "unified conditional generation architecture",
			configJSON: `{"architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"num_hidden_layers":48}}`,
			cfg:        sourceModelConfig{Architectures: []string{"Gemma4UnifiedForConditionalGeneration"}},
			wantLayers: 48,
		},
		{
			name:       "unified model type fallback",
			configJSON: `{"model_type":"gemma4_unified","text_config":{"num_hidden_layers":48}}`,
			cfg:        sourceModelConfig{ModelType: "gemma4_unified"},
			wantLayers: 48,
		},
		{
			name:       "malformed config is a hard error",
			configJSON: `{"architectures":["Gemma4UnifiedForConditionalGeneration"],"num_hidden_layers":"oops"}`,
			cfg:        sourceModelConfig{Architectures: []string{"Gemma4UnifiedForConditionalGeneration"}},
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inv := Inventory{Config: tt.cfg, RawConfig: json.RawMessage(tt.configJSON)}

			transform, err := newTensorImportTransform(inv)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("newTensorImportTransform() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("newTensorImportTransform() error = %v", err)
			}

			gemmaTransform, ok := transform.(gemma4ImportTransform)
			if !ok {
				t.Fatalf("newTensorImportTransform() = %T, want gemma4ImportTransform", transform)
			}
			if gemmaTransform.numLayers != tt.wantLayers {
				t.Fatalf("numLayers = %d, want %d", gemmaTransform.numLayers, tt.wantLayers)
			}
		})
	}
}

func TestGemma4QuantizationType(t *testing.T) {
	// 26B MoE: 30 layers, 128 experts
	transform26B := gemma4ImportTransform{numLayers: 30, numExperts: 128}
	// 8-expert model (hypothetical)
	transform8E := gemma4ImportTransform{numLayers: 30, numExperts: 8}
	// Dense model (12B/31B): no experts, so the layer-position heuristic applies
	transformDense := gemma4ImportTransform{numLayers: 30}

	aligned := []int32{2816, 2816} // divisible by 64 (int4/int8 group size) and 16 (nvfp4)
	// 26B-A4B source shapes, which are aligned for every quant family.
	expertDown := []int32{128, 2816, 704}
	expertGateUp := []int32{128, 1408, 2816}

	tests := []struct {
		name      string
		transform gemma4ImportTransform
		tensor    string
		shape     []int32
		quantize  string
		want      string
	}{
		// === embed_tokens: quantize to 8-bit variant (serves as both embed and lm_head) ===
		{"embed_tokens int4", transform26B, "model.embed_tokens.weight", aligned, "int4", "int8"},
		{"embed_tokens nvfp4", transform26B, "model.embed_tokens.weight", aligned, "nvfp4", "mxfp8"},
		{"embed_tokens mxfp4", transform26B, "model.embed_tokens.weight", aligned, "mxfp4", "mxfp8"},
		{"embed_tokens int8", transform26B, "model.embed_tokens.weight", aligned, "int8", "int8"},
		{"embed_tokens mxfp8", transform26B, "model.embed_tokens.weight", aligned, "mxfp8", "mxfp8"},

		// === Sparse MoE dense body: the generic policy decides ===
		// The architecture override steps aside, so v/k/down are promoted at
		// every layer rather than at useMoreBits layers, and q/o/gate/up stay
		// at the requested type.
		{"body v_proj int4 first layer", transform26B, "model.language_model.layers.0.self_attn.v_proj.weight", aligned, "int4", "int8"},
		{"body v_proj int4 middle layer", transform26B, "model.language_model.layers.4.self_attn.v_proj.weight", aligned, "int4", "int8"},
		{"body v_proj int4 last layer", transform26B, "model.language_model.layers.29.self_attn.v_proj.weight", aligned, "int4", "int8"},
		// nvfp4: promote to mxfp8 (cross-family, validated by MLX quantized_matmul)
		{"body v_proj nvfp4 middle layer", transform26B, "model.language_model.layers.4.self_attn.v_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"body k_proj nvfp4", transform26B, "model.language_model.layers.4.self_attn.k_proj.weight", []int32{2048, 2816}, "nvfp4", "mxfp8"},
		{"body mlp down_proj nvfp4", transform26B, "model.language_model.layers.4.mlp.down_proj.weight", []int32{2816, 2112}, "nvfp4", "mxfp8"},
		{"body q_proj nvfp4", transform26B, "model.language_model.layers.4.self_attn.q_proj.weight", []int32{4096, 2816}, "nvfp4", "nvfp4"},
		{"body o_proj nvfp4", transform26B, "model.language_model.layers.4.self_attn.o_proj.weight", []int32{2816, 4096}, "nvfp4", "nvfp4"},
		{"body mlp gate_proj nvfp4", transform26B, "model.language_model.layers.4.mlp.gate_proj.weight", []int32{2112, 2816}, "nvfp4", "nvfp4"},
		{"body mlp up_proj nvfp4", transform26B, "model.language_model.layers.4.mlp.up_proj.weight", []int32{2112, 2816}, "nvfp4", "nvfp4"},
		// mxfp4: promoted to mxfp8 (same mxfp family)
		{"body v_proj mxfp4", transform26B, "model.language_model.layers.4.self_attn.v_proj.weight", aligned, "mxfp4", "mxfp8"},
		// int8/mxfp8: no promotion (already 8-bit)
		{"body v_proj int8 base", transform26B, "model.language_model.layers.0.self_attn.v_proj.weight", aligned, "int8", "int8"},
		{"body v_proj mxfp8 base", transform26B, "model.language_model.layers.0.self_attn.v_proj.weight", aligned, "mxfp8", "mxfp8"},

		// === Sparse MoE routed experts: one format for the whole bank ===
		// Gemma 4 stacks the experts of a layer into one tensor, so these names
		// carry no expert index and "down_proj" matches the bank itself.
		// Promoting part of it splits the bank across two formats, which
		// measured far worse than leaving all of it at the requested type.
		{"stacked expert down int4", transform26B, "model.language_model.layers.0.experts.down_proj", expertDown, "int4", "int4"},
		{"stacked expert down nvfp4", transform26B, "model.language_model.layers.0.experts.down_proj", expertDown, "nvfp4", "nvfp4"},
		{"stacked expert down mxfp4", transform26B, "model.language_model.layers.0.experts.down_proj", expertDown, "mxfp4", "mxfp4"},
		{"stacked expert down mxfp8", transform26B, "model.language_model.layers.0.experts.down_proj", expertDown, "mxfp8", "mxfp8"},
		{"stacked expert gate_up int4", transform26B, "model.language_model.layers.0.experts.gate_up_proj", expertGateUp, "int4", "int4"},
		{"stacked expert gate_up nvfp4", transform26B, "model.language_model.layers.0.experts.gate_up_proj", expertGateUp, "nvfp4", "nvfp4"},
		{"stacked expert gate_up mxfp4", transform26B, "model.language_model.layers.0.experts.gate_up_proj", expertGateUp, "mxfp4", "mxfp4"},
		// Per-expert naming reaches the same decision.
		{"per-expert down int4", transform26B, "model.layers.0.moe.experts.42.down_proj.weight", aligned, "int4", "int4"},
		{"per-expert down nvfp4", transform26B, "model.layers.0.moe.experts.42.down_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"per-expert gate_up nvfp4", transform26B, "model.layers.0.moe.experts.42.gate_up_proj.weight", aligned, "nvfp4", "nvfp4"},

		// === Dense model: v/down promoted by layer position ===
		// Layer 0 is in first 1/8 (30/8=3) → promoted
		{"dense v_proj int4 promoted layer", transformDense, "model.layers.0.self_attn.v_proj.weight", aligned, "int4", "int8"},
		// Layer 4 is NOT in useMoreBits → base quant
		{"dense v_proj int4 non-promoted layer", transformDense, "model.layers.4.self_attn.v_proj.weight", aligned, "int4", "int4"},
		// Layer 29 is in last 1/8 → promoted
		{"dense v_proj int4 last layer promoted", transformDense, "model.layers.29.self_attn.v_proj.weight", aligned, "int4", "int8"},
		{"dense v_proj nvfp4 promoted layer", transformDense, "model.layers.0.self_attn.v_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"dense v_proj nvfp4 non-promoted layer", transformDense, "model.layers.4.self_attn.v_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"dense v_proj mxfp4 promoted layer", transformDense, "model.layers.0.self_attn.v_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"dense v_proj mxfp4 non-promoted layer", transformDense, "model.layers.4.self_attn.v_proj.weight", aligned, "mxfp4", "mxfp4"},
		{"dense down_proj int4 promoted", transformDense, "model.layers.0.mlp.down_proj.weight", aligned, "int4", "int8"},
		{"dense down_proj int4 non-promoted", transformDense, "model.layers.4.mlp.down_proj.weight", aligned, "int4", "int4"},
		{"dense down_proj nvfp4 promoted", transformDense, "model.layers.0.mlp.down_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"dense down_proj nvfp4 non-promoted", transformDense, "model.layers.4.mlp.down_proj.weight", aligned, "nvfp4", "nvfp4"},
		// A dense model's q/o/gate/up are not sensitive: base quant everywhere.
		{"dense q_proj nvfp4", transformDense, "model.layers.0.self_attn.q_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"dense o_proj nvfp4", transformDense, "model.layers.0.self_attn.o_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"dense gate_proj nvfp4", transformDense, "model.layers.0.mlp.gate_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"dense up_proj nvfp4", transformDense, "model.layers.0.mlp.up_proj.weight", aligned, "nvfp4", "nvfp4"},

		// === Router projection: expert selection is sensitive; keep source precision ===
		{"router proj int4", transform26B, "model.layers.0.router.proj.weight", aligned, "int4", ""},
		{"router proj nvfp4", transform26B, "model.layers.0.router.proj.weight", aligned, "nvfp4", ""},
		{"router proj mxfp4", transform26B, "model.layers.0.router.proj.weight", aligned, "mxfp4", ""},

		// === k_proj: 8-expert models promote it through their own path ===
		{"k_proj 8 experts int4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "int4", "int8"},
		{"k_proj 8 experts nvfp4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"k_proj 8 experts mxfp4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"k_proj dense non-promoted layer", transformDense, "model.layers.4.self_attn.k_proj.weight", aligned, "int4", "int4"},
		{"k_proj dense promoted layer", transformDense, "model.layers.0.self_attn.k_proj.weight", aligned, "int4", "int4"},

		// === Sparse MoE body: q/o/gate/up are not sensitive under int4 either ===
		{"body q_proj int4", transform26B, "model.layers.0.self_attn.q_proj.weight", aligned, "int4", "int4"},
		{"body o_proj int4", transform26B, "model.layers.0.self_attn.o_proj.weight", aligned, "int4", "int4"},
		{"body gate_proj int4", transform26B, "model.layers.0.mlp.gate_proj.weight", aligned, "int4", "int4"},
		{"body up_proj int4", transform26B, "model.layers.0.mlp.up_proj.weight", aligned, "int4", "int4"},

		// === Non-quantizable tensors: always bf16 ===
		{"embed_tokens per_layer skip", transform26B, "model.embed_tokens_per_layer.weight", aligned, "int4", ""},
		{"norm", transform26B, "model.layers.0.input_layernorm.weight", []int32{2816}, "int4", ""},
		{"router scale", transform26B, "model.layers.0.router.scale", []int32{2816}, "int4", ""},

		// === Audio/vision tower tensors: must pass through unquantized for all quant types ===
		// These contain .v_proj and down_proj but should NOT be intercepted by
		// the sensitive-tensor promotion logic.
		{"audio norm int4", transform26B, "model.audio_tower.subsample_conv_projection.layer0.norm.weight", []int32{128}, "int4", ""},
		{"audio norm nvfp4", transform26B, "model.audio_tower.subsample_conv_projection.layer0.norm.weight", []int32{128}, "nvfp4", ""},
		{"audio norm int8", transform26B, "model.audio_tower.subsample_conv_projection.layer0.norm.weight", []int32{128}, "int8", ""},
		{"audio norm mxfp8", transform26B, "model.audio_tower.subsample_conv_projection.layer0.norm.weight", []int32{128}, "mxfp8", ""},
		{"audio conv int4", transform26B, "model.audio_tower.subsample_conv_projection.layer0.conv.weight", []int32{128, 1, 3, 3}, "int4", ""},
		{"audio conv nvfp4", transform26B, "model.audio_tower.subsample_conv_projection.layer0.conv.weight", []int32{128, 1, 3, 3}, "nvfp4", ""},
		{"audio linear int4", transform26B, "model.audio_tower.subsample_conv_projection.input_proj_linear.weight", aligned, "int4", ""},
		{"audio linear nvfp4", transform26B, "model.audio_tower.subsample_conv_projection.input_proj_linear.weight", aligned, "nvfp4", ""},
		// Audio tower v_proj — must NOT be promoted despite containing .v_proj
		{"audio v_proj int4", transform26B, "model.audio_tower.layers.0.self_attn.v_proj.linear.weight", aligned, "int4", ""},
		{"audio v_proj nvfp4", transform26B, "model.audio_tower.layers.0.self_attn.v_proj.linear.weight", aligned, "nvfp4", ""},
		// Vision tower — source precision for every quant family.
		{"vision v_proj int4", transform26B, "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight", aligned, "int4", ""},
		{"vision v_proj nvfp4", transform26B, "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight", aligned, "nvfp4", ""},
		{"vision q_proj nvfp4", transform26B, "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", aligned, "nvfp4", ""},
		{"unified vision embedder nvfp4", transform26B, "model.vision_embedder.patch_dense.weight", aligned, "nvfp4", ""},
		{"vision projection nvfp4", transform26B, "model.embed_vision.embedding_projection.linear.weight", aligned, "nvfp4", ""},
		// Audio tower down_proj
		{"audio down_proj int4", transform26B, "model.audio_tower.layers.0.mlp.down_proj.linear.weight", aligned, "int4", ""},
		{"audio down_proj nvfp4", transform26B, "model.audio_tower.layers.0.mlp.down_proj.linear.weight", aligned, "nvfp4", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.transform.quantizationType(tt.tensor, tt.shape, tt.quantize)
			if got != tt.want {
				t.Errorf("quantizationType(%q, %v, %q) = %q, want %q",
					tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

func TestUseMoreBits(t *testing.T) {
	// 30 layers: first 1/8 = layers 0-2, last 1/8 = layers 27-29
	// In between: every 3rd from offset (i - n/8) % 3 == 2
	n := 30
	promoted := map[int]bool{}
	for i := range n {
		if useMoreBits(i, n) {
			promoted[i] = true
		}
	}

	// First 1/8 (30/8 = 3): layers 0, 1, 2
	for _, i := range []int{0, 1, 2} {
		if !promoted[i] {
			t.Errorf("layer %d should be promoted (first 1/8)", i)
		}
	}

	// Last 1/8: layers 26, 27, 28, 29 (>= 7*30/8 = 26)
	for _, i := range []int{26, 27, 28, 29} {
		if !promoted[i] {
			t.Errorf("layer %d should be promoted (last 1/8)", i)
		}
	}

	// Some middle layers should NOT be promoted
	for _, i := range []int{3, 4, 6, 7} {
		if promoted[i] {
			t.Errorf("layer %d should NOT be promoted", i)
		}
	}

	// Layer 5 should be promoted: (5 - 3) % 3 == 2
	if !promoted[5] {
		t.Errorf("layer 5 should be promoted (periodic)")
	}
}
