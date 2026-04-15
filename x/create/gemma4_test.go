package create

import (
	"testing"
)

func TestGemma4QuantizationType(t *testing.T) {
	// 26B MoE: 30 layers, 128 experts
	transform26B := gemma4ImportTransform{numLayers: 30, numExperts: 128}
	// 8-expert model (hypothetical)
	transform8E := gemma4ImportTransform{numLayers: 30, numExperts: 8}

	aligned := []int32{2816, 2816} // divisible by 64 (int4/int8 group size) and 16 (nvfp4)

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

		// === v_proj: layer-position heuristic for int4/nvfp4 ===
		// Layer 0 is in first 1/8 (30/8=3) → promoted
		{"v_proj int4 promoted layer", transform26B, "model.layers.0.self_attn.v_proj.weight", aligned, "int4", "int8"},
		// Layer 4 is NOT in useMoreBits → base quant
		{"v_proj int4 non-promoted layer", transform26B, "model.layers.4.self_attn.v_proj.weight", aligned, "int4", "int4"},
		// Layer 29 is in last 1/8 → promoted
		{"v_proj int4 last layer promoted", transform26B, "model.layers.29.self_attn.v_proj.weight", aligned, "int4", "int8"},
		// nvfp4: promote to mxfp8 (cross-family, validated by MLX quantized_matmul)
		{"v_proj nvfp4 promoted layer", transform26B, "model.layers.0.self_attn.v_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"v_proj nvfp4 non-promoted layer", transform26B, "model.layers.4.self_attn.v_proj.weight", aligned, "nvfp4", "nvfp4"},
		// mxfp4: promoted to mxfp8 at promoted layers (same mxfp family)
		{"v_proj mxfp4 promoted layer", transform26B, "model.layers.0.self_attn.v_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"v_proj mxfp4 non-promoted layer", transform26B, "model.layers.4.self_attn.v_proj.weight", aligned, "mxfp4", "mxfp4"},
		// int8/mxfp8: no promotion (already 8-bit)
		{"v_proj int8 base", transform26B, "model.layers.0.self_attn.v_proj.weight", aligned, "int8", "int8"},
		{"v_proj mxfp8 base", transform26B, "model.layers.0.self_attn.v_proj.weight", aligned, "mxfp8", "mxfp8"},

		// === down_proj (dense MLP): same heuristic as v_proj ===
		{"dense down_proj int4 promoted", transform26B, "model.layers.0.mlp.down_proj.weight", aligned, "int4", "int8"},
		{"dense down_proj int4 non-promoted", transform26B, "model.layers.4.mlp.down_proj.weight", aligned, "int4", "int4"},
		{"dense down_proj nvfp4 promoted", transform26B, "model.layers.0.mlp.down_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"dense down_proj nvfp4 non-promoted", transform26B, "model.layers.4.mlp.down_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"dense down_proj mxfp4 promoted", transform26B, "model.layers.0.mlp.down_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"dense down_proj mxfp4 non-promoted", transform26B, "model.layers.4.mlp.down_proj.weight", aligned, "mxfp4", "mxfp4"},

		// === Expert down_proj: int4→int8, nvfp4→nvfp8 at promoted layers ===
		{"expert down_proj int4 promoted", transform26B, "model.layers.0.moe.experts.42.down_proj.weight", aligned, "int4", "int8"},
		{"expert down_proj int4 non-promoted", transform26B, "model.layers.4.moe.experts.42.down_proj.weight", aligned, "int4", "int4"},
		// nvfp4 experts: promote to mxfp8 (all experts at a layer get same treatment,
		// so GatherQMM sees uniform quant per projection per layer)
		{"expert down_proj nvfp4 promoted layer", transform26B, "model.layers.0.moe.experts.42.down_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"expert down_proj nvfp4 non-promoted layer", transform26B, "model.layers.4.moe.experts.42.down_proj.weight", aligned, "nvfp4", "nvfp4"},
		// mxfp4 experts: promote to mxfp8 (same mxfp family, GatherQMM compatible)
		{"expert down_proj mxfp4 promoted layer", transform26B, "model.layers.0.moe.experts.42.down_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"expert down_proj mxfp4 non-promoted layer", transform26B, "model.layers.4.moe.experts.42.down_proj.weight", aligned, "mxfp4", "mxfp4"},

		// === Expert gate_up_proj: always base quant (not a sensitive tensor) ===
		{"expert gate_up int4", transform26B, "model.layers.0.moe.experts.42.gate_up_proj.weight", aligned, "int4", "int4"},
		{"expert gate_up nvfp4", transform26B, "model.layers.0.moe.experts.42.gate_up_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"expert gate_up mxfp4", transform26B, "model.layers.0.moe.experts.42.gate_up_proj.weight", aligned, "mxfp4", "mxfp4"},

		// === Router projection: expert selection is sensitive; keep source precision ===
		{"router proj int4", transform26B, "model.layers.0.router.proj.weight", aligned, "int4", ""},
		{"router proj nvfp4", transform26B, "model.layers.0.router.proj.weight", aligned, "nvfp4", ""},
		{"router proj mxfp4", transform26B, "model.layers.0.router.proj.weight", aligned, "mxfp4", ""},

		// === k_proj: promoted only for 8-expert models ===
		{"k_proj 128 experts int4", transform26B, "model.layers.0.self_attn.k_proj.weight", aligned, "int4", "int4"},
		{"k_proj 8 experts int4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "int4", "int8"},
		{"k_proj 8 experts nvfp4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"k_proj 8 experts mxfp4", transform8E, "model.layers.0.self_attn.k_proj.weight", aligned, "mxfp4", "mxfp8"},

		// === q_proj, o_proj, gate_proj, up_proj: always base quant ===
		{"q_proj int4", transform26B, "model.layers.0.self_attn.q_proj.weight", aligned, "int4", "int4"},
		{"o_proj int4", transform26B, "model.layers.0.self_attn.o_proj.weight", aligned, "int4", "int4"},
		{"gate_proj int4", transform26B, "model.layers.0.mlp.gate_proj.weight", aligned, "int4", "int4"},
		{"up_proj int4", transform26B, "model.layers.0.mlp.up_proj.weight", aligned, "int4", "int4"},

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
		// Vision tower v_proj — vision tower IS quantized (unlike audio tower),
		// but not intercepted by gemma4's layer-position heuristic.
		// Falls through to GetTensorQuantization which applies uniform promotion.
		{"vision v_proj int4", transform26B, "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight", aligned, "int4", "int8"},
		{"vision v_proj nvfp4", transform26B, "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight", aligned, "nvfp4", "nvfp4"},
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

func TestIsGemma4StackedMoETensor(t *testing.T) {
	tests := []struct {
		label      string
		tensorName string
		shape      []int32
		want       bool
	}{
		// New-style: .experts.gate_up_proj
		{"experts gate_up_proj 3D", "model.layers.0.experts.gate_up_proj", []int32{128, 1408, 2816}, true},
		{"experts down_proj 3D", "model.layers.0.experts.down_proj", []int32{128, 2816, 704}, true},
		// Old-style: .moe.gate_proj
		{"moe gate_proj 3D", "model.layers.0.moe.gate_proj", []int32{128, 2112, 2816}, true},
		{"moe down_proj 3D", "model.layers.0.moe.down_proj.weight", []int32{128, 2816, 2112}, true},
		// Not stacked: 2D
		{"2D weight", "model.layers.0.experts.gate_up_proj", []int32{1408, 2816}, false},
		// Not expert
		{"non-expert 3D", "model.layers.0.mlp.gate_proj", []int32{3, 2816, 2816}, false},
		// Not a projection
		{"expert non-proj", "model.layers.0.experts.scale", []int32{128, 1, 1}, false},
	}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			got := isGemma4StackedMoETensor(tt.tensorName, tt.shape)
			if got != tt.want {
				t.Errorf("isGemma4StackedMoETensor(%q, %v) = %v, want %v",
					tt.tensorName, tt.shape, got, tt.want)
			}
		})
	}
}
