package create

import "testing"

func TestRemapDiffusionGemmaName(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// Decoder stack -> canonical model.*
		{"model.decoder.embed_tokens.weight", "model.embed_tokens.weight"},
		{"model.decoder.embed_tokens.scales", "model.embed_tokens.scales"},
		{"model.decoder.norm.weight", "model.norm.weight"},
		{"model.decoder.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"model.decoder.layers.0.self_attn.q_proj.biases", "model.layers.0.self_attn.q_proj.biases"},
		{"model.decoder.layers.3.layer_scalar", "model.layers.3.layer_scalar"},
		{"model.decoder.layers.5.experts.gate_up_proj.weight", "model.layers.5.experts.gate_up_proj.weight"},
		// Self-conditioning MLP -> self_cond_*, preserving the quant companion suffix.
		{"model.decoder.self_conditioning.pre_norm.weight", "self_cond_pre_norm.weight"},
		{"model.decoder.self_conditioning.gate_proj.weight", "self_cond_gate.weight"},
		{"model.decoder.self_conditioning.gate_proj.scales", "self_cond_gate.scales"},
		{"model.decoder.self_conditioning.gate_proj.biases", "self_cond_gate.biases"},
		{"model.decoder.self_conditioning.up_proj.scales", "self_cond_up.scales"},
		{"model.decoder.self_conditioning.down_proj.biases", "self_cond_down.biases"},
		// Already-canonical names pass through unchanged.
		{"model.norm.weight", "model.norm.weight"},
	}
	for _, c := range cases {
		if got := remapDiffusionGemmaName(c.in); got != c.want {
			t.Errorf("remapDiffusionGemmaName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestDiffusionGemmaSkipTensor(t *testing.T) {
	tr := diffusionGemmaImportTransform{gemma4ImportTransform{numLayers: 30, numExperts: 128}}
	cases := []struct {
		name string
		want bool
	}{
		// Entire encoder stack (vision tower + encoder per-layer scale) is dropped.
		{"model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", true},
		{"model.encoder.embed_vision.embedding_projection.weight", true},
		{"model.encoder.language_model.layers.0.layer_scalar", true},
		// Decoder stack and self-conditioning are kept.
		{"model.decoder.layers.0.self_attn.q_proj.weight", false},
		{"model.decoder.layers.0.layer_scalar", false},
		{"model.decoder.self_conditioning.gate_proj.scales", false},
		{"model.decoder.embed_tokens.weight", false},
	}
	for _, c := range cases {
		if got := tr.skipTensor(c.name); got != c.want {
			t.Errorf("skipTensor(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}
