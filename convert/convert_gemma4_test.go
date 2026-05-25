package convert

import (
	"strings"
	"testing"
)

func TestGemma4AudioReplacements(t *testing.T) {
	p := gemma4Model{}
	r := strings.NewReplacer(p.Replacements()...)

	tests := []struct {
		name string
		in   string
		want string
	}{
		// SSCP convolution blocks
		{
			"sscp conv0 weight",
			"model.audio_tower.subsample_conv_projection.conv_0.conv.weight",
			"a.conv1d.0.weight",
		},
		{
			"sscp conv0 norm",
			"model.audio_tower.subsample_conv_projection.conv_0.norm.weight",
			"a.conv1d.0.norm.weight",
		},
		{
			"sscp conv1 weight",
			"model.audio_tower.subsample_conv_projection.conv_1.conv.weight",
			"a.conv1d.1.weight",
		},
		{
			"sscp input proj weight",
			"model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
			"a.pre_encode.out.weight",
		},
		{
			"sscp input proj bias",
			"model.audio_tower.subsample_conv_projection.input_proj_linear.bias",
			"a.pre_encode.out.bias",
		},
		{
			"sscp layer0 conv weight (new naming)",
			"model.audio_tower.subsample_conv_projection.layer0.conv.weight",
			"a.conv1d.0.weight",
		},
		{
			"sscp layer1 norm weight (new naming)",
			"model.audio_tower.subsample_conv_projection.layer1.norm.weight",
			"a.conv1d.1.norm.weight",
		},

		// Conformer attention
		{
			"attn q weight",
			"model.audio_tower.conformer.0.attention.attn.q_proj.linear.weight",
			"a.blk.0.attn_q.weight",
		},
		{
			"attn k weight",
			"model.audio_tower.conformer.5.attention.attn.k_proj.linear.weight",
			"a.blk.5.attn_k.weight",
		},
		{
			"attn v clamp input_min",
			"model.audio_tower.conformer.0.attention.attn.v_proj.input_min",
			"a.blk.0.attn_v.input_min",
		},
		{
			"attn out weight (ClippableLinear)",
			"model.audio_tower.conformer.0.attention.post.linear.weight",
			"a.blk.0.attn_out.weight",
		},
		{
			"attn out clamp output_max",
			"model.audio_tower.conformer.0.attention.post.output_max",
			"a.blk.0.attn_out.output_max",
		},
		{
			"attn pre norm",
			"model.audio_tower.conformer.0.attention.pre_attn_norm.weight",
			"a.blk.0.ln1.weight",
		},
		{
			"attn post norm",
			"model.audio_tower.conformer.0.attention.post_norm.weight",
			"a.blk.0.ln2.weight",
		},
		{
			"linear pos",
			"model.audio_tower.conformer.0.attention.attn.relative_position_embedding.pos_proj.weight",
			"a.blk.0.linear_pos.weight",
		},
		{
			"per dim scale",
			"model.audio_tower.conformer.0.attention.attn.per_dim_scale",
			"a.blk.0.per_dim_scale",
		},
		{
			"per dim key scale",
			"model.audio_tower.conformer.0.attention.attn.per_dim_key_scale",
			"a.blk.0.per_dim_k_scale",
		},
		{
			"attn relative k proj (new naming)",
			"model.audio_tower.layers.0.self_attn.relative_k_proj.weight",
			"a.blk.0.linear_pos.weight",
		},
		{
			"attn pre norm (new naming)",
			"model.audio_tower.layers.0.norm_pre_attn.weight",
			"a.blk.0.ln1.weight",
		},
		{
			"attn post norm (new naming)",
			"model.audio_tower.layers.0.norm_post_attn.weight",
			"a.blk.0.ln2.weight",
		},
		{
			"attn out clamp output_max (new naming)",
			"model.audio_tower.layers.0.self_attn.post.output_max",
			"a.blk.0.attn_out.output_max",
		},
		{
			"per dim scale (new naming)",
			"model.audio_tower.layers.0.self_attn.per_dim_scale",
			"a.blk.0.per_dim_scale",
		},

		// Conformer feedforward start
		{
			"ffn up weight",
			"model.audio_tower.conformer.0.ffw_layer_start.ffw_layer_1.linear.weight",
			"a.blk.0.ffn_up.weight",
		},
		{
			"ffn down weight",
			"model.audio_tower.conformer.0.ffw_layer_start.ffw_layer_2.linear.weight",
			"a.blk.0.ffn_down.weight",
		},
		{
			"ffn norm",
			"model.audio_tower.conformer.0.ffw_layer_start.pre_layer_norm.weight",
			"a.blk.0.ffn_norm.weight",
		},
		{
			"ffn post norm",
			"model.audio_tower.conformer.0.ffw_layer_start.post_layer_norm.weight",
			"a.blk.0.ffn_post_norm.weight",
		},

		// Conformer feedforward end
		{
			"ffn up 1 weight",
			"model.audio_tower.conformer.0.ffw_layer_end.ffw_layer_1.linear.weight",
			"a.blk.0.ffn_up_1.weight",
		},
		{
			"ffn down 1 weight",
			"model.audio_tower.conformer.0.ffw_layer_end.ffw_layer_2.linear.weight",
			"a.blk.0.ffn_down_1.weight",
		},
		{
			"ffn norm 1",
			"model.audio_tower.conformer.0.ffw_layer_end.pre_layer_norm.weight",
			"a.blk.0.ffn_norm_1.weight",
		},
		{
			"ffn post norm 1",
			"model.audio_tower.conformer.0.ffw_layer_end.post_layer_norm.weight",
			"a.blk.0.ffn_post_norm_1.weight",
		},
		{
			"ffn up output_max (new naming)",
			"model.audio_tower.layers.10.feed_forward1.ffw_layer_1.output_max",
			"a.blk.10.ffn_up.output_max",
		},
		{
			"ffn down output_min (new naming)",
			"model.audio_tower.layers.0.feed_forward1.ffw_layer_2.output_min",
			"a.blk.0.ffn_down.output_min",
		},
		{
			"ffn up 1 input_max (new naming)",
			"model.audio_tower.layers.0.feed_forward2.ffw_layer_1.input_max",
			"a.blk.0.ffn_up_1.input_max",
		},
		{
			"ffn norm 1 (new naming)",
			"model.audio_tower.layers.0.feed_forward2.pre_layer_norm.weight",
			"a.blk.0.ffn_norm_1.weight",
		},

		// Conformer lightweight conv1d
		{
			"conv dw weight",
			"model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight",
			"a.blk.0.conv_dw.weight",
		},
		{
			"conv norm (pre_layer_norm)",
			"model.audio_tower.conformer.0.lconv1d.pre_layer_norm.weight",
			"a.blk.0.conv_norm.weight",
		},
		{
			"norm conv (conv_norm)",
			"model.audio_tower.conformer.0.lconv1d.conv_norm.weight",
			"a.blk.0.norm_conv.weight",
		},
		{
			"conv pw1 weight",
			"model.audio_tower.conformer.0.lconv1d.linear_start.linear.weight",
			"a.blk.0.conv_pw1.weight",
		},
		{
			"conv pw2 weight",
			"model.audio_tower.conformer.0.lconv1d.linear_end.linear.weight",
			"a.blk.0.conv_pw2.weight",
		},

		// Audio embedder
		{
			"audio embedder projection weight",
			"model.embed_audio.embedding_projection.linear.weight",
			"mm.a.input_projection.weight",
		},
		{
			"audio embedder projection bias",
			"model.embed_audio.embedding_projection.linear.bias",
			"mm.a.input_projection.bias",
		},

		// Audio output projection
		{
			"audio output proj weight",
			"model.audio_tower.output_proj.weight",
			"mm.a.fc.weight",
		},
		{
			"audio output proj bias",
			"model.audio_tower.output_proj.bias",
			"mm.a.fc.bias",
		},

		// Verify vision tensors still work
		{
			"vision q weight",
			"model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
			"v.blk.0.attn_q.weight",
		},
		{
			"vision std bias",
			"model.vision_tower.std_bias",
			"v.std_bias",
		},
		{
			"vision std scale",
			"model.vision_tower.std_scale",
			"v.std_scale",
		},
		{
			"vision patch embd",
			"model.vision_tower.patch_embedder.input_proj.weight",
			"v.patch_embd.weight",
		},
		{
			"vision projector",
			"model.embed_vision.embedding_projection.linear.weight",
			"mm.input_projection.weight",
		},

		// Verify text tensors still work
		{
			"text attn q",
			"model.language_model.layers.0.self_attn.q_proj.weight",
			"blk.0.attn_q.weight",
		},
		{
			"text token embd",
			"model.language_model.embed_tokens.weight",
			"token_embd.weight",
		},
		{
			"text moe gate up fused",
			"model.language_model.layers.0.experts.gate_up_proj",
			"blk.0.ffn_gate_up_exps.weight",
		},
		{
			"text moe down",
			"model.language_model.layers.0.experts.down_proj",
			"blk.0.ffn_down_exps.weight",
		},
		{
			"text moe down with weight suffix",
			"model.language_model.layers.0.experts.down_proj.weight",
			"blk.0.ffn_down_exps.weight",
		},
		{
			"text moe per expert scale",
			"model.language_model.layers.0.router.per_expert_scale",
			"blk.0.ffn_down_exps.scale",
		},
		{
			"text moe per expert scale with weight suffix",
			"model.language_model.layers.0.router.per_expert_scale.weight",
			"blk.0.ffn_down_exps.scale",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := r.Replace(tt.in); got != tt.want {
				t.Errorf("Replace(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
