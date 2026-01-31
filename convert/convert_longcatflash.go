package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type longcatflashModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	NumLayers             uint32  `json:"num_layers"`
	FFNHiddenSize         uint32  `json:"ffn_hidden_size"`
	ExpertFFNHiddenSize   uint32  `json:"expert_ffn_hidden_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	RopeTheta     float32 `json:"rope_theta"`
	QKRopeHeadDim uint32  `json:"qk_rope_head_dim"`
	QKNopeHeadDim uint32  `json:"qk_nope_head_dim"`
	VHeadDim      uint32  `json:"v_head_dim"`
	QLoraRank     uint32  `json:"q_lora_rank"`
	KVLoraRank    uint32  `json:"kv_lora_rank"`

	ExpertCount         uint32  `json:"n_routed_experts"`
	ExpertUsedCount     uint32  `json:"moe_topk"`
	RoutedScalingFactor float32 `json:"routed_scaling_factor"`
	ZeroExpertNum       uint32  `json:"zero_expert_num"`
	SlidingWindow       uint32  `json:"sliding_window"`

	RopeScaling struct {
		Factor                        float32 `json:"factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
		Type                          string  `json:"rope_type"`
		MScaleAllDim                  float32 `json:"mscale_all_dim"`
	} `json:"rope_scaling"`
}

var _ ModelConverter = (*longcatflashModel)(nil)

func (m *longcatflashModel) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "longcatflash"
	kv["general.file_type"] = uint32(4)
	kv["longcatflash.block_count"] = cmp.Or(m.NumLayers, 0)
	kv["longcatflash.embedding_length"] = m.HiddenSize
	kv["longcatflash.context_length"] = cmp.Or(m.MaxPositionEmbeddings, uint32(327680))
	kv["longcatflash.attention.head_count"] = m.NumAttentionHeads
	kv["longcatflash.attention.head_count_kv"] = cmp.Or(m.NumKeyValueHeads, m.NumAttentionHeads)
	kv["longcatflash.attention.key_length"] = m.QKNopeHeadDim + m.QKRopeHeadDim
	kv["longcatflash.attention.value_length"] = m.VHeadDim
	kv["longcatflash.attention.layer_norm_rms_epsilon"] = cmp.Or(m.RMSNormEPS, 1e-5)
	kv["longcatflash.attention.q_lora_rank"] = m.QLoraRank
	kv["longcatflash.attention.kv_lora_rank"] = m.KVLoraRank
	kv["longcatflash.attention.sliding_window"] = m.SlidingWindow
	kv["longcatflash.rope.freq_base"] = cmp.Or(m.RopeTheta, float32(5000000))
	kv["longcatflash.rope.dimension_count"] = m.QKRopeHeadDim
	kv["longcatflash.rope.scaling.factor"] = cmp.Or(m.RopeScaling.Factor, 1)
	kv["longcatflash.rope.scaling.original_context_length"] = m.RopeScaling.OriginalMaxPositionEmbeddings
	kv["longcatflash.rope.scaling.type"] = cmp.Or(m.RopeScaling.Type, "default")
	kv["longcatflash.rope.scaling.yarn_log_multiplier"] = 0.1 * m.RopeScaling.MScaleAllDim
	kv["longcatflash.expert_count"] = m.ExpertCount + m.ZeroExpertNum
	kv["longcatflash.expert_used_count"] = m.ExpertUsedCount
	kv["longcatflash.expert_weights_scale"] = cmp.Or(m.RoutedScalingFactor, 1)
	return kv
}

func (m *longcatflashModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"model.norm", "output_norm",
		"input_layernorm.0", "attn_norm_0",
		"input_layernorm.1", "attn_norm_1",
		"post_attention_layernorm.0", "ffn_norm_0",
		"post_attention_layernorm.1", "ffn_norm_1",
		"self_attn.0.q_a_proj", "attn_q_a_0",
		"self_attn.0.q_a_layernorm", "attn_q_a_norm_0",
		"self_attn.0.q_b_proj", "attn_q_b_0",
		"self_attn.0.kv_a_proj_with_mqa", "attn_kv_a_mqa_0",
		"self_attn.0.kv_a_layernorm", "attn_kv_a_norm_0",
		"self_attn.0.kv_b_proj", "attn_kv_b_0",
		"self_attn.0.o_proj", "attn_out_0",
		"self_attn.1.q_a_proj", "attn_q_a_1",
		"self_attn.1.q_a_layernorm", "attn_q_a_norm_1",
		"self_attn.1.q_b_proj", "attn_q_b_1",
		"self_attn.1.kv_a_proj_with_mqa", "attn_kv_a_mqa_1",
		"self_attn.1.kv_a_layernorm", "attn_kv_a_norm_1",
		"self_attn.1.kv_b_proj", "attn_kv_b_1",
		"self_attn.1.o_proj", "attn_out_1",
		"mlp.router.classifier", "ffn_gate_inp",
		"mlp.router.e_score_correction_bias", "exp_probs_b.bias",
		"mlp.experts.gate_up_proj", "ffn_gate_up_exps",
		"mlp.experts.down_proj", "ffn_down_exps",
		"mlps.0.gate_proj", "ffn_gate_0",
		"mlps.0.up_proj", "ffn_up_0",
		"mlps.0.down_proj", "ffn_down_0",
		"mlps.1.gate_proj", "ffn_gate_1",
		"mlps.1.up_proj", "ffn_up_1",
		"mlps.1.down_proj", "ffn_down_1",
	}
}

func (m *longcatflashModel) Tensors(ts []Tensor) []*ggml.Tensor {
	out := make([]*ggml.Tensor, 0, len(ts))
	for _, t := range ts {
		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}
	return out
}
