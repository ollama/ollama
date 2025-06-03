package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type gptossModel struct {
	ModelParameters
	HiddenLayers         uint32  `json:"num_hidden_layers"`
	HiddenSize           uint32  `json:"hidden_size"`
	IntermediateSize     uint32  `json:"intermediate_size"`
	AttentionHeads       uint32  `json:"num_attention_heads"`
	KeyValueHeads        uint32  `json:"num_key_value_heads"`
	HeadDim              uint32  `json:"head_dim"`
	Experts              uint32  `json:"num_experts"`
	ExpertsPerToken      uint32  `json:"experts_per_token"`
	RMSNormEpsilon       float32 `json:"rms_norm_eps"`
	InitialContextLength uint32  `json:"initial_context_length"`
	RopeTheta            float32 `json:"rope_theta"`
	RopeScalingFactor    float32 `json:"rope_scaling_factor"`
	SlidingWindow        uint32  `json:"sliding_window"`
}

var _ ModelConverter = (*gptossModel)(nil)

func (m *gptossModel) KV(t *Tokenizer) ggml.KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "gptoss"
	kv["general.file_type"] = uint32(4)
	kv["gptoss.context_length"] = uint32(m.RopeScalingFactor * float32(m.InitialContextLength))
	kv["gptoss.block_count"] = m.HiddenLayers
	kv["gptoss.embedding_length"] = m.HiddenSize
	kv["gptoss.feed_forward_length"] = m.IntermediateSize
	kv["gptoss.expert_count"] = m.Experts
	kv["gptoss.expert_used_count"] = m.ExpertsPerToken
	kv["gptoss.attention.head_count"] = m.AttentionHeads
	kv["gptoss.attention.head_count_kv"] = m.KeyValueHeads
	kv["gptoss.attention.key_length"] = m.HeadDim
	kv["gptoss.attention.value_length"] = m.HeadDim
	kv["gptoss.attention.layer_norm_rms_epsilon"] = cmp.Or(m.RMSNormEpsilon, 1e-5)
	kv["gptoss.attention.sliding_window"] = m.SlidingWindow
	kv["gptoss.rope.freq_base"] = m.RopeTheta
	kv["gptoss.rope.scaling.factor"] = m.RopeScalingFactor
	kv["gptoss.rope.scaling.original_context_length"] = m.InitialContextLength
	kv["tokenizer.ggml.bos_token_id"] = uint32(199998) // <|startoftext|>
	kv["tokenizer.ggml.add_bos_token"] = false
	kv["tokenizer.ggml.eos_token_id"] = uint32(199999) // <|endoftext|>
	kv["tokenizer.ggml.eos_token_ids"] = []int32{
		199999, /* <|endoftext|> */
		200002, /* <|return|> */
		200012, /* <|call|> */
	}
	kv["tokenizer.ggml.add_eos_token"] = false
	return kv
}

func (m *gptossModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
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

func (m *gptossModel) Replacements() []string {
	return []string{
		"block", "blk",
		"attn.norm", "attn_norm",
		"attn.qkv", "attn_qkv",
		"attn.sinks", "attn_sinks",
		"attn.out", "attn_out",
		"mlp.norm", "ffn_norm",
		"mlp.gate", "ffn_gate_inp",
		"mlp.mlp1_", "ffn_gate_up_exps.",
		"mlp.mlp2_", "ffn_down_exps.",
		"embedding", "token_embd",
		"norm", "output_norm",
		"unembedding", "output",
		"scale", "weight",
	}
}
