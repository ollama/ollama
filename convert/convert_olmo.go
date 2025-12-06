package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type olmoModel struct {
	ModelParameters

	HiddenSize            uint32   `json:"hidden_size"`
	NumHiddenLayers       uint32   `json:"num_hidden_layers"`
	IntermediateSize      uint32   `json:"intermediate_size"`
	NumAttentionHeads     uint32   `json:"num_attention_heads"`
	NumKeyValueHeads      uint32   `json:"num_key_value_heads"`
	MaxPositionEmbeddings uint32   `json:"max_position_embeddings"`
	RMSNormEPS            float32  `json:"rms_norm_eps"`
	RopeTheta             float32  `json:"rope_theta"`
	ClampKQV              float32  `json:"f_clamp_kqv"`
	SlidingWindow         uint32   `json:"sliding_window"`
	LayerTypes            []string `json:"layer_types"`
}

var _ ModelConverter = (*olmoModel)(nil)

func (p *olmoModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "olmo"
	kv["olmo.block_count"] = p.NumHiddenLayers
	kv["olmo.context_length"] = p.MaxPositionEmbeddings
	kv["olmo.embedding_length"] = p.HiddenSize
	kv["olmo.feed_forward_length"] = p.IntermediateSize
	kv["olmo.attention.head_count"] = p.NumAttentionHeads
	kv["olmo.attention.head_count_kv"] = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)

	if p.RopeTheta > 0 {
		kv["olmo.rope.freq_base"] = p.RopeTheta
	} else {
		kv["olmo.rope.freq_base"] = float32(10000.0)
	}

	if p.RMSNormEPS > 0 {
		kv["olmo.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if p.ClampKQV > 0 {
		kv["olmo.attention.clamp_kqv"] = p.ClampKQV
	}

	if p.SlidingWindow > 0 {
		kv["olmo.attention.sliding_window"] = p.SlidingWindow
	}

	if len(p.LayerTypes) > 0 {
		kv["olmo.attention.layer_types"] = p.LayerTypes
	}

	return kv
}

func (p *olmoModel) Tensors(ts []Tensor) []*ggml.Tensor {
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

func (p *olmoModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"model.norm", "output_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_norm", "attn_k_norm",
		"post_attention_layernorm", "post_attention_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
	}
}
