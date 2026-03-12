package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type ropeScaling struct {
	Factor                    float32 `json:"factor"`
	OriginalMaxPositionEmbeds uint32  `json:"original_max_position_embeddings"`
	AttentionFactor           float32 `json:"attention_factor"`
	BetaFast                  float32 `json:"beta_fast"`
	BetaSlow                  float32 `json:"beta_slow"`
	RopeType                  string  `json:"rope_type"`
	ExtrapolationFactor       float32 `json:"extrapolation_factor"`
}

type olmoModel struct {
	ModelParameters

	HiddenSize            uint32       `json:"hidden_size"`
	NumHiddenLayers       uint32       `json:"num_hidden_layers"`
	IntermediateSize      uint32       `json:"intermediate_size"`
	NumAttentionHeads     uint32       `json:"num_attention_heads"`
	NumKeyValueHeads      uint32       `json:"num_key_value_heads"`
	MaxPositionEmbeddings uint32       `json:"max_position_embeddings"`
	RMSNormEPS            float32      `json:"rms_norm_eps"`
	RopeTheta             float32      `json:"rope_theta"`
	RopeScaling           *ropeScaling `json:"rope_scaling"`
	SlidingWindow         uint32       `json:"sliding_window"`
	LayerTypes            []string     `json:"layer_types"`
}

var _ ModelConverter = (*olmoModel)(nil)

func (p *olmoModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "olmo3"
	kv["olmo3.block_count"] = p.NumHiddenLayers
	kv["olmo3.context_length"] = p.MaxPositionEmbeddings
	kv["olmo3.embedding_length"] = p.HiddenSize
	kv["olmo3.feed_forward_length"] = p.IntermediateSize
	kv["olmo3.attention.head_count"] = p.NumAttentionHeads
	kv["olmo3.attention.head_count_kv"] = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)

	if p.RopeTheta > 0 {
		kv["olmo3.rope.freq_base"] = p.RopeTheta
	}

	if p.RopeScaling != nil {
		if p.RopeScaling.Factor > 0 {
			kv["olmo3.rope.scaling.factor"] = p.RopeScaling.Factor
		}
		if p.RopeScaling.OriginalMaxPositionEmbeds > 0 {
			kv["olmo3.rope.scaling.original_context_length"] = p.RopeScaling.OriginalMaxPositionEmbeds
		}
		if p.RopeScaling.AttentionFactor > 0 {
			kv["olmo3.rope.scaling.attn_factor"] = p.RopeScaling.AttentionFactor
		}
		if p.RopeScaling.RopeType != "" {
			kv["olmo3.rope.scaling.type"] = p.RopeScaling.RopeType
		}
	}

	if p.RMSNormEPS > 0 {
		kv["olmo3.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if p.SlidingWindow > 0 {
		kv["olmo3.attention.sliding_window"] = p.SlidingWindow
	}

	if len(p.LayerTypes) > 0 {
		slidingPattern := make([]bool, len(p.LayerTypes))
		for i, layerType := range p.LayerTypes {
			slidingPattern[i] = (layerType == "sliding_attention")
		}
		kv["olmo3.attention.sliding_window_pattern"] = slidingPattern
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
