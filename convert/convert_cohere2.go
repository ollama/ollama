package convert

import (
	"cmp"

	"github.com/ollama/ollama/llm"
)

type cohere2Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	LayerNormEPS          float32 `json:"layer_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	UseQKNorm             bool    `json:"use_qk_norm"`
	MaxLength             uint32  `json:"model_max_length"`
	LogitScale            float32 `json:"logit_scale"`
	NCtx                  uint32  `json:"n_ctx"`
	SlidingWindow         uint32  `json:"sliding_window"`
	HeadDim               uint32  `json:"head_dim"`
	RotaryPct             float32 `json:"rotary_pct"`
	VocabSize             uint32  `json:"vocab_size"`
}

var _ ModelConverter = (*cohere2Model)(nil)

func (p *cohere2Model) KV(t *Tokenizer) llm.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "cohere2"
	kv["general.name"] = "C4Ai Command R7B"
	kv["cohere2.context_length"] = cmp.Or(p.MaxLength, p.MaxPositionEmbeddings, p.NCtx)
	kv["cohere2.embedding_length"] = p.HiddenSize
	kv["cohere2.block_count"] = p.HiddenLayers
	kv["cohere2.feed_forward_length"] = p.IntermediateSize
	kv["cohere2.attention.head_count"] = p.NumAttentionHeads
	kv["cohere2.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["cohere2.attention.key_length"] = p.HeadDim
	kv["cohere2.attention.layer_norm_epsilon"] = p.LayerNormEPS
	kv["cohere2.attention.sliding_window"] = p.SlidingWindow
	kv["cohere2.attention.value_length"] = p.HeadDim
	kv["cohere2.max_position_embeddings"] = cmp.Or(p.MaxLength, p.MaxPositionEmbeddings)
	kv["cohere2.logit_scale"] = p.LogitScale
	kv["cohere2.rope.dimension_count"] = uint32(p.RotaryPct * float32(p.HiddenSize/p.NumAttentionHeads))
	kv["cohere2.rope.freq_base"] = p.RopeTheta
	kv["cohere2.rope.scaling.type"] = "none"
	kv["cohere2.vocab_size"] = p.VocabSize

	return kv
}

func (p *cohere2Model) Tensors(ts []Tensor) []llm.Tensor {
	var out []llm.Tensor
	for _, t := range ts {
		out = append(out, llm.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *cohere2Model) Replacements() []string {
	return []string{
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_norm", "attn_k_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"self_attn.k_proj", "attn_k",
		"self_attn.o_proj", "attn_output",
		"self_attn.q_proj", "attn_q",
		"self_attn.v_proj", "attn_v",
		"model.norm", "output_norm",
		"model.embed_tokens", "token_embd",
	}
}
