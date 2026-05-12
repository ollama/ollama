package convert

import "github.com/ollama/ollama/fs/ggml"

type qwen2Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           struct {
		Type                          string     `json:"type"`
		Factor                        ropeFactor `json:"factor"`
		OriginalMaxPositionEmbeddings uint32     `json:"original_max_position_embeddings"`
		MropeSection                  []int32    `json:"mrope_section"`
	} `json:"rope_scaling"`
	RMSNormEPS float32 `json:"rms_norm_eps"`
}

var _ ModelConverter = (*qwen2Model)(nil)

func (q *qwen2Model) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen2"
	kv["qwen2.block_count"] = q.HiddenLayers
	kv["qwen2.context_length"] = q.MaxPositionEmbeddings
	kv["qwen2.embedding_length"] = q.HiddenSize
	kv["qwen2.feed_forward_length"] = q.IntermediateSize
	kv["qwen2.attention.head_count"] = q.NumAttentionHeads
	kv["qwen2.attention.head_count_kv"] = q.NumKeyValueHeads
	kv["qwen2.rope.freq_base"] = q.RopeTheta
	kv["qwen2.attention.layer_norm_rms_epsilon"] = q.RMSNormEPS

	switch q.RopeScaling.Type {
	case "":
		// no scaling
	case "yarn":
		kv["qwen2.rope.scaling.type"] = q.RopeScaling.Type
		kv["qwen2.rope.scaling.factor"] = q.RopeScaling.Factor
	case "mrope", "default":
		kv["qwen2.rope.mrope_section"] = q.RopeScaling.MropeSection
	default:
		panic("unknown rope scaling type")
	}
	return kv
}

func (q *qwen2Model) Tensors(ts []Tensor) []*ggml.Tensor {
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

func (p *qwen2Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"model.norm", "output_norm",
	}
}
