package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type commandrModel struct {
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
}

var _ ModelConverter = (*commandrModel)(nil)

func (p *commandrModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "command-r"
	kv["general.name"] = "command-r"
	kv["context_length"] = cmp.Or(p.MaxLength, p.MaxPositionEmbeddings, p.NCtx)
	kv["embedding_length"] = p.HiddenSize
	kv["block_count"] = p.HiddenLayers
	kv["feed_forward_length"] = p.IntermediateSize
	kv["attention.head_count"] = p.NumAttentionHeads
	kv["attention.head_count_kv"] = p.NumKeyValueHeads
	kv["attention.layer_norm_epsilon"] = p.LayerNormEPS
	kv["rope.freq_base"] = p.RopeTheta
	kv["max_position_embeddings"] = cmp.Or(p.MaxLength, p.MaxPositionEmbeddings)
	kv["logit_scale"] = p.LogitScale
	kv["rope.scaling.type"] = "none"

	return kv
}

func (p *commandrModel) Tensors(ts []Tensor) []*ggml.Tensor {
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

func (p *commandrModel) Replacements() []string {
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
