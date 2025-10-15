package convert

import "github.com/ollama/ollama/fs/ggml"

type gemma2Model struct {
	gemmaModel
	SlidingWindow         uint32  `json:"sliding_window"`
	AttentionLogitSoftcap float32 `json:"attn_logit_softcapping"`
	FinalLogitSoftcap     float32 `json:"final_logit_softcapping"`
}

func (p *gemma2Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma2"
	kv["context_length"] = p.MaxPositionEmbeddings
	kv["embedding_length"] = p.HiddenSize
	kv["block_count"] = p.HiddenLayers
	kv["feed_forward_length"] = p.IntermediateSize
	kv["attention.head_count"] = p.NumAttentionHeads
	kv["attention.head_count_kv"] = p.NumKeyValueHeads
	kv["attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["attention.key_length"] = p.HeadDim
	kv["attention.value_length"] = p.HeadDim
	kv["attention.sliding_window"] = p.SlidingWindow
	kv["attn_logit_softcapping"] = p.AttentionLogitSoftcap
	kv["final_logit_softcapping"] = p.FinalLogitSoftcap
	kv["tokenizer.ggml.eot_token_id"] = uint32(107)
	kv["tokenizer.ggml.middle_token_id"] = uint32(68)
	kv["tokenizer.ggml.prefix_token_id"] = uint32(67)
	kv["tokenizer.ggml.suffix_token_id"] = uint32(69)
	return kv
}

func (p *gemma2Model) Replacements() []string {
	return []string{
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
	}
}
