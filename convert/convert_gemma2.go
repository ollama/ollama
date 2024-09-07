package convert

import (
	"github.com/ollama/ollama/llm"
)

type gemma2Model struct {
	gemmaModel
	SlidingWindow         uint32  `json:"sliding_window"`
	AttentionLogitSoftcap float32 `json:"attn_logit_softcapping"`
	FinalLogitSoftcap     float32 `json:"final_logit_softcapping"`
}

func (p *gemma2Model) KV(t *Tokenizer) llm.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma2"
	kv["gemma2.context_length"] = p.MaxPositionEmbeddings
	kv["gemma2.embedding_length"] = p.HiddenSize
	kv["gemma2.block_count"] = p.HiddenLayers
	kv["gemma2.feed_forward_length"] = p.IntermediateSize
	kv["gemma2.attention.head_count"] = p.NumAttentionHeads
	kv["gemma2.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["gemma2.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["gemma2.attention.key_length"] = p.HeadDim
	kv["gemma2.attention.value_length"] = p.HeadDim
	kv["gemma2.attention.sliding_window"] = p.SlidingWindow
	kv["gemma2.attn_logit_softcapping"] = p.AttentionLogitSoftcap
	kv["gemma2.final_logit_softcapping"] = p.FinalLogitSoftcap
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
