package convert

import (
	"strings"

	"github.com/ollama/ollama/llm"
)

type gemma2 struct {
	gemma
}

func (p *gemma2) KV(t *Tokenizer) llm.KV {
	kv := p.Parameters.KV(t)
	kv["general.architecture"] = "gemma2"
	kv["general.name"] = "gemma2"
	kv["gemma2.context_length"] = p.MaxPositionEmbeddings
	kv["gemma2.embedding_length"] = p.HiddenSize
	kv["gemma2.block_count"] = p.HiddenLayers
	kv["gemma2.feed_forward_length"] = p.IntermediateSize
	kv["gemma2.attention.head_count"] = p.NumAttentionHeads
	kv["gemma2.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["gemma2.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["gemma2.attention.key_length"] = p.HeadDim
	kv["gemma2.attention.value_length"] = p.HeadDim
	kv["tokenizer.ggml.eot_token_id"] = uint32(107)
	kv["tokenizer.ggml.middle_token_id"] = uint32(68)
	kv["tokenizer.ggml.prefix_token_id"] = uint32(67)
	kv["tokenizer.ggml.suffix_token_id"] = uint32(69)
	return kv
}

func (p *gemma2) tensorName(n string) string {
	return p.gemma.tensorName(
		strings.NewReplacer(
			"post_attention_layernorm", "post_attention_norm",
			"pre_feedforward_layernorm", "ffn_norm",
			"post_feedforward_layernorm", "post_ffw_norm",
		).Replace(n),
	)
}
