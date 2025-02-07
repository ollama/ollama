package convert

import "github.com/ollama/ollama/fs/ggml"

type gemma3Model struct {
	gemmaModel
	TextModel   gemma3TextModel   `json:"text_config"`
	VisionModel gemma3VisionModel `json:"vision_config"`
}

type gemma3TextModel struct {
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	HeadDim               uint32  `json:"head_dim"`
	SlidingWindow         uint32  `json:"sliding_window"`
	AttentionLogitSoftcap float32 `json:"attn_logit_softcapping"`
	FinalLogitSoftcap     float32 `json:"final_logit_softcapping"`
	RopeLocalTheta        float32 `json:"rope_local_base_freq"`
	RopeGlobalTheta       float32 `json:"rope_global_base_freq"`
}

type gemma3VisionModel struct {
	ImageSize    uint32 `json:"image_size"`
	NumChannels  uint32 `json:"num_channels"`
	HiddenLayers uint32 `json:"num_hidden_layers"`
}

func (p *gemma3Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma3"
	kv["gemma3.context_length"] = p.TextModel.MaxPositionEmbeddings
	kv["gemma3.embedding_length"] = p.TextModel.HiddenSize
	kv["gemma3.block_count"] = p.TextModel.HiddenLayers
	kv["gemma3.text.feed_forward_length"] = p.TextModel.IntermediateSize
	kv["gemma3.attention.head_count"] = p.TextModel.NumAttentionHeads
	kv["gemma3.attention.head_count_kv"] = p.TextModel.NumKeyValueHeads
	kv["gemma3.text.attention.layer_norm_rms_epsilon"] = p.TextModel.RMSNormEPS
	kv["gemma3.attention.key_length"] = p.TextModel.HeadDim
	kv["gemma3.attention.value_length"] = p.TextModel.HeadDim
	kv["gemma3.text.attention.sliding_window"] = p.TextModel.SlidingWindow
	kv["gemma3.text.final_logit_softcapping"] = p.TextModel.FinalLogitSoftcap
	kv["gemma3.text.rope.local.freq_base"] = p.TextModel.RopeLocalTheta
	kv["gemma3.text.rope.global.freq_base"] = p.TextModel.RopeGlobalTheta
	kv["tokenizer.ggml.bos_token_id"] = uint32(2)
	kv["tokenizer.ggml.eot_token_id"] = uint32(1)
	kv["gemma3.vision.image_size"] = p.VisionModel.ImageSize
	kv["gemma3.vision.num_channels"] = p.VisionModel.NumChannels
	kv["gemma3.vision.block_count"] = p.VisionModel.HiddenLayers
	return kv
}

func (p *gemma3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"vision_model.vision_model", "v",
		"language_model.", "",
		"model.layers", "blk",
		"encoder.layers", "blk",
		"vision_tower.vision_model.embeddings", "v",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
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
