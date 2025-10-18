package convert

import (
	"cmp"

	"github.com/ollama/ollama/fs/ggml"
)

type gemma3Model struct {
	gemmaModel
	Architecture string
	TextModel    struct {
		HeadDim          uint32 `json:"head_dim"`
		HiddenSize       uint32 `json:"hidden_size"`
		HiddenLayers     uint32 `json:"num_hidden_layers"`
		IntermediateSize uint32 `json:"intermediate_size"`
		SlidingWindow    uint32 `json:"sliding_window"`
	} `json:"text_config"`
	VisionModel struct {
		NumAttentionHeads uint32  `json:"num_attention_heads"` // attention.head_count 16
		LayerNormEpsilon  float32 `json:"layer_norm_eps"`      // attention.layer_norm_epsilon 1e-05
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`   // block_count 32
		HiddenSize        uint32  `json:"hidden_size"`         // embedding_length 1280
		IntermediateSize  uint32  `json:"intermediate_size"`   // feed_forward_length 5120
		ImageSize         uint32  `json:"image_size"`          // image_size 560
		NumChannels       uint32  `json:"num_channels"`        // num_channels 3
		PatchSize         uint32  `json:"patch_size"`          // patch_size 14
	} `json:"vision_config"`
	MaxPositionEmbeddings    uint32  `json:"max_position_embeddings"`
	NumAttentionHeads        uint32  `json:"num_attention_heads"`
	NumKeyValueHeads         uint32  `json:"num_key_value_heads"`
	RMSNormEPS               float32 `json:"rms_norm_eps"`
	HeadDim                  uint32  `json:"head_dim"`
	FinalLogitSoftcap        float32 `json:"final_logit_softcapping"`
	RopeLocalTheta           float32 `json:"rope_local_base_freq"`
	RopeGlobalTheta          float32 `json:"rope_global_base_freq"`
	SlidingWindow            uint32  `json:"sliding_window"`
	MultiModalTokensPerImage uint32  `json:"mm_tokens_per_image"`
}

const (
	gemma4BLayerCount  = 34
	gemma12BLayerCount = 48
	gemma27BLayerCount = 62
)

func (p *gemma3Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV("gemma3", t)
	numBlocks := cmp.Or(p.HiddenLayers, p.TextModel.HiddenLayers)
	kv["block_count"] = numBlocks

	var (
		numHeads   uint32
		numKVHeads uint32
	)

	switch numBlocks {
	case gemma4BLayerCount:
		numHeads = 8
		numKVHeads = 4
	case gemma12BLayerCount:
		numHeads = 16
		numKVHeads = 8
	case gemma27BLayerCount:
		numHeads = 32
		numKVHeads = 16
	default:
		numHeads = p.NumAttentionHeads
		numKVHeads = p.NumKeyValueHeads
	}

	kv["attention.head_count"] = numHeads
	kv["attention.head_count_kv"] = numKVHeads

	switch p.Architecture {
	case "Gemma3ForCausalLM":
		kv["context_length"] = p.MaxPositionEmbeddings
		kv["attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
		kv["attention.key_length"] = p.HeadDim
		kv["attention.value_length"] = p.HeadDim
		kv["attention.sliding_window"] = p.SlidingWindow
		kv["final_logit_softcapping"] = cmp.Or(p.FinalLogitSoftcap, 30)
		kv["rope.local.freq_base"] = cmp.Or(p.RopeLocalTheta, 10000.0)
		kv["rope.global.freq_base"] = cmp.Or(p.RopeGlobalTheta, 1000000.0)
		kv["embedding_length"] = p.HiddenSize
		kv["feed_forward_length"] = p.IntermediateSize
	default:
		kv["context_length"] = cmp.Or(p.MaxPositionEmbeddings, 131072)
		kv["embedding_length"] = p.TextModel.HiddenSize
		kv["feed_forward_length"] = p.TextModel.IntermediateSize
		kv["attention.sliding_window"] = p.TextModel.SlidingWindow
		kv["vision.block_count"] = p.VisionModel.NumHiddenLayers
		kv["vision.embedding_length"] = p.VisionModel.HiddenSize
		kv["vision.feed_forward_length"] = p.VisionModel.IntermediateSize
		kv["vision.image_size"] = p.VisionModel.ImageSize
		kv["vision.patch_size"] = p.VisionModel.PatchSize
		kv["vision.num_channels"] = cmp.Or(p.VisionModel.NumChannels, 3)
		kv["vision.attention.head_count"] = p.VisionModel.NumAttentionHeads
		kv["vision.attention.layer_norm_epsilon"] = cmp.Or(p.VisionModel.LayerNormEpsilon, 1e-6)
		kv["attention.key_length"] = cmp.Or(p.TextModel.HeadDim, 256)
		kv["attention.value_length"] = cmp.Or(p.TextModel.HeadDim, 256)
	}

	if p.MultiModalTokensPerImage > 0 {
		kv["mm.tokens_per_image"] = p.MultiModalTokensPerImage
	}

	return kv
}

func (p *gemma3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"vision_tower.vision_model.embeddings", "v",
		"vision_tower.vision_model", "v",
		"vision_model.vision_model.embeddings", "v",
		"vision_model.vision_model", "v",
		"language_model.", "",
		"model.layers", "blk",
		"encoder.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"self_attn.out_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
		"input_projection_weight", "input_projection.weight",
		"multi_modal_projector", "mm",
	}
}
