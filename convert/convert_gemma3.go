package convert

import (
	"cmp"
	"slices"
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
	MaxPositionEmbeddings    uint32   `json:"max_position_embeddings"`
	NumAttentionHeads        uint32   `json:"num_attention_heads"`
	NumKeyValueHeads         uint32   `json:"num_key_value_heads"`
	RMSNormEPS               float32  `json:"rms_norm_eps"`
	HeadDim                  uint32   `json:"head_dim"`
	FinalLogitSoftcap        float32  `json:"final_logit_softcapping"`
	RopeLocalTheta           float32  `json:"rope_local_base_freq"`
	RopeTheta                float32  `json:"rope_theta"`
	SlidingWindow            uint32   `json:"sliding_window"`
	SlidingWindowPattern     *uint32  `json:"sliding_window_pattern"`
	LayerTypes               []string `json:"layer_types"`
	MultiModalTokensPerImage uint32   `json:"mm_tokens_per_image"`
	RopeScaling              *struct {
		Type                          string  `json:"rope_type"`
		Factor                        float32 `json:"factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
		ExtrapolationFactor           float32 `json:"extrapolation_factor"`
		BetaFast                      float32 `json:"beta_fast"`
		BetaSlow                      float32 `json:"beta_slow"`
	} `json:"rope_scaling"`
}

const (
	gemma4BLayerCount  = 34
	gemma12BLayerCount = 48
	gemma27BLayerCount = 62
)

func (p *gemma3Model) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "gemma3"

	numBlocks := cmp.Or(p.HiddenLayers, p.TextModel.HiddenLayers)
	kv["gemma3.block_count"] = numBlocks

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

	kv["gemma3.attention.head_count"] = numHeads
	kv["gemma3.attention.head_count_kv"] = numKVHeads

	switch p.Architecture {
	case "Gemma3ForCausalLM":
		kv["gemma3.context_length"] = p.MaxPositionEmbeddings
		kv["gemma3.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
		kv["gemma3.attention.key_length"] = p.HeadDim
		kv["gemma3.attention.value_length"] = p.HeadDim
		kv["gemma3.attention.sliding_window"] = p.SlidingWindow

		// The sliding window pattern is either provided as the sliding_window_pattern
		// key (an int) or as the layer_types key (a list of strings).
		if p.SlidingWindowPattern != nil || len(p.LayerTypes) > 0 {
			kv["gemma3.attention.sliding_window_pattern"] = slices.Collect(func(yield func(bool) bool) {
				for i := range numBlocks {
					var isLocal bool
					if len(p.LayerTypes) > 0 && int(i) < len(p.LayerTypes) {
						isLocal = p.LayerTypes[i] == "sliding_attention"
					} else if p.SlidingWindowPattern != nil && *p.SlidingWindowPattern > 0 {
						isLocal = (i+1)%*p.SlidingWindowPattern != 0
					}
					if !yield(isLocal) {
						break
					}
				}
			})
		}
		if p.FinalLogitSoftcap > 0 {
			kv["gemma3.final_logit_softcapping"] = p.FinalLogitSoftcap
		}
		kv["gemma3.rope.local.freq_base"] = cmp.Or(p.RopeLocalTheta, 10000.0)
		kv["gemma3.rope.freq_base"] = cmp.Or(p.RopeTheta, 1000000.0)
		if p.RopeScaling != nil && p.RopeScaling.Type == "yarn" && p.RopeScaling.Factor > 0 {
			kv["gemma3.rope.scaling.type"] = "yarn"
			kv["gemma3.rope.scaling.factor"] = p.RopeScaling.Factor
			kv["gemma3.rope.scaling.original_context_length"] = p.RopeScaling.OriginalMaxPositionEmbeddings
			kv["gemma3.rope.scaling.extrapolation_factor"] = cmp.Or(p.RopeScaling.ExtrapolationFactor, float32(1.0))
			kv["gemma3.rope.scaling.beta_fast"] = cmp.Or(p.RopeScaling.BetaFast, float32(64.0))
			kv["gemma3.rope.scaling.beta_slow"] = cmp.Or(p.RopeScaling.BetaSlow, float32(1.0))
		}

		kv["gemma3.embedding_length"] = p.HiddenSize
		kv["gemma3.feed_forward_length"] = p.IntermediateSize
	default:
		kv["gemma3.context_length"] = cmp.Or(p.MaxPositionEmbeddings, 131072)
		kv["gemma3.embedding_length"] = p.TextModel.HiddenSize
		kv["gemma3.feed_forward_length"] = p.TextModel.IntermediateSize
		kv["gemma3.attention.sliding_window"] = p.TextModel.SlidingWindow
		kv["gemma3.vision.block_count"] = p.VisionModel.NumHiddenLayers
		kv["gemma3.vision.embedding_length"] = p.VisionModel.HiddenSize
		kv["gemma3.vision.feed_forward_length"] = p.VisionModel.IntermediateSize
		kv["gemma3.vision.image_size"] = p.VisionModel.ImageSize
		kv["gemma3.vision.patch_size"] = p.VisionModel.PatchSize
		kv["gemma3.vision.num_channels"] = cmp.Or(p.VisionModel.NumChannels, 3)
		kv["gemma3.vision.attention.head_count"] = p.VisionModel.NumAttentionHeads
		kv["gemma3.vision.attention.layer_norm_epsilon"] = cmp.Or(p.VisionModel.LayerNormEpsilon, 1e-6)
		kv["gemma3.attention.key_length"] = cmp.Or(p.TextModel.HeadDim, 256)
		kv["gemma3.attention.value_length"] = cmp.Or(p.TextModel.HeadDim, 256)
	}

	if p.MultiModalTokensPerImage > 0 {
		kv["gemma3.mm.tokens_per_image"] = p.MultiModalTokensPerImage
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
