package hy4_flash

import (
	"encoding/json"
	"fmt"
)

type config struct {
	Text              textConfig    `json:"text_config"`
	Vision            *visionConfig `json:"vision_config"`
	ImageStartTokenID int32         `json:"image_start_token_id"`
	ImageEndTokenID   int32         `json:"image_end_token_id"`
	ImageTokenID      int32         `json:"image_token_id"`
	VideoTokenID      int32         `json:"video_token_id"`
	DraftLayers       int           `json:"dfly_num_hidden_layers"`
	DraftKVLayers     []int         `json:"dfly_target_kv_layers"`
	DraftIndexLayers  []int         `json:"dfly_target_index_layer"`
	DraftIndexTopK    int           `json:"dfly_target_index_topk"`
	DraftMaskToken    int32         `json:"dfly_mask_token_id"`
	DraftFusion       string        `json:"dfly_mtp_fusion"`
	DraftCorrection   bool          `json:"dfly_enable_hidden_correction"`
	DraftLocalWindow  int           `json:"dfly_context_local_window"`
	DraftCausal       *bool         `json:"dfly_attn_causal"`
	Speculative       struct {
		Method string `json:"method"`
		Tokens int    `json:"num_speculative_tokens"`
	} `json:"speculative_config"`
}

type textConfig struct {
	HiddenSize  int     `json:"hidden_size"`
	Layers      int     `json:"num_hidden_layers"`
	Context     int     `json:"max_position_embeddings"`
	RMSEps      float32 `json:"rms_norm_eps"`
	Streams     int     `json:"hc_mult"`
	HCEps       float32 `json:"hc_eps"`
	HCMagnitude float32 `json:"hc_magnitude"`

	AttentionTypes []string              `json:"layer_types"`
	Rope           map[string]ropeConfig `json:"rope_parameters"`
	Indexed        attentionShape
	Sliding        attentionShape
	Heads          int `json:"num_attention_heads"`
	NoPE           int `json:"qk_nope_head_dim"`
	RoPE           int `json:"qk_rope_head_dim"`
	Value          int `json:"v_head_dim"`
	SlidingHeads   int `json:"swa_num_attention_heads"`
	SlidingNoPE    int `json:"swa_qk_nope_head_dim"`
	SlidingRoPE    int `json:"swa_qk_rope_head_dim"`
	SlidingValue   int `json:"swa_v_head_dim"`
	Window         int `json:"sliding_window"`
	QRank          int `json:"q_lora_rank"`
	KVRank         int `json:"kv_lora_rank"`

	IndexRatio int `json:"compressed_indexer_compress_ratio"`
	IndexTopK  int `json:"compressed_indexer_group_topk"`
	IndexHeads int `json:"compressed_indexer_num_attention_heads"`
	IndexDim   int `json:"compressed_indexer_head_dim"`

	MLPTypes       []string `json:"mlp_layer_types"`
	Experts        int      `json:"n_routed_experts"`
	ExpertsPerTok  int      `json:"num_experts_per_tok"`
	ExpertWidth    int      `json:"moe_intermediate_size"`
	LatentWidth    int      `json:"routed_expert_hidden_size"`
	RouterGroups   int      `json:"n_group"`
	RouterGroupsK  int      `json:"topk_group"`
	NormalizeRoute bool     `json:"norm_topk_prob"`
	RouteScale     float32  `json:"routed_scaling_factor"`
	ExpertClamp    float32  `json:"swiglu_limit"`

	EngramLayers []int              `json:"engram_layer_ids"`
	EngramModuli map[string][][]int `json:"engram_layer_head_vocab_sizes"`
	EngramOrder  int                `json:"engram_max_ngram_size"`
	EngramHeads  int                `json:"engram_n_head_per_ngram"`
	EngramWidth  int                `json:"engram_n_embed_per_ngram"`
	EngramFP8    bool               `json:"engram_fp8_embedding"`
	EngramPadID  int32              `json:"engram_pad_id"`
}

type ropeConfig struct {
	Theta    float32 `json:"rope_theta"`
	Type     string  `json:"rope_type"`
	Sections [3]int  `json:"mrope_section"`
}

type attentionShape struct {
	heads, plain, rotary, value int
	theta                       float32
	sections                    [3]int
}

const (
	indexedAttention = "deepseek_sparse_attention"
	slidingAttention = "sliding_attention"
)

func parseConfig(data []byte) (config, error) {
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	c := &cfg.Text
	if c.HiddenSize <= 0 || c.Streams <= 0 || c.Layers <= 0 || c.Context <= 0 ||
		c.RMSEps <= 0 || c.HCEps <= 0 || c.HCMagnitude <= 0 {
		return cfg, fmt.Errorf("invalid Hy4 Flash model dimensions or normalization configuration")
	}
	if len(c.AttentionTypes) != c.Layers || len(c.MLPTypes) != c.Layers {
		return cfg, fmt.Errorf("Hy4 Flash layer schedules must contain %d layers", c.Layers)
	}
	for i, attentionType := range c.AttentionTypes {
		if attentionType != indexedAttention && attentionType != slidingAttention {
			return cfg, fmt.Errorf("layer %d: unsupported attention type %q", i, attentionType)
		}
		if c.MLPTypes[i] != "dense" && c.MLPTypes[i] != "sparse" {
			return cfg, fmt.Errorf("layer %d: unsupported MLP type %q", i, c.MLPTypes[i])
		}
	}
	for _, name := range []string{indexedAttention, slidingAttention} {
		if rope := c.Rope[name]; rope.Theta <= 0 || rope.Type != "default" {
			return cfg, fmt.Errorf("unsupported Hy4 Flash RoPE for %s", name)
		}
	}
	c.Indexed = attentionShape{c.Heads, c.NoPE, c.RoPE, c.Value, c.Rope[indexedAttention].Theta, c.Rope[indexedAttention].Sections}
	c.Sliding = attentionShape{c.SlidingHeads, c.SlidingNoPE, c.SlidingRoPE, c.SlidingValue, c.Rope[slidingAttention].Theta, c.Rope[slidingAttention].Sections}
	for _, shape := range []attentionShape{c.Indexed, c.Sliding} {
		if shape.heads <= 0 || shape.plain <= 0 || shape.rotary <= 0 || shape.rotary%2 != 0 || shape.value <= 0 {
			return cfg, fmt.Errorf("invalid Hy4 Flash attention shape")
		}
	}
	if c.QRank <= 0 || c.KVRank <= 0 || c.Window <= 0 || c.IndexRatio <= 0 ||
		c.IndexTopK <= 1 || c.IndexHeads <= 0 || c.IndexDim < c.RoPE {
		return cfg, fmt.Errorf("invalid Hy4 Flash KV or index configuration")
	}
	if c.ExpertsPerTok <= 0 || c.ExpertsPerTok > c.Experts || c.ExpertWidth <= 0 ||
		c.LatentWidth <= 0 || c.RouterGroups != 1 || c.RouterGroupsK != 1 {
		return cfg, fmt.Errorf("unsupported Hy4 Flash routing configuration")
	}
	if len(c.EngramLayers) > 0 && (c.EngramOrder < 2 || c.EngramHeads <= 0 || c.EngramWidth <= 0 ||
		c.EngramWidth%c.EngramHeads != 0 || c.EngramPadID < 0) {
		return cfg, fmt.Errorf("invalid Hy4 Flash Engram configuration")
	}
	if cfg.Vision != nil {
		if err := cfg.Vision.validate(c.HiddenSize); err != nil {
			return cfg, err
		}
	}
	return cfg, nil
}
