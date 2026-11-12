package deepseek_v41

import (
	"encoding/json"
	"fmt"
)

type quantizationConfig struct {
	Method           string `json:"quant_method"`
	ActivationScheme string `json:"activation_scheme"`
	WeightBlockSize  []int  `json:"weight_block_size"`
	ScaleFormat      string `json:"scale_fmt"`
	ExpertDType      string `json:"expert_dtype"`
}

type ropeScaling struct {
	Type                          string  `json:"rope_type"`
	Factor                        float64 `json:"factor"`
	BetaFast                      float64 `json:"beta_fast"`
	BetaSlow                      float64 `json:"beta_slow"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
}

type visionConfig struct {
	ModelType         string   `json:"model_type"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	HiddenSize        int      `json:"hidden_size"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	IntermediateSize  int      `json:"intermediate_size"`
	PatchSize         int      `json:"patch_size"`
	RoPETheta         float64  `json:"rope_theta"`
	DownsampleRatio   int      `json:"downsample_ratio"`
	MaxImageTokens    int      `json:"max_image_tokens"`
	MinPixels         int      `json:"min_pixels"`
	MaxWHRatio        *float64 `json:"max_wh_ratio"`
}

// Config is the publisher's nested text architecture plus vision metadata.
// Keep the layer maps explicit: they determine shared KV ownership and the
// engram table layout, so a malformed import must fail before loading weights.
type Config struct {
	ImageTokenID              int          `json:"-"`
	Vision                    visionConfig `json:"-"`
	ModelType                 string       `json:"model_type"`
	VocabSize                 int          `json:"vocab_size"`
	HiddenSize                int          `json:"hidden_size"`
	MoeIntermediateSize       int          `json:"moe_intermediate_size"`
	NumHiddenLayers           int          `json:"num_hidden_layers"`
	NumAttentionHeads         int          `json:"num_attention_heads"`
	NumKeyValueHeads          int          `json:"num_key_value_heads"`
	HeadDim                   int          `json:"head_dim"`
	QKRoPEHeadDim             int          `json:"qk_rope_head_dim"`
	QLoRARank                 int          `json:"q_lora_rank"`
	OLoRARank                 int          `json:"o_lora_rank"`
	OGroups                   int          `json:"o_groups"`
	HiddenAct                 string       `json:"hidden_act"`
	SwiGLULimit               float64      `json:"swiglu_limit"`
	RMSNormEps                float64      `json:"rms_norm_eps"`
	AttentionBias             bool         `json:"attention_bias"`
	AttentionDropout          float64      `json:"attention_dropout"`
	UseCache                  bool         `json:"use_cache"`
	TieWordEmbeddings         bool         `json:"tie_word_embeddings"`
	MaxPositionEmbeddings     int          `json:"max_position_embeddings"`
	RoPETheta                 float64      `json:"rope_theta"`
	RoPEScaling               ropeScaling  `json:"rope_scaling"`
	NumRoutedExperts          int          `json:"n_routed_experts"`
	NumSharedExperts          int          `json:"n_shared_experts"`
	NumExpertsPerToken        int          `json:"num_experts_per_tok"`
	ScoringFunc               string       `json:"scoring_func"`
	TopKMethod                string       `json:"topk_method"`
	NormTopKProb              bool         `json:"norm_topk_prob"`
	RoutedScalingFactor       float64      `json:"routed_scaling_factor"`
	SlidingWindow             int          `json:"sliding_window"`
	CompressRatios            []int        `json:"compress_ratios"`
	CompressRoPETheta         float64      `json:"compress_rope_theta"`
	KVSourceLayerIDs          []int        `json:"kv_source_layer_ids"`
	IndexSourceLayerIDs       []int        `json:"index_source_layer_ids"`
	IndexNumHeads             int          `json:"index_n_heads"`
	IndexHeadDim              int          `json:"index_head_dim"`
	IndexTopK                 int          `json:"index_topk"`
	CandidateSourceLayerID    int          `json:"candidate_source_layer_id"`
	CandidateTopKBlocks       int          `json:"candidate_topk_blocks"`
	CandidateBlockSize        int          `json:"candidate_block_size"`
	HCMult                    int          `json:"hc_mult"`
	HCSinkhornIters           int          `json:"hc_sinkhorn_iters"`
	HCEps                     float64      `json:"hc_eps"`
	EngramLayerIDs            []int        `json:"engram_layer_ids"`
	EngramNumEmbeddings       []int64      `json:"engram_num_embeddings"`
	EngramMaxNGramSize        int          `json:"engram_max_ngram_size"`
	EngramVocabSize           int64        `json:"engram_vocab_size"`
	EngramNumHeads            int          `json:"engram_n_heads"`
	EngramHeadDim             int          `json:"engram_head_dim"`
	EngramPadTokenID          int          `json:"engram_pad_token_id"`
	EngramCompressedVocabSize int          `json:"engram_compressed_vocab_size"`
	NumNextNPredictLayers     int          `json:"num_nextn_predict_layers"`
	DSparkBlockSize           int          `json:"dspark_block_size"`
	DSparkNoiseTokenID        int          `json:"dspark_noise_token_id"`
	DSparkTargetLayerIDs      []int        `json:"dspark_target_layer_ids"`
	DSparkMarkovRank          int          `json:"dspark_markov_rank"`
	DSparkNumRoutedExperts    int          `json:"dspark_n_routed_experts"`
	DSparkNumExpertsPerToken  int          `json:"dspark_num_experts_per_tok"`
}

func parseConfig(data []byte) (Config, error) {
	var envelope struct {
		Architectures []string           `json:"architectures"`
		ModelType     string             `json:"model_type"`
		Quantization  quantizationConfig `json:"quantization_config"`
		Text          json.RawMessage    `json:"text_config"`
		Vision        *visionConfig      `json:"vision_config"`
		ImageTokenID  int                `json:"image_token_id"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	if len(envelope.Architectures) != 1 || envelope.Architectures[0] != "DeepseekV41ForCausalLM" || envelope.ModelType != "deepseek_v41" {
		return Config{}, fmt.Errorf("unsupported DeepSeek V4.1 architecture: %v, %q", envelope.Architectures, envelope.ModelType)
	}
	q := envelope.Quantization
	if q.Method != "fp8" || q.ActivationScheme != "dynamic" || len(q.WeightBlockSize) != 2 || q.WeightBlockSize[0] != 32 || q.WeightBlockSize[1] != 32 || q.ScaleFormat != "ue8m0" || q.ExpertDType != "fp4" {
		return Config{}, fmt.Errorf("unsupported DeepSeek V4.1 quantization: %+v", q)
	}
	if len(envelope.Text) == 0 {
		return Config{}, fmt.Errorf("missing text_config")
	}
	var cfg Config
	if err := json.Unmarshal(envelope.Text, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse text_config: %w", err)
	}
	if envelope.Vision != nil {
		cfg.Vision = *envelope.Vision
		if cfg.Vision.ModelType == "" {
			return Config{}, fmt.Errorf("unsupported DeepSeek V4.1 vision configuration")
		}
	}
	cfg.ImageTokenID = envelope.ImageTokenID
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// ValidateConfig checks whether the DeepSeek V4.1 runner supports the source
// config before a model import advertises its capabilities.
func ValidateConfig(data []byte) error {
	_, err := parseConfig(data)
	return err
}

func (cfg *Config) validate() error {
	if cfg.Vision.ModelType != "" {
		v := cfg.Vision
		if v.ModelType != "deepseek_v41_vision" || v.NumHiddenLayers <= 0 ||
			v.HiddenSize <= 0 || v.NumAttentionHeads <= 0 ||
			v.HiddenSize%v.NumAttentionHeads != 0 ||
			(v.HiddenSize/v.NumAttentionHeads)%4 != 0 ||
			v.IntermediateSize <= 0 || v.PatchSize <= 0 || v.RoPETheta <= 0 ||
			v.DownsampleRatio <= 0 || v.MaxImageTokens <= 2 || v.MinPixels <= 0 ||
			cfg.ImageTokenID < 0 || cfg.ImageTokenID >= cfg.VocabSize || v.MaxWHRatio != nil {
			return fmt.Errorf("unsupported DeepSeek V4.1 vision configuration")
		}
	}
	if cfg.ModelType != "deepseek_v41_text" || cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.VocabSize <= 0 {
		return fmt.Errorf("invalid text architecture or dimensions: %q, hidden=%d, layers=%d, vocab=%d", cfg.ModelType, cfg.HiddenSize, cfg.NumHiddenLayers, cfg.VocabSize)
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads != 1 || cfg.HeadDim <= 0 || cfg.QKRoPEHeadDim <= 0 || cfg.QKRoPEHeadDim > cfg.HeadDim || cfg.QKRoPEHeadDim%2 != 0 || cfg.OGroups <= 0 || cfg.NumAttentionHeads%cfg.OGroups != 0 {
		return fmt.Errorf("invalid attention dimensions")
	}
	if cfg.QLoRARank <= 0 || cfg.OLoRARank <= 0 || cfg.OGroups <= 0 || cfg.RMSNormEps <= 0 || cfg.RoPETheta <= 0 || cfg.CompressRoPETheta <= 0 || cfg.SlidingWindow <= 0 {
		return fmt.Errorf("invalid attention ranks, normalization, or window")
	}
	if cfg.HiddenAct != "silu" || cfg.SwiGLULimit <= 0 || cfg.AttentionBias || cfg.AttentionDropout != 0 || !cfg.UseCache || cfg.TieWordEmbeddings {
		return fmt.Errorf("unsupported attention, activation, or embedding contract")
	}
	if cfg.RoPEScaling.Type != "yarn" || cfg.RoPEScaling.Factor <= 0 || cfg.RoPEScaling.OriginalMaxPositionEmbeddings <= 0 || cfg.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("unsupported RoPE configuration")
	}
	if cfg.NumRoutedExperts <= 0 || cfg.NumExpertsPerToken <= 0 || cfg.NumExpertsPerToken > cfg.NumRoutedExperts || cfg.NumSharedExperts != 1 || cfg.MoeIntermediateSize <= 0 || cfg.ScoringFunc != "sqrtsoftplus" || cfg.TopKMethod != "noaux_tc" || !cfg.NormTopKProb || cfg.RoutedScalingFactor <= 0 {
		return fmt.Errorf("unsupported MoE configuration")
	}
	if cfg.HCMult <= 0 || cfg.HCSinkhornIters <= 0 || cfg.HCEps <= 0 {
		return fmt.Errorf("invalid hyper-connection configuration")
	}
	if cfg.NumNextNPredictLayers < 0 || len(cfg.CompressRatios) != cfg.NumHiddenLayers+cfg.NumNextNPredictLayers {
		return fmt.Errorf("compress_ratios has %d entries, want %d", len(cfg.CompressRatios), cfg.NumHiddenLayers+cfg.NumNextNPredictLayers)
	}
	for i, ratio := range cfg.CompressRatios {
		if ratio < 0 || ratio > 2 {
			return fmt.Errorf("unsupported compress_ratios[%d]=%d", i, ratio)
		}
	}
	for _, group := range []struct {
		name string
		ids  []int
	}{
		{"kv_source_layer_ids", cfg.KVSourceLayerIDs},
		{"index_source_layer_ids", cfg.IndexSourceLayerIDs},
		{"engram_layer_ids", cfg.EngramLayerIDs},
	} {
		if err := validateLayerIDs(group.name, group.ids, cfg.NumHiddenLayers); err != nil {
			return err
		}
	}
	if cfg.IndexNumHeads <= 0 || cfg.IndexHeadDim <= 0 || cfg.IndexHeadDim > cfg.HeadDim || cfg.IndexTopK <= 0 || cfg.CandidateSourceLayerID < 0 || cfg.CandidateSourceLayerID >= cfg.NumHiddenLayers || cfg.CandidateTopKBlocks <= 0 || cfg.CandidateBlockSize <= 0 {
		return fmt.Errorf("invalid sparse index configuration")
	}
	indexSources := make(map[int]bool, len(cfg.IndexSourceLayerIDs))
	for _, id := range cfg.IndexSourceLayerIDs {
		indexSources[id] = true
	}
	for _, source := range cfg.KVSourceLayerIDs {
		if cfg.CompressRatios[source] == 0 || !indexSources[source] {
			return fmt.Errorf("KV source layer %d needs compression and an index source", source)
		}
	}
	if !indexSources[cfg.CandidateSourceLayerID] {
		return fmt.Errorf("candidate source layer %d needs an indexer", cfg.CandidateSourceLayerID)
	}
	for _, id := range cfg.IndexSourceLayerIDs {
		owner := -1
		for _, source := range cfg.KVSourceLayerIDs {
			if source <= id && source > owner {
				owner = source
			}
		}
		if owner < 0 || cfg.CompressRatios[id] != cfg.CompressRatios[owner] {
			return fmt.Errorf("index source layer %d has no matching KV source", id)
		}
	}
	if cfg.EngramMaxNGramSize < 2 || cfg.EngramVocabSize <= 0 || cfg.EngramNumHeads <= 0 || cfg.EngramHeadDim <= 0 || cfg.EngramCompressedVocabSize <= 0 || cfg.EngramPadTokenID < 0 || cfg.EngramPadTokenID >= cfg.VocabSize {
		return fmt.Errorf("invalid engram configuration")
	}
	if len(cfg.EngramLayerIDs) != len(cfg.EngramNumEmbeddings) || len(cfg.EngramLayerIDs) == 0 {
		return fmt.Errorf("engram layer and table counts differ")
	}
	if len(cfg.DSparkTargetLayerIDs) != cfg.NumNextNPredictLayers || cfg.DSparkBlockSize <= 0 || cfg.DSparkNumRoutedExperts <= 0 || cfg.DSparkNumExpertsPerToken <= 0 {
		return fmt.Errorf("invalid DSpark configuration")
	}
	return nil
}

func validateLayerIDs(name string, ids []int, limit int) error {
	seen := make(map[int]bool, len(ids))
	for _, id := range ids {
		if id < 0 || id >= limit || seen[id] {
			return fmt.Errorf("%s has duplicate or out-of-range layer %d", name, id)
		}
		seen[id] = true
	}
	return nil
}
