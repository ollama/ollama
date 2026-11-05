package qwen4_exp

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

type ropeParameters struct {
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	MRoPEInterleaved    bool    `json:"mrope_interleaved"`
	MRoPESection        []int32 `json:"mrope_section"`
}

type mtpConfig struct {
	Hybrid                  bool     `json:"hybrid"`
	LayerTypes              []string `json:"layer_types"`
	NumHiddenLayers         int32    `json:"num_hidden_layers"`
	UseHiddenStateFromLayer *int32   `json:"mtp_use_hidden_state_from_layer"`
	RopeTheta               float32  `json:"rope_theta"`
}

// Config is the text portion of a Qwen 4 checkpoint config.
// Keep the publisher's field names represented explicitly: this schema is an
// artifact contract and must not silently reinterpret unknown shapes as
// Qwen3.5 defaults.
type Config struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int32    `json:"hidden_size"`
	NumHiddenLayers       int32    `json:"num_hidden_layers"`
	NumAttentionHeads     int32    `json:"num_attention_heads"`
	NumKeyValueHeads      int32    `json:"num_key_value_heads"`
	HeadDim               int32    `json:"head_dim"`
	RMSNormEps            float32  `json:"rms_norm_eps"`
	VocabSize             int32    `json:"vocab_size"`
	EOSTokenID            int64    `json:"eos_token_id"`
	MaxPositionEmbeddings int32    `json:"max_position_embeddings"`
	LayerTypes            []string `json:"layer_types"`

	FullAttentionInterval int32   `json:"full_attention_interval"`
	OutputGateType        string  `json:"output_gate_type"`
	AttentionBias         bool    `json:"attention_bias"`
	AttentionDropout      float32 `json:"attention_dropout"`
	HiddenAct             string  `json:"hidden_act"`
	MambaSSMDType         string  `json:"mamba_ssm_dtype"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	UseCache              bool    `json:"use_cache"`

	LinearNumValueHeads int32 `json:"linear_num_value_heads"`
	LinearNumKeyHeads   int32 `json:"linear_num_key_heads"`
	LinearKeyHeadDim    int32 `json:"linear_key_head_dim"`
	LinearValueHeadDim  int32 `json:"linear_value_head_dim"`
	LinearConvKernelDim int32 `json:"linear_conv_kernel_dim"`

	NumExperts                   int32 `json:"num_experts"`
	NumExpertsPerTok             int32 `json:"num_experts_per_tok"`
	MoeIntermediateSize          int32 `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize int32 `json:"shared_expert_intermediate_size"`

	HCCount   int32 `json:"hc_count"`
	HCLowRank int32 `json:"hc_lowrank"`

	IndexerBudget        int32 `json:"indexer_budget"`
	IndexerCompressRatio int32 `json:"indexer_compress_ratio"`
	IndexerHeadDim       int32 `json:"indexer_head_dim"`
	IndexerKVHeads       int32 `json:"indexer_kv_heads"`
	IndexerNumHeads      int32 `json:"indexer_n_heads"`

	PLEConvKernelSize int32   `json:"ple_conv_kernel_size"`
	PLEEmbedDim       int32   `json:"ple_embed_dim"`
	PLELayerIDs       []int32 `json:"ple_layer_ids"`

	HeadsPerNGram               int32 `json:"heads_per_ngram"`
	MakeNGramVocabSizeDivisible int32 `json:"make_ngram_vocab_size_divisible_by"`
	NGramSize                   int32 `json:"ngram_size"`
	NGramVocabSizeBase          int64 `json:"ngram_vocab_size_base"`
	SplitNGramParts             int32 `json:"split_ngram_parts"`

	MTPNumHiddenLayers        int32     `json:"mtp_num_hidden_layers"`
	MTPUseDedicatedEmbeddings bool      `json:"mtp_use_dedicated_embeddings"`
	MTP                       mtpConfig `json:"mtp"`

	PartialRotaryFactor float32        `json:"partial_rotary_factor"`
	RopeParameters      ropeParameters `json:"rope_parameters"`

	Scale   float32 `json:"-"`
	RopeDim int32   `json:"-"`
}

func parseConfig(data []byte) (Config, error) {
	var envelope struct {
		TextConfig json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return Config{}, fmt.Errorf("parse config envelope: %w", err)
	}

	active := data
	if len(envelope.TextConfig) > 0 && string(envelope.TextConfig) != "null" {
		active = envelope.TextConfig
	}

	var cfg Config
	if err := json.Unmarshal(active, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse text config: %w", err)
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (cfg *Config) validate() error {
	if cfg.ModelType != "qwen4_exp_text" {
		return fmt.Errorf("unsupported model_type %q", cfg.ModelType)
	}
	if cfg.HiddenSize <= 0 || cfg.NumHiddenLayers <= 0 {
		return fmt.Errorf("invalid model dimensions: hidden_size=%d num_hidden_layers=%d", cfg.HiddenSize, cfg.NumHiddenLayers)
	}
	if cfg.EOSTokenID < 0 || cfg.EOSTokenID >= int64(cfg.VocabSize) {
		return fmt.Errorf("invalid eos_token_id %d for vocab_size %d", cfg.EOSTokenID, cfg.VocabSize)
	}
	if cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.HeadDim <= 0 {
		return fmt.Errorf("invalid attention dimensions: heads=%d kv_heads=%d head_dim=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("invalid max_position_embeddings %d", cfg.MaxPositionEmbeddings)
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return fmt.Errorf("layer_types has %d entries, want %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	for i, kind := range cfg.LayerTypes {
		switch kind {
		case "linear_attention", "full_attention":
		default:
			return fmt.Errorf("layer_types[%d] has unsupported value %q", i, kind)
		}
	}

	if cfg.LinearNumKeyHeads <= 0 || cfg.LinearNumValueHeads <= 0 || cfg.LinearKeyHeadDim <= 0 || cfg.LinearValueHeadDim <= 0 || cfg.LinearConvKernelDim <= 0 {
		return fmt.Errorf("invalid linear attention dimensions")
	}
	if cfg.LinearNumValueHeads%cfg.LinearNumKeyHeads != 0 {
		return fmt.Errorf("linear_num_value_heads (%d) must be divisible by linear_num_key_heads (%d)", cfg.LinearNumValueHeads, cfg.LinearNumKeyHeads)
	}
	if cfg.OutputGateType != "sigmoid" {
		return fmt.Errorf("unsupported output_gate_type %q", cfg.OutputGateType)
	}
	if cfg.AttentionBias || cfg.AttentionDropout != 0 {
		return fmt.Errorf("unsupported attention bias/dropout: bias=%v dropout=%v", cfg.AttentionBias, cfg.AttentionDropout)
	}
	if cfg.HiddenAct != "silu" || cfg.MambaSSMDType != "float32" {
		return fmt.Errorf("unsupported activation/recurrent dtype: hidden_act=%q mamba_ssm_dtype=%q", cfg.HiddenAct, cfg.MambaSSMDType)
	}
	if cfg.TieWordEmbeddings || !cfg.UseCache {
		return fmt.Errorf("unsupported embedding/cache contract: tied=%v use_cache=%v", cfg.TieWordEmbeddings, cfg.UseCache)
	}

	if cfg.NumExperts <= 0 || cfg.NumExpertsPerTok <= 0 || cfg.NumExpertsPerTok > cfg.NumExperts {
		return fmt.Errorf("invalid MoE dimensions: experts=%d experts_per_tok=%d", cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	if cfg.MoeIntermediateSize <= 0 || cfg.SharedExpertIntermediateSize <= 0 {
		return fmt.Errorf("invalid MoE intermediate dimensions")
	}

	if cfg.HCCount <= 0 || cfg.HCLowRank <= 0 {
		return fmt.Errorf("invalid hyper-connection dimensions: count=%d lowrank=%d", cfg.HCCount, cfg.HCLowRank)
	}
	if cfg.IndexerBudget <= 0 || cfg.IndexerCompressRatio <= 0 || cfg.IndexerHeadDim <= 0 || cfg.IndexerKVHeads <= 0 || cfg.IndexerNumHeads <= 0 {
		return fmt.Errorf("invalid sparse indexer dimensions")
	}
	if cfg.IndexerBudget < cfg.IndexerCompressRatio {
		return fmt.Errorf("indexer_budget (%d) must be at least indexer_compress_ratio (%d)", cfg.IndexerBudget, cfg.IndexerCompressRatio)
	}
	if cfg.IndexerKVHeads != 1 && cfg.IndexerKVHeads != cfg.IndexerNumHeads {
		return fmt.Errorf("indexer_kv_heads (%d) must be 1 or equal indexer_n_heads (%d)", cfg.IndexerKVHeads, cfg.IndexerNumHeads)
	}
	if cfg.PLEEmbedDim != cfg.HiddenSize || cfg.PLEConvKernelSize <= 0 || len(cfg.PLELayerIDs) == 0 {
		return fmt.Errorf("invalid PLE dimensions: embed_dim=%d conv_kernel=%d layer_ids=%v", cfg.PLEEmbedDim, cfg.PLEConvKernelSize, cfg.PLELayerIDs)
	}
	pleLayers := make(map[int32]struct{}, len(cfg.PLELayerIDs))
	for _, layer := range cfg.PLELayerIDs {
		if layer < 1 || layer > cfg.NumHiddenLayers {
			return fmt.Errorf("PLE layer ID %d is outside [1, %d]", layer, cfg.NumHiddenLayers)
		}
		if _, ok := pleLayers[layer]; ok {
			return fmt.Errorf("duplicate PLE layer ID %d", layer)
		}
		pleLayers[layer] = struct{}{}
	}
	if cfg.NGramSize <= 0 || cfg.HeadsPerNGram <= 0 || cfg.SplitNGramParts <= 0 || cfg.NGramVocabSizeBase <= 0 {
		return fmt.Errorf("invalid n-gram embedding dimensions")
	}
	if cfg.MTPNumHiddenLayers != 1 || cfg.MTP.NumHiddenLayers != 1 || len(cfg.MTP.LayerTypes) != 1 || cfg.MTP.LayerTypes[0] != "full_attention" {
		return fmt.Errorf("unsupported MTP layout")
	}
	if cfg.MTPUseDedicatedEmbeddings || !cfg.MTP.Hybrid || cfg.MTP.UseHiddenStateFromLayer != nil {
		return fmt.Errorf("unsupported MTP embedding/hidden-state contract")
	}

	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeParameters.RopeTheta <= 0 {
		return fmt.Errorf("invalid rope_theta: %v", cfg.RopeParameters.RopeTheta)
	}
	if cfg.RopeParameters.PartialRotaryFactor > 0 {
		cfg.PartialRotaryFactor = cfg.RopeParameters.PartialRotaryFactor
	}
	cfg.RopeDim = int32(float32(cfg.HeadDim) * cfg.PartialRotaryFactor)
	if cfg.RopeDim <= 0 || cfg.RopeDim > cfg.HeadDim || cfg.RopeDim > cfg.IndexerHeadDim || cfg.RopeDim%2 != 0 {
		return fmt.Errorf("invalid rotary dimension %d from head_dim=%d partial_rotary_factor=%v", cfg.RopeDim, cfg.HeadDim, cfg.PartialRotaryFactor)
	}
	if !cfg.RopeParameters.MRoPEInterleaved || len(cfg.RopeParameters.MRoPESection) != 3 {
		return fmt.Errorf("unsupported MRoPE layout: interleaved=%v sections=%v", cfg.RopeParameters.MRoPEInterleaved, cfg.RopeParameters.MRoPESection)
	}
	var ropeSections int32
	for _, section := range cfg.RopeParameters.MRoPESection {
		ropeSections += section
	}
	if ropeSections != cfg.RopeDim/2 {
		return fmt.Errorf("MRoPE sections total %d, want rotary half-dimension %d", ropeSections, cfg.RopeDim/2)
	}
	if cfg.MTP.RopeTheta != cfg.RopeParameters.RopeTheta {
		return fmt.Errorf("MTP rope_theta %v does not match text rope_theta %v", cfg.MTP.RopeTheta, cfg.RopeParameters.RopeTheta)
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	return nil
}

func (cfg *Config) layerIsLinear(layer int) bool {
	return strings.EqualFold(cfg.LayerTypes[layer], "linear_attention")
}
