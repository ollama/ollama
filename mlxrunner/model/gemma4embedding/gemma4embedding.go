// Package gemma4embedding implements the Gemma4Embedding (EmbeddingGemma v2)
// text embedding model for MLX.
//
// Unlike the generative Gemma 4, every layer attends bidirectionally:
// full_attention layers are padding-mask-only and sliding_attention layers
// use a symmetric ±window/2 band. There is no KV cache and no lm_head —
// the output head is an in-model embedding_projection applied per token.
// Pooling and L2 normalization happen in the runner pipeline, matching
// sentence-transformers' 1_Pooling + 2_Normalize modules.
package gemma4embedding

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/gemma4"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("EmbeddingGemma2Model", newModel)
}

var _ model.Model = (*Model)(nil)

// Attention scaling is 1.0: q/k RMS norms handle magnitude control.

// RopeParams holds per-layer-type RoPE settings.
type RopeParams struct {
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeTheta           float32 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
}

// PerLayerConfig is the per-layer-type override table keyed by layer index
// as a string ("05", "11", ...) in config.json.
type PerLayerConfig struct {
	HeadDim          int32 `json:"head_dim"`
	NumKeyValueHeads int32 `json:"num_key_value_heads"`
}

// Config is the Gemma4Embedding config; the text stack lives under
// text_config.
type Config struct {
	Architectures       []string `json:"architectures"`
	EmbeddingDimensions []int    `json:"embedding_dimensions"`

	TextConfig `json:"text_config"`

	// Multimodal subtrees. Every embeddinggemma-2 checkpoint carries both
	// blocks regardless of which towers the safetensors blob actually
	// contains, so tensor presence — not config presence — decides whether
	// a tower is loaded.
	VisionConfig *gemma4.VisionConfig `json:"vision_config"`
	AudioConfig  *gemma4.AudioConfig  `json:"audio_config"`

	// Media token IDs and the per-image soft-token budget the embed
	// pipeline splices into the token stream. eoa falls back to the
	// eoa_token_index spelling some checkpoints use (same as gemma4).
	ImageTokenID             int32 `json:"image_token_id"`
	AudioTokenID             int32 `json:"audio_token_id"`
	BOITokenID               int32 `json:"boi_token_id"`
	EOITokenID               int32 `json:"eoi_token_id"`
	BOATokenID               int32 `json:"boa_token_id"`
	EOATokenID               int32 `json:"eoa_token_id"`
	EOATokenIndex            int32 `json:"eoa_token_index"`
	VisionSoftTokensPerImage int32 `json:"vision_soft_tokens_per_image"`

	// Token strings for the bracket markers and soft placeholders; filled
	// in parseConfig so the runner can expand user text that names one.
	TokenStrings MediaTokenStrings `json:"-"`

	// Quantization parameters (populated at load).
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	EmbedScale      float32 `json:"-"` // sqrt(hidden_size)
	SlidingRopeBase float32 `json:"-"`
	FullRopeBase    float32 `json:"-"`
	FullRopeDims    int     `json:"-"` // rotary rank for full layers (head_dim of full layers)
}

// MediaTokenStrings are the tokenizer spellings of the media marker and
// placeholder tokens; the runner expands these in user content before
// tokenizing.
type MediaTokenStrings struct {
	BOI, EOI, Image string
	BOA, EOA, Audio string
}

type TextConfig struct {
	HiddenSize                int32                      `json:"hidden_size"`
	NumHiddenLayers           int32                      `json:"num_hidden_layers"`
	IntermediateSize          int32                      `json:"intermediate_size"`
	NumAttentionHeads         int32                      `json:"num_attention_heads"`
	NumKeyValueHeads          int32                      `json:"num_key_value_heads"`
	HeadDim                   int32                      `json:"head_dim"`
	VocabSize                 int32                      `json:"vocab_size"`
	RMSNormEps                float32                    `json:"rms_norm_eps"`
	MaxPositionEmbeddings     int32                      `json:"max_position_embeddings"`
	SlidingWindow             int32                      `json:"sliding_window"`
	LayerTypes                []string                   `json:"layer_types"`
	HiddenSizePerLayer        int32                      `json:"hidden_size_per_layer_input"`
	VocabSizePerLayer         int32                      `json:"vocab_size_per_layer_input"`
	EmbeddingDim              int32                      `json:"embedding_dim"`
	RopeParameters            map[string]*RopeParams     `json:"rope_parameters"`
	PerLayerConfig            map[string]*PerLayerConfig `json:"per_layer_config"`
	UseBidirectionalAttention string                     `json:"use_bidirectional_attention"`

	// Computed.
	PerLayerHeadDim map[int32]int32 `json:"-"` // layer -> head_dim override
	PerLayerKVHeads map[int32]int32 `json:"-"` // layer -> num_key_value_heads override
}

func parseConfig(configData []byte) (*Config, error) {
	var cfg Config
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	tc := &cfg.TextConfig
	if tc.HeadDim == 0 {
		tc.HeadDim = 256
	}
	if tc.NumAttentionHeads == 0 {
		tc.NumAttentionHeads = 4
	}
	if tc.NumKeyValueHeads == 0 {
		tc.NumKeyValueHeads = 1
	}
	if tc.RMSNormEps == 0 {
		tc.RMSNormEps = 1e-6
	}

	// vocab_size_per_layer_input is the token-identity half of PLE. We only
	// implement the projection-only half (used when it's 0); HF combines both
	// as (proj + identity) * 2^-0.5. Fail loudly for checkpoints that ship a
	// non-zero value rather than emitting silently-wrong embeddings.
	if tc.VocabSizePerLayer > 0 {
		return nil, fmt.Errorf("vocab_size_per_layer_input > 0 (token-identity PLE) is not implemented; got %d", tc.VocabSizePerLayer)
	}

	// VisionConfig.RopeTheta is json:"-" (derived in gemma4's
	// parseMultimodalConfig, which gemma4embedding does not call), so
	// rebuild it here the same way: base 100, checkpoint override wins.
	// The VisionTower.Encode RoPE tables consume it; leaving it zero would
	// silently corrupt vision embeddings. Zero-field defaults for the
	// remaining vision/audio config fields are NOT mirrored: the shipped
	// embeddinggemma-2 configs populate them all.
	if v := cfg.VisionConfig; v != nil && v.ModelType == "gemma4_vision" {
		if v.HeadDim == 0 && v.NumAttentionHeads > 0 {
			v.HeadDim = v.HiddenSize / v.NumAttentionHeads
		}
		v.RopeTheta = 100
		if v.RopeParameters != nil && v.RopeParameters.RopeTheta > 0 {
			v.RopeTheta = v.RopeParameters.RopeTheta
		}
	}

	// The runner substitutes friendly spellings for the raw special-token
	// text; both exist in embeddinggemma-2's vocab, and sentence-transformers
	// accepts either (its prompt template emits the bracketed forms, while
	// users typing into a curl payload reach the angle forms more easily).
	cfg.TokenStrings = MediaTokenStrings{
		BOI: "<|image>", EOI: "<image|>", Image: "<|image|>",
		BOA: "<|audio>", EOA: "<audio|>", Audio: "<|audio|>",
	}
	if cfg.EOATokenID == 0 {
		cfg.EOATokenID = cfg.EOATokenIndex
	}

	cfg.EmbedScale = float32(math.Sqrt(float64(tc.HiddenSize)))
	cfg.SlidingRopeBase = 10000
	cfg.FullRopeBase = 1000000
	cfg.FullRopeDims = int(tc.HeadDim) // default; overridden per-layer in forward

	// Per-layer overrides first: RoPE dims for full layers key off them.
	tc.PerLayerHeadDim = make(map[int32]int32)
	tc.PerLayerKVHeads = make(map[int32]int32)
	for k, pl := range tc.PerLayerConfig {
		if pl == nil {
			continue
		}
		var idx int
		if _, err := fmt.Sscanf(k, "%d", &idx); err != nil {
			continue
		}
		if pl.HeadDim > 0 {
			tc.PerLayerHeadDim[int32(idx)] = pl.HeadDim
		}
		if pl.NumKeyValueHeads > 0 {
			tc.PerLayerKVHeads[int32(idx)] = pl.NumKeyValueHeads
		}
	}

	for lt, rp := range tc.RopeParameters {
		if rp.RopeType != "" && rp.RopeType != "default" {
			return nil, fmt.Errorf("unsupported rope_type %q for %s", rp.RopeType, lt)
		}
	}

	if rp := tc.RopeParameters; rp != nil {
		if sp := rp["sliding_attention"]; sp != nil && sp.RopeTheta > 0 {
			cfg.SlidingRopeBase = sp.RopeTheta
		}
		if fp := rp["full_attention"]; fp != nil && fp.RopeTheta > 0 {
			cfg.FullRopeBase = fp.RopeTheta
		}
	}
	cfg.FullRopeDims = int(tc.fullHeadDim())

	return &cfg, nil
}

// fullHeadDim is the head_dim of full_attention layers, per the override
// table (any full layer with an override; they all share one value in the
// shipped configs).
func (tc *TextConfig) fullHeadDim() int32 {
	for i := range tc.NumHiddenLayers {
		if tc.IsSliding(i) {
			continue
		}
		if d, ok := tc.PerLayerHeadDim[i]; ok {
			return d
		}
	}
	return tc.HeadDim
}

func (tc *TextConfig) IsSliding(layer int32) bool {
	if int(layer) < len(tc.LayerTypes) {
		return tc.LayerTypes[layer] == "sliding_attention"
	}
	return false
}

func (tc *TextConfig) layerHeadDim(layer int32) int32 {
	if d, ok := tc.PerLayerHeadDim[layer]; ok {
		return d
	}
	return tc.HeadDim
}

func (tc *TextConfig) layerKVHeads(layer int32) int32 {
	if h, ok := tc.PerLayerKVHeads[layer]; ok {
		return h
	}
	return tc.NumKeyValueHeads
}

// Attention is the per-layer attention block with Q/K norms and a
// weightless V norm.
type Attention struct {
	QProj, KProj, VProj, OProj nn.LinearLayer
	QNorm, KNorm               *nn.RMSNorm
	QNormScaled, KNormScaled   *mlx.Array
}

// MLP is the GELU-tanh feed-forward block.
type MLP struct {
	GateProj, UpProj, DownProj nn.LinearLayer
}

// PLELayer holds the per-layer PLE (per-layer embedding) weights.
type PLELayer struct {
	InputGate      nn.LinearLayer
	Projection     nn.LinearLayer
	PostNorm       *nn.RMSNorm
	PostNormScaled *mlx.Array
}

// DecoderLayer is a single transformer block.
type DecoderLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm *nn.RMSNorm
	Attention                                      *Attention
	MLP                                            *MLP
	PLE                                            *PLELayer

	InputNormScaled, PostAttnNormScaled *mlx.Array
	PreFFNormScaled, PostFFNormScaled   *mlx.Array

	// LayerScalar multiplies the layer output (trained, e.g. 0.39/0.17 in
	// the embeddinggemma-2 checkpoint — not all-ones).
	LayerScalar *mlx.Array

	IsSliding bool
	LayerIdx  int32
}

// Model is the Gemma4Embedding text tower + embedding projection, with
// optional vision/audio towers for the multimodal checkpoints.
type Model struct {
	EmbedTokens         nn.EmbeddingLayer
	Layers              []*DecoderLayer
	Norm                *nn.RMSNorm
	EmbeddingProjection nn.LinearLayer

	// Multimodal towers; nil unless the checkpoint carries their weights.
	// The embedders hold the tower-to-text projection and are applied
	// inside the towers' Encode methods.
	VisionTower *gemma4.VisionTower
	AudioTower  *gemma4.AudioTower
	EmbedVision *gemma4.MultimodalEmbedder
	EmbedAudio  *gemma4.MultimodalEmbedder

	// Projection-only PLE (no per-layer embedding table when
	// vocab_size_per_layer_input == 0).
	PerLayerModelProj nn.LinearLayer
	PerLayerProjNorm  *nn.RMSNorm

	NormScaled             *mlx.Array
	PerLayerProjNormWeight *mlx.Array

	tok *tokenizer.Tokenizer

	// Cfg must stay exported: mlx.Collect walks only exported fields.
	Cfg *Config
}

func newModel(root *model.Root) (model.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			cfg.QuantGroupSize = gs
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer config: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if tokConfigData, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = tokConfigData
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	return &Model{
		Layers: make([]*DecoderLayer, cfg.NumHiddenLayers),
		Cfg:    cfg,
		tok:    tok,
	}, nil
}

// resolveWeightPrefix finds the language_model prefix used by the safetensors
// blob (with or without the "model." root that the HF class nesting adds).
func resolveWeightPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{"language_model.", "model.language_model.", ""} {
		if tensors[prefix+"embed_tokens.weight"] != nil {
			return prefix
		}
	}
	return ""
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := resolveWeightPrefix(tensors)
	tc := &m.Cfg.TextConfig
	linears := model.NewLinearFactory(tensors, m.Cfg.QuantGroupSize, m.Cfg.QuantBits, m.Cfg.QuantMode, m.Cfg.TensorQuant)

	embedTokens := model.MakeEmbeddingLayer(tensors, prefix+"embed_tokens", m.Cfg.QuantGroupSize, m.Cfg.QuantBits, m.Cfg.QuantMode, m.Cfg.TensorQuant)
	if embedTokens == nil {
		return fmt.Errorf("missing embedding weight: %sembed_tokens.weight", prefix)
	}
	m.EmbedTokens = embedTokens

	if w := tensors[prefix+"norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, tc.RMSNormEps)
	} else {
		return fmt.Errorf("missing final norm weight: %snorm.weight", prefix)
	}

	// embedding_projection may sit alongside the language model (new
	// checkpoint: language_model.embedding_projection) or one level above
	// (older spelling: model.embedding_projection). Try both.
	for _, cand := range []string{prefix + "embedding_projection", stringSuffix(prefix, "language_model.") + "embedding_projection"} {
		if p := linears.Make(cand); p != nil {
			m.EmbeddingProjection = p
			break
		}
	}
	if m.EmbeddingProjection == nil {
		return fmt.Errorf("missing embedding_projection weight")
	}

	// Projection-only PLE. Current checkpoints nest the shared projection
	// under .ple.*; older spellings kept it at .per_layer_*.
	if tc.HiddenSizePerLayer > 0 {
		m.PerLayerModelProj = linears.Make(prefix + "ple.per_layer_model_projection")
		if m.PerLayerModelProj == nil {
			m.PerLayerModelProj = linears.Make(prefix + "per_layer_model_projection")
		}
		if m.PerLayerModelProj == nil {
			return fmt.Errorf("missing per_layer_model_projection weight")
		}
		if w := tensors[prefix+"ple.per_layer_projection_norm.weight"]; w != nil {
			m.PerLayerProjNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		} else if w := tensors[prefix+"per_layer_projection_norm.weight"]; w != nil {
			m.PerLayerProjNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		} else {
			return fmt.Errorf("missing per_layer_projection_norm weight")
		}
	}

	for i := range tc.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		layer := &DecoderLayer{
			LayerIdx:  i,
			IsSliding: tc.IsSliding(i),
			Attention: &Attention{},
			MLP:       &MLP{},
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}
		if w := tensors[layerPrefix+".pre_feedforward_layernorm.weight"]; w != nil {
			layer.PreFFNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}

		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
		}

		layer.MLP.GateProj = linears.Make(layerPrefix + ".mlp.gate_proj")
		layer.MLP.UpProj = linears.Make(layerPrefix + ".mlp.up_proj")
		layer.MLP.DownProj = linears.Make(layerPrefix + ".mlp.down_proj")

		if w := tensors[layerPrefix+".layer_scalar"]; w != nil {
			layer.LayerScalar = w
		} else {
			return fmt.Errorf("layer %d: missing layer_scalar", i)
		}

		if tc.HiddenSizePerLayer > 0 {
			// Current checkpoints nest PLE under .ple_block.*; older
			// spellings kept them flat at .per_layer_*.
			layer.PLE = &PLELayer{
				InputGate:  linears.Make(layerPrefix + ".ple_block.per_layer_input_gate"),
				Projection: linears.Make(layerPrefix + ".ple_block.per_layer_projection"),
			}
			if layer.PLE.InputGate == nil {
				layer.PLE.InputGate = linears.Make(layerPrefix + ".per_layer_input_gate")
			}
			if layer.PLE.Projection == nil {
				layer.PLE.Projection = linears.Make(layerPrefix + ".per_layer_projection")
			}
			if w := tensors[layerPrefix+".ple_block.post_per_layer_input_norm.weight"]; w != nil {
				layer.PLE.PostNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
			} else if w := tensors[layerPrefix+".post_per_layer_input_norm.weight"]; w != nil {
				layer.PLE.PostNorm = nn.NewRMSNorm(w, tc.RMSNormEps)
			}
			if layer.PLE.InputGate == nil || layer.PLE.Projection == nil || layer.PLE.PostNorm == nil {
				return fmt.Errorf("layer %d: missing PLE weights", i)
			}
		}

		if layer.InputNorm == nil || layer.PostAttnNorm == nil || layer.PreFFNorm == nil || layer.PostFFNorm == nil {
			return fmt.Errorf("layer %d: missing norms", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("layer %d: missing q/k norms", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing mlp projections", i)
		}

		m.Layers[i] = layer
	}

	if err := m.loadMediaTowers(tensors, linears); err != nil {
		return err
	}

	m.precomputeScaledWeights()
	return nil
}

// hasTensorsWithPrefix reports whether the tensor map carries any key under
// a (possibly model.-rooted) subtree, e.g. "vision_tower.". Prefix scan
// rather than a single sentinel because the caller needn't know which of
// the root spellings the blob uses.
func hasTensorsWithPrefix(tensors map[string]*mlx.Array, subtrees ...string) bool {
	for k, v := range tensors {
		if v == nil {
			continue
		}
		for _, sub := range subtrees {
			if strings.HasPrefix(k, sub) || strings.HasPrefix(k, "model."+sub) {
				return true
			}
		}
	}
	return false
}

// loadMediaTowers loads the optional vision and audio towers. The tower
// loaders re-derive the blob's root prefix themselves; here we only decide
// presence. A tower's weights with a missing config subtree is a hard
// error — embeddinggemma-2 checkpoints always ship both subtrees, so this
// can only trip on a hand-built manifest.
func (m *Model) loadMediaTowers(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	if hasTensorsWithPrefix(tensors, "vision_tower.", "embed_vision.") {
		if m.Cfg.VisionConfig == nil {
			return fmt.Errorf("checkpoint carries vision_tower weights but config.json has no vision_config")
		}
		tower, proj, err := gemma4.LoadVisionTower(tensors, m.Cfg.VisionConfig, linears)
		if err != nil {
			return fmt.Errorf("load vision tower: %w", err)
		}
		m.VisionTower, m.EmbedVision = tower, proj
	}
	if hasTensorsWithPrefix(tensors, "audio_tower.", "embed_audio.") {
		if m.Cfg.AudioConfig == nil {
			return fmt.Errorf("checkpoint carries audio_tower weights but config.json has no audio_config")
		}
		tower, proj, err := gemma4.LoadAudioTower(tensors, m.Cfg.AudioConfig, linears)
		if err != nil {
			return fmt.Errorf("load audio tower: %w", err)
		}
		m.AudioTower, m.EmbedAudio = tower, proj
	}
	return nil
}

func stringSuffix(s, suffix string) string {
	if len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix {
		return s[:len(s)-len(suffix)]
	}
	return s
}

// precomputeScaledWeights assigns raw norm weights to the scaled fields
// (Gemma 4 norms use scale_shift=0, so the raw weight is the scale).
func (m *Model) precomputeScaledWeights() {
	m.NormScaled = m.Norm.Weight
	if m.PerLayerProjNorm != nil {
		m.PerLayerProjNormWeight = m.PerLayerProjNorm.Weight
	}
	for _, l := range m.Layers {
		l.InputNormScaled = l.InputNorm.Weight
		l.PostAttnNormScaled = l.PostAttnNorm.Weight
		l.PreFFNormScaled = l.PreFFNorm.Weight
		l.PostFFNormScaled = l.PostFFNorm.Weight
		l.Attention.QNormScaled = l.Attention.QNorm.Weight
		l.Attention.KNormScaled = l.Attention.KNorm.Weight
		if l.PLE != nil {
			l.PLE.PostNormScaled = l.PLE.PostNorm.Weight
		}
	}
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// maxContextLength is the vendor-validated context window. The checkpoint
// config declares 262144 max_position_embeddings because the RoPE math
// machinery is sound out to that span (it's the Gemma 4 backbone's default),
// but the model itself is trained and evaluated at 8K. The 8192 cap gates
// the runner's pre-flight budget and the server's truncation behavior.
// Override via options.num_ctx for longer spans at the operator's discretion.
const maxContextLength = 8192

// MaxContextLength returns the model's validated context window. The
// checkpoint's max_position_embeddings (262144) is a Gemma 4 backbone
// artifact — see maxContextLength.
func (m *Model) MaxContextLength() int { return maxContextLength }

// NewCaches returns nil slots per layer: embedding forward is a single
// cache-free bidirectional pass.
func (m *Model) NewCaches() []cache.Cache {
	return make([]cache.Cache, m.Cfg.NumHiddenLayers)
}

// EmbeddingDim returns the output embedding size (after projection).
func (m *Model) EmbeddingDim() int { return int(m.Cfg.TextConfig.EmbeddingDim) }

// HiddenSize reports the text stack's hidden width for allocation estimates.
func (m *Model) HiddenSize() int32 { return m.Cfg.TextConfig.HiddenSize }

// MediaTokenStrings returns the special-token spellings the runner expands
// in user content when media is present.
func (m *Model) MediaTokenStrings() (boi, eoi, image, boa, eoa, audio string) {
	t := m.Cfg.TokenStrings
	return t.BOI, t.EOI, t.Image, t.BOA, t.EOA, t.Audio
}

// SupportsImages / SupportsAudio report tower presence; the runner 400s a
// media request against a text-only checkpoint.
func (m *Model) SupportsImages() bool { return m.VisionTower != nil }
func (m *Model) SupportsAudio() bool  { return m.AudioTower != nil }

// VisionSoftTokenBudget resolves the per-image budget (request override
// wins, else the config default) and validates it against the processor's
// supported set — so a checkpoint declaring an odd budget fails at startup.
func (m *Model) VisionSoftTokenBudget() (int32, error) {
	if m.Cfg.VisionConfig == nil {
		return 0, errors.New("no vision_config")
	}
	budget := gemma4.VisionSoftTokenBudget(m.Cfg.VisionSoftTokensPerImage, m.Cfg.VisionConfig)
	if err := gemma4.ValidateVisionSoftTokenBudget(budget); err != nil {
		return 0, err
	}
	return budget, nil
}

// PrepareMedia implements a narrow slice of model.MediaModel for the embed
// pipeline: caller-supplied segment slices (tokenized text between media),
// expanded to the full token stream with boi/eoi and boa/eoa brackets and
// soft-token placeholder runs.
func (m *Model) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	c := m.Cfg
	prepared := &model.PreparedRequest{}
	for s, seg := range segments {
		switch seg.Kind {
		case "":
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
		case "image":
			budget, err := m.VisionSoftTokenBudget()
			if err != nil {
				return nil, err
			}
			// ProcessImage runs here (CPU) rather than on the MLX thread;
			// the tower call in Forward consumes the patch grid.
			pixels, positions, geom, err := gemma4.ProcessImage(seg.Data, c.VisionConfig, budget)
			if err != nil {
				return nil, err
			}
			prepared.Tokens = append(prepared.Tokens, c.BOITokenID)
			start := len(prepared.Tokens)
			for range geom.NumSoftTokens {
				prepared.Tokens = append(prepared.Tokens, c.ImageTokenID)
			}
			prepared.Tokens = append(prepared.Tokens, c.EOITokenID)
			n := int(geom.PatchesH * geom.PatchesW)
			prepared.Items = append(prepared.Items, model.PreparedItem{
				Range:     [2]int{start, len(prepared.Tokens)},
				Source:    s,
				MediaData: pixels,
				Dims:      []int{n, len(pixels) / n},
				Opaque:    preparedImage{positions: positions, geom: geom},
			})
		case "audio":
			chunks, err := gemma4.ProcessAudio(seg.Data)
			if err != nil {
				return nil, err
			}
			prepared.Tokens = append(prepared.Tokens, c.BOATokenID)
			for _, chunk := range chunks {
				start := len(prepared.Tokens)
				for range chunk.NumTokens {
					prepared.Tokens = append(prepared.Tokens, c.AudioTokenID)
				}
				end := len(prepared.Tokens)
				prepared.Items = append(prepared.Items, model.PreparedItem{
					Range:     [2]int{start, end},
					Source:    s,
					MediaData: chunk.Data,
					Dims:      []int{chunk.Frames, len(chunk.Data) / chunk.Frames},
					Opaque:    preparedAudio{numTokens: int32(chunk.NumTokens)},
				})
			}
			prepared.Tokens = append(prepared.Tokens, c.EOATokenID)
		default:
			return nil, fmt.Errorf("gemma4embedding does not support %s input", seg.Kind)
		}
	}
	return prepared, nil
}

// EmbeddingDimensions returns the trained matryoshka truncation sizes from
// this checkpoint (128/256/512/768 for embeddinggemma-2), or nil if the
// checkpoint's config declares none.
func (m *Model) EmbeddingDimensions() []int {
	if len(m.Cfg.EmbeddingDimensions) > 0 {
		return m.Cfg.EmbeddingDimensions
	}
	// Not declared in the EAP config.json; this is the set the DeepMind team
	// documents for embeddinggemma-2. Future embedding arch revisions that
	// carry the key in config.json override it; other architectures surface
	// nil through the model interface and skip validation at the runner.
	if m.Cfg.TextConfig.EmbeddingDim == 768 {
		return []int{128, 256, 512, 768}
	}
	return nil
}
