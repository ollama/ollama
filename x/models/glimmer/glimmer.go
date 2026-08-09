// Package glimmer provides the Meta Glimmer text model implementation for MLX.
package glimmer

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("MuseGlimmerForConditionalGeneration", newModel)
}

var _ base.Model = (*Model)(nil)

// Config holds the decoder and vision fields from an Glimmer configuration.
type Config struct {
	HiddenSize               int32    `json:"hidden_size"`
	NumHiddenLayers          int32    `json:"num_hidden_layers"`
	IntermediateSize         int32    `json:"intermediate_size"`
	NumAttentionHeads        int32    `json:"num_attention_heads"`
	NumKeyValueHeads         int32    `json:"num_key_value_heads"`
	HeadDim                  int32    `json:"head_dim"`
	VocabSize                int32    `json:"vocab_size"`
	RMSNormEps               float32  `json:"rms_norm_eps"`
	PostNormEps              float32  `json:"post_norm_eps"`
	RopeTheta                float32  `json:"rope_theta"`
	MaxPositionEmbeddings    int32    `json:"max_position_embeddings"`
	SlidingWindow            int32    `json:"sliding_window"`
	LayerTypes               []string `json:"layer_types"`
	NoRopeLayers             []int32  `json:"no_rope_layers"`
	NormalizeTokenEmbeddings bool     `json:"normalize_tok_embeddings"`
	UseQKNorm                bool     `json:"use_qk_norm"`
	QKScaleFactor            float32  `json:"qk_scale_factor"`
	UseAttentionOutputGate   bool     `json:"use_attn_output_gate"`
	OutputMultiplier         float32  `json:"output_multiplier"`
	OutputSoftCapTemp        float32  `json:"output_soft_cap_temp"`
	TieWordEmbeddings        bool     `json:"tie_word_embeddings"`

	HasVision                   bool     `json:"has_vision"`
	PatchTokenID                int32    `json:"patch_token_id"`
	VisionAdapterDim            int32    `json:"vision_adapter_dim"`
	VisionDownsampleFactor      int32    `json:"vision_downsample_factor"`
	VisionHeads                 int32    `json:"vision_heads"`
	VisionLatentDim             int32    `json:"vision_latent_dim"`
	VisionLayers                int32    `json:"vision_layers"`
	VisionMLPRatio              float32  `json:"vision_mlp_ratio"`
	VisionOutputDim             int32    `json:"vision_output_dim"`
	VisionPatchSize             int32    `json:"vision_patch_size"`
	VisionPatchTemporal         int32    `json:"vision_patch_temporal"`
	VisionPosEmbeddingGridH     int32    `json:"vision_pos_emb_grid_h"`
	VisionPosEmbeddingGridW     int32    `json:"vision_pos_emb_grid_w"`
	VisionSparseAttentionFactor int32    `json:"vision_sparse_attention_factor"`
	VisionLayerTypes            []string `json:"-"`

	// Quantization parameters are populated from the created model metadata.
	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// Computed fields.
	AttentionScale float32 `json:"-"`
	QueryScale     float32 `json:"-"`

	NormalizeVisionEmbeddings bool `json:"-"`
	VisionRoPEInterleaved     bool `json:"-"`

	ImageStartTokenID int32 `json:"-"`
	ImageEndTokenID   int32 `json:"-"`
}

type glimmerHFTextConfig struct {
	HiddenSize            int32     `json:"hidden_size"`
	NumHiddenLayers       int32     `json:"num_hidden_layers"`
	IntermediateSize      int32     `json:"intermediate_size"`
	NumAttentionHeads     int32     `json:"num_attention_heads"`
	NumKeyValueHeads      int32     `json:"num_key_value_heads"`
	HeadDim               int32     `json:"head_dim"`
	VocabSize             int32     `json:"vocab_size"`
	RMSNormEps            float32   `json:"rms_norm_eps"`
	PostNormEps           float32   `json:"post_norm_eps"`
	MaxPositionEmbeddings int32     `json:"max_position_embeddings"`
	SlidingWindow         int32     `json:"sliding_window"`
	LayerTypes            []string  `json:"layer_types"`
	LayerRoPETheta        []float32 `json:"layer_rope_theta"`
	QKScaleFactor         float32   `json:"qk_scale_factor"`
	OutputMultiplier      float32   `json:"output_multiplier"`
	FinalLogitSoftcapping float32   `json:"final_logit_softcapping"`
	TieWordEmbeddings     bool      `json:"tie_word_embeddings"`
	RopeParameters        struct {
		RopeTheta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

type glimmerHFVisionConfig struct {
	HiddenSize         int32    `json:"hidden_size"`
	IntermediateSize   int32    `json:"intermediate_size"`
	NumAttentionHeads  int32    `json:"num_attention_heads"`
	NumHiddenLayers    int32    `json:"num_hidden_layers"`
	PatchSize          int32    `json:"patch_size"`
	PatchTemporal      int32    `json:"patch_temporal"`
	MergeSize          int32    `json:"merge_size"`
	PosEmbeddingHeight int32    `json:"pos_emb_height"`
	PosEmbeddingWidth  int32    `json:"pos_emb_width"`
	LayerTypes         []string `json:"layer_types"`
}

type glimmerHFConfig struct {
	TextConfig          *glimmerHFTextConfig   `json:"text_config"`
	VisionConfig        *glimmerHFVisionConfig `json:"vision_config"`
	ImageTokenID        int32                  `json:"image_token_id"`
	OutHiddenSize       int32                  `json:"out_hidden_size"`
	ProjectorHiddenSize int32                  `json:"projector_hidden_size"`
}

func configFromHF(hf glimmerHFConfig) Config {
	t := hf.TextConfig
	cfg := Config{
		HiddenSize:                t.HiddenSize,
		NumHiddenLayers:           t.NumHiddenLayers,
		IntermediateSize:          t.IntermediateSize,
		NumAttentionHeads:         t.NumAttentionHeads,
		NumKeyValueHeads:          t.NumKeyValueHeads,
		HeadDim:                   t.HeadDim,
		VocabSize:                 t.VocabSize,
		RMSNormEps:                t.RMSNormEps,
		PostNormEps:               t.PostNormEps,
		RopeTheta:                 t.RopeParameters.RopeTheta,
		MaxPositionEmbeddings:     t.MaxPositionEmbeddings,
		SlidingWindow:             t.SlidingWindow,
		LayerTypes:                t.LayerTypes,
		UseQKNorm:                 true,
		QKScaleFactor:             t.QKScaleFactor,
		UseAttentionOutputGate:    true,
		OutputMultiplier:          t.OutputMultiplier,
		OutputSoftCapTemp:         t.FinalLogitSoftcapping,
		TieWordEmbeddings:         t.TieWordEmbeddings,
		NormalizeVisionEmbeddings: true,
		// The official HF config omits normalize_tok_embeddings because the HF
		// checkpoint bakes that norm into the embed table (verified against the
		// publisher checkpoint: hf == rms_norm(publisher) to bf16 noise), so the
		// HF path must NOT re-normalize.
	}
	for _, theta := range t.LayerRoPETheta {
		if theta == 0 {
			cfg.NoRopeLayers = append(cfg.NoRopeLayers, 0)
		} else {
			cfg.NoRopeLayers = append(cfg.NoRopeLayers, 1)
		}
	}

	if v := hf.VisionConfig; v != nil {
		cfg.HasVision = true
		cfg.PatchTokenID = hf.ImageTokenID
		cfg.VisionAdapterDim = hf.ProjectorHiddenSize
		cfg.VisionDownsampleFactor = v.MergeSize
		cfg.VisionHeads = v.NumAttentionHeads
		cfg.VisionLatentDim = v.HiddenSize
		cfg.VisionLayers = v.NumHiddenLayers
		cfg.VisionMLPRatio = float32(v.IntermediateSize) / float32(v.HiddenSize)
		cfg.VisionOutputDim = hf.OutHiddenSize
		cfg.VisionPatchSize = v.PatchSize
		cfg.VisionPatchTemporal = v.PatchTemporal
		cfg.VisionPosEmbeddingGridH = v.PosEmbeddingHeight
		cfg.VisionPosEmbeddingGridW = v.PosEmbeddingWidth
		cfg.VisionLayerTypes = v.LayerTypes
		for i, layerType := range v.LayerTypes {
			if layerType == "full_attention" {
				cfg.VisionSparseAttentionFactor = int32(i + 1)
				break
			}
		}
	}

	return cfg
}

// Model is the Glimmer text decoder with an optional vision encoder.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	Norm        *nn.RMSNorm
	LMHead      nn.LinearLayer

	// auxHiddenLayers are the tapped layers for a DFlash draft; empty means
	// the final hidden.
	auxHiddenLayers []int

	VisionEncoder    *VisionEncoder
	VisionAdapter    *VisionAdapter
	VisionProjection nn.LinearLayer

	tok *tokenizer.Tokenizer

	*Config
}

// Layer is one Glimmer decoder block.
type Layer struct {
	Attention *Attention
	MLP       *MLP

	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm
	PostAttnNorm      *nn.RMSNorm
	PostFFNNorm       *nn.RMSNorm

	IsSliding bool
	UseRope   bool
}

// Attention implements grouped-query attention with scaleless Q/K norm and an
// output gate applied before the output projection.
type Attention struct {
	QProj          nn.LinearLayer
	KProj          nn.LinearLayer
	VProj          nn.LinearLayer
	OProj          nn.LinearLayer
	OutputGateProj nn.LinearLayer
}

// MLP is the Glimmer SwiGLU feed-forward block.
type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}

	var hf glimmerHFConfig
	if err := json.Unmarshal(data, &hf); err != nil {
		return Config{}, fmt.Errorf("parse config: %w", err)
	}
	officialHF := hf.TextConfig != nil
	if officialHF {
		cfg = configFromHF(hf)
	} else {
		cfg.NormalizeVisionEmbeddings = cfg.NormalizeTokenEmbeddings
		cfg.VisionRoPEInterleaved = true
		if cfg.HasVision {
			cfg.VisionLayerTypes = make([]string, cfg.VisionLayers)
			for i := range cfg.VisionLayerTypes {
				cfg.VisionLayerTypes[i] = "window_attention"
				if i == len(cfg.VisionLayerTypes)-1 || (cfg.VisionSparseAttentionFactor > 0 && (i+1)%int(cfg.VisionSparseAttentionFactor) == 0) {
					cfg.VisionLayerTypes[i] = "full_attention"
				}
			}
		}
	}

	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.IntermediateSize <= 0 {
		return Config{}, fmt.Errorf("invalid intermediate_size: %d", cfg.IntermediateSize)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_key_value_heads: %d", cfg.NumKeyValueHeads)
	}
	if cfg.HeadDim <= 0 {
		return Config{}, fmt.Errorf("invalid head_dim: %d", cfg.HeadDim)
	}
	if cfg.VocabSize <= 0 {
		return Config{}, fmt.Errorf("invalid vocab_size: %d", cfg.VocabSize)
	}
	if cfg.MaxPositionEmbeddings <= 0 {
		return Config{}, fmt.Errorf("invalid max_position_embeddings: %d", cfg.MaxPositionEmbeddings)
	}
	if cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return Config{}, fmt.Errorf("num_attention_heads (%d) must be divisible by num_key_value_heads (%d)", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("layer_types has %d entries, want %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	if len(cfg.NoRopeLayers) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("no_rope_layers has %d entries, want %d", len(cfg.NoRopeLayers), cfg.NumHiddenLayers)
	}
	hasSlidingLayer := false
	for i, layerType := range cfg.LayerTypes {
		if layerType != "sliding_attention" && layerType != "full_attention" {
			return Config{}, fmt.Errorf("layer_types[%d] has unsupported value %q", i, layerType)
		}
		hasSlidingLayer = hasSlidingLayer || layerType == "sliding_attention"
		if cfg.NoRopeLayers[i] != 0 && cfg.NoRopeLayers[i] != 1 {
			return Config{}, fmt.Errorf("no_rope_layers[%d] has unsupported value %d", i, cfg.NoRopeLayers[i])
		}
	}
	if hasSlidingLayer && cfg.SlidingWindow <= 0 {
		return Config{}, fmt.Errorf("invalid sliding_window: %d", cfg.SlidingWindow)
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	if cfg.PostNormEps == 0 {
		cfg.PostNormEps = 1e-8
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 500000
	}
	if cfg.OutputMultiplier == 0 {
		cfg.OutputMultiplier = 1
	}
	if cfg.QKScaleFactor == 0 {
		cfg.QKScaleFactor = 1
	}
	if cfg.HasVision {
		if cfg.PatchTokenID <= 0 {
			return Config{}, fmt.Errorf("invalid patch_token_id: %d", cfg.PatchTokenID)
		}
		if cfg.VisionAdapterDim <= 0 || cfg.VisionLatentDim <= 0 || cfg.VisionOutputDim <= 0 {
			return Config{}, fmt.Errorf("invalid vision dimensions")
		}
		if cfg.VisionHeads <= 0 || cfg.VisionLayers <= 0 {
			return Config{}, fmt.Errorf("invalid vision architecture")
		}
		if cfg.VisionLatentDim%cfg.VisionHeads != 0 {
			return Config{}, fmt.Errorf("vision_latent_dim (%d) must be divisible by vision_heads (%d)", cfg.VisionLatentDim, cfg.VisionHeads)
		}
		if cfg.VisionPatchSize <= 0 || cfg.VisionPatchTemporal <= 0 || cfg.VisionDownsampleFactor <= 0 {
			return Config{}, fmt.Errorf("invalid vision patch configuration")
		}
		if cfg.VisionPosEmbeddingGridH <= 0 || cfg.VisionPosEmbeddingGridW <= 0 {
			return Config{}, fmt.Errorf("invalid vision position embedding grid")
		}
		if cfg.VisionSparseAttentionFactor <= 0 {
			cfg.VisionSparseAttentionFactor = 1
		}
		if len(cfg.VisionLayerTypes) != int(cfg.VisionLayers) {
			return Config{}, fmt.Errorf("vision layer_types has %d entries, want %d", len(cfg.VisionLayerTypes), cfg.VisionLayers)
		}
		for i, layerType := range cfg.VisionLayerTypes {
			if layerType != "window_attention" && layerType != "full_attention" {
				return Config{}, fmt.Errorf("vision layer_types[%d] has unsupported value %q", i, layerType)
			}
		}
	}

	cfg.AttentionScale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.QueryScale = cfg.QKScaleFactor
	if !officialHF {
		cfg.QueryScale = float32(float64(cfg.QKScaleFactor) / math.Sqrt(float64(cfg.HeadDim)))
	}
	return cfg, nil
}

func newModel(root *model.Root) (base.Model, error) {
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
	if data, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokConfig.GenerationConfigJSON = data
	}
	if data, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokConfig.TokenizerConfigJSON = data
	}

	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	for _, token := range []string{"<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"} {
		if _, ok := tok.GetSpecialToken(token); !ok {
			return nil, fmt.Errorf("missing special token %q", token)
		}
	}
	if err := validateTokenizer(tok); err != nil {
		return nil, err
	}
	if cfg.HasVision {
		var ok bool
		if _, ok = tok.GetSpecialToken("<|image|>"); !ok {
			return nil, fmt.Errorf("missing image sentinel token")
		}
		if cfg.ImageStartTokenID, ok = tok.GetSpecialToken("<|image_start|>"); !ok {
			return nil, fmt.Errorf("missing image start token")
		}
		if cfg.ImageEndTokenID, ok = tok.GetSpecialToken("<|image_end|>"); !ok {
			return nil, fmt.Errorf("missing image end token")
		}
	}

	m := &Model{
		Layers: make([]*Layer, cfg.NumHiddenLayers),
		Config: &cfg,
		tok:    tok,
	}
	for i := range cfg.NumHiddenLayers {
		m.Layers[i] = &Layer{
			IsSliding: cfg.LayerTypes[i] == "sliding_attention",
			// Despite its historical name, 1 means RoPE and 0 means NoPE.
			UseRope: cfg.NoRopeLayers[i] == 1,
		}
	}
	return m, nil
}

func validateTokenizer(tok *tokenizer.Tokenizer) error {
	eom, _ := tok.GetSpecialToken("<|eom|>")
	if tok.IsEOS(eom) {
		return fmt.Errorf("invalid generation config: <|eom|> must be a non-terminal message boundary")
	}

	for _, token := range []string{"<|eot|>", "<|end_of_text|>"} {
		id, _ := tok.GetSpecialToken(token)
		if !tok.IsEOS(id) {
			return fmt.Errorf("invalid generation config: %s must be an EOS token", token)
		}
	}
	return nil
}

func shiftedRMSNorm(weight *mlx.Array, eps float32) *nn.RMSNorm {
	if weight == nil {
		return nil
	}
	return nn.NewRMSNorm(mlx.AddScalar(weight, 1), eps)
}

func firstLinear(linears model.LinearFactory, paths ...string) nn.LinearLayer {
	for _, path := range paths {
		if linear := linears.Make(path); linear != nil {
			return linear
		}
	}
	return nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	officialHF := tensors["model.language_model.embed_tokens.weight"] != nil
	textPrefix := "model"
	if officialHF {
		textPrefix = "model.language_model"
	}

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, textPrefix+".embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight: %s.embed_tokens.weight", textPrefix)
	}

	normWeight := tensors[textPrefix+".norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing final norm weight: %s.norm.weight", textPrefix)
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.RMSNormEps)

	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else {
		m.LMHead = linears.Make("lm_head")
	}
	if m.LMHead == nil {
		return fmt.Errorf("missing language model head: lm_head.weight")
	}

	for i := range m.NumHiddenLayers {
		prefix := fmt.Sprintf("%s.layers.%d", textPrefix, i)
		layer := m.Layers[i]
		layer.Attention = &Attention{
			QProj:          linears.Make(prefix + ".self_attn.q_proj"),
			KProj:          linears.Make(prefix + ".self_attn.k_proj"),
			VProj:          linears.Make(prefix + ".self_attn.v_proj"),
			OProj:          linears.Make(prefix + ".self_attn.o_proj"),
			OutputGateProj: firstLinear(linears, prefix+".self_attn.gate_proj", prefix+".self_attn.output_gate_proj"),
		}
		layer.MLP = &MLP{
			GateProj: linears.Make(prefix + ".mlp.gate_proj"),
			UpProj:   linears.Make(prefix + ".mlp.up_proj"),
			DownProj: linears.Make(prefix + ".mlp.down_proj"),
		}
		layer.InputNorm = shiftedRMSNorm(tensors[prefix+".input_layernorm.weight"], m.RMSNormEps)
		if officialHF {
			layer.PostAttentionNorm = shiftedRMSNorm(tensors[prefix+".pre_feedforward_layernorm.weight"], m.RMSNormEps)
			layer.PostAttnNorm = shiftedRMSNorm(tensors[prefix+".post_attention_layernorm.weight"], m.PostNormEps)
			layer.PostFFNNorm = shiftedRMSNorm(tensors[prefix+".post_feedforward_layernorm.weight"], m.PostNormEps)
		} else {
			layer.PostAttentionNorm = shiftedRMSNorm(tensors[prefix+".post_attention_layernorm.weight"], m.RMSNormEps)
			layer.PostAttnNorm = shiftedRMSNorm(tensors[prefix+".post_attn_norm.weight"], m.PostNormEps)
			layer.PostFFNNorm = shiftedRMSNorm(tensors[prefix+".post_ffn_norm.weight"], m.PostNormEps)
		}

		if layer.InputNorm == nil || layer.PostAttentionNorm == nil || layer.PostAttnNorm == nil || layer.PostFFNNorm == nil {
			return fmt.Errorf("layer %d: missing normalization weight", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("layer %d: missing attention projection", i)
		}
		if m.UseAttentionOutputGate && layer.Attention.OutputGateProj == nil {
			return fmt.Errorf("layer %d: missing attention output gate projection", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("layer %d: missing MLP projection", i)
		}
	}

	if m.HasVision {
		if err := m.loadVisionWeights(tensors, linears); err != nil {
			return err
		}
	}

	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	h := m.embed(b.InputIDs)
	if len(b.Media) > 0 {
		h = m.scatterMedia(h, b)
	}
	return m.forwardEmbeddings(h, b, caches)
}

func (m *Model) embed(inputIDs *mlx.Array) *mlx.Array {
	h := m.EmbedTokens.Forward(inputIDs)
	if m.NormalizeTokenEmbeddings {
		h = mlx.RMSNormFn(h, nil, m.RMSNormEps)
	}
	return h
}

func (m *Model) forwardEmbeddings(h *mlx.Array, b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	var features []*mlx.Array
	for i, layer := range m.Layers {
		var c cache.Cache
		if caches != nil && i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
		if slices.Contains(m.auxHiddenLayers, i) {
			features = append(features, h)
		}
	}

	out := m.Norm.Forward(h, m.RMSNormEps)
	if features != nil {
		return out, mlx.Concatenate(features, -1)
	}
	return out, out
}

// SetAuxHiddenLayers taps the listed layers' outputs, which Forward then
// returns as the draft-conditioning state in place of the final hidden.
func (m *Model) SetAuxHiddenLayers(layers []int) {
	m.auxHiddenLayers = layers
}

// TokenEmbeddings is the raw table lookup, for a draft that embeds with the
// target's table. Note the model's optional embedding normalization does not
// apply here; the draft mirrors what the target was trained with.
func (m *Model) TokenEmbeddings(ids *mlx.Array) *mlx.Array {
	return m.EmbedTokens.Forward(ids)
}

// RawLogits is the raw head projection, skipping the float32 cast, output
// multiplier and softcap the target's own Unembed applies.
func (m *Model) RawLogits(hidden *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(hidden)
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x).AsType(mlx.DTypeFloat32)
	if m.OutputMultiplier != 1 {
		logits = mlx.MulScalar(logits, m.OutputMultiplier)
	}
	if m.OutputSoftCapTemp > 0 {
		cap := mlx.FromValue(m.OutputSoftCapTemp).AsType(logits.DType())
		logits = mlx.LogitSoftcap(logits, cap)
	}
	return logits
}

func (m *Model) NumLayers() int                  { return len(m.Layers) }
func (m *Model) MaxContextLength() int           { return int(m.MaxPositionEmbeddings) }
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// NewCaches uses bounded rotating caches for sliding layers and full KV caches
// for the periodic NoPE/global-attention layers.
func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		if layer.IsSliding {
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	attnInput := l.InputNorm.Forward(x, cfg.RMSNormEps)
	attnOut := l.Attention.Forward(attnInput, b, c, positions, B, L, l.UseRope, cfg)
	h := mlx.Add(x, l.PostAttnNorm.Forward(attnOut, cfg.PostNormEps))

	mlpInput := l.PostAttentionNorm.Forward(h, cfg.RMSNormEps)
	mlpOut := l.MLP.Forward(mlpInput)
	return mlx.Add(h, l.PostFFNNorm.Forward(mlpOut, cfg.PostNormEps))
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, useRope bool, cfg *Config) *mlx.Array {
	q := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	if cfg.UseQKNorm {
		q = mlx.MulScalar(mlx.RMSNormFn(q, nil, cfg.RMSNormEps), cfg.QueryScale)
		k = mlx.RMSNormFn(k, nil, cfg.RMSNormEps)
	}

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	if useRope {
		q = mlx.RoPEWithBase(q, int(cfg.HeadDim), true, cfg.RopeTheta, 1, positions)
		k = mlx.RoPEWithBase(k, int(cfg.HeadDim), true, cfg.RopeTheta, 1, positions)
	}

	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.AttentionScale, kv, nn.WithMask(nn.CausalMask()))
	out = mlx.Transpose(out, 0, 2, 1, 3)

	if cfg.UseAttentionOutputGate {
		gate := mlx.Reshape(a.OutputGateProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
		out = mlx.Mul(out, mlx.Sigmoid(gate))
	}

	out = mlx.Reshape(out, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}
