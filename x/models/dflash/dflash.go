// Package dflash implements DFlash block-diffusion draft models for MLX.
package dflash

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

func init() {
	base.RegisterDraft("DFlashDraftModel", newModel)
	base.RegisterDraft("dflash", newModel)
}

var _ base.DFlashDraftModel = (*Model)(nil)

type Config struct {
	HiddenSize            int32              `json:"hidden_size"`
	NumHiddenLayers       int32              `json:"num_hidden_layers"`
	NumAttentionHeads     int32              `json:"num_attention_heads"`
	NumKeyValueHeads      int32              `json:"num_key_value_heads"`
	HeadDim               int32              `json:"head_dim"`
	IntermediateSize      int32              `json:"intermediate_size"`
	VocabSize             int32              `json:"vocab_size"`
	RMSNormEps            float32            `json:"rms_norm_eps"`
	RopeTheta             float32            `json:"rope_theta"`
	RopeScaling           *nn.RopeParameters `json:"rope_scaling"`
	RopeParameters        *nn.RopeParameters `json:"rope_parameters"`
	MaxPositionEmbeddings int32              `json:"max_position_embeddings"`
	BlockSizeValue        int32              `json:"block_size"`
	NumTargetLayers       int32              `json:"num_target_layers"`
	LayerTypes            []string           `json:"layer_types"`
	SlidingWindow         int32              `json:"sliding_window"`
	FinalLogitSoftcapping *float32           `json:"final_logit_softcapping"`
	DFlash                struct {
		TargetLayerIDs []int `json:"target_layer_ids"`
		MaskTokenID    int32 `json:"mask_token_id"`
	} `json:"dflash_config"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
	Scale          float32                           `json:"-"`
	RopeFreqs      *mlx.Array                        `json:"-"`
	RopeScale      float32                           `json:"-"`
}

type Model struct {
	FC         nn.LinearLayer
	HiddenNorm *nn.RMSNorm
	Layers     []*Layer
	Norm       *nn.RMSNorm

	target           base.Model
	targetEmbeddings base.MTPEmbeddingModel
	tensorPrefix     string

	*Config
}

type Layer struct {
	Attention         *Attention
	MLP               *MLP
	InputNorm         *nn.RMSNorm
	PostAttentionNorm *nn.RMSNorm
}

type Attention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	Sliding bool
}

type MLP struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func parseConfig(data []byte) (Config, error) {
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse dflash config: %w", err)
	}
	if cfg.HiddenSize <= 0 {
		return Config{}, fmt.Errorf("invalid hidden_size: %d", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers <= 0 {
		return Config{}, fmt.Errorf("invalid num_hidden_layers: %d", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads <= 0 {
		return Config{}, fmt.Errorf("invalid num_attention_heads: %d", cfg.NumAttentionHeads)
	}
	if cfg.NumKeyValueHeads <= 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim <= 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return Config{}, fmt.Errorf("hidden_size (%d) must be divisible by num_attention_heads (%d)", cfg.HiddenSize, cfg.NumAttentionHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeTheta == 0 {
		ropeParams := cfg.RopeParameters
		if ropeParams == nil {
			ropeParams = cfg.RopeScaling
		}
		if ropeParams != nil && ropeParams.RopeTheta > 0 {
			cfg.RopeTheta = ropeParams.RopeTheta
		}
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000
	}
	if cfg.BlockSizeValue <= 0 {
		return Config{}, fmt.Errorf("invalid block_size: %d", cfg.BlockSizeValue)
	}
	if len(cfg.DFlash.TargetLayerIDs) == 0 {
		return Config{}, fmt.Errorf("dflash_config.target_layer_ids is required")
	}
	if !sort.IntsAreSorted(cfg.DFlash.TargetLayerIDs) {
		return Config{}, fmt.Errorf("dflash_config.target_layer_ids must be sorted")
	}
	if len(cfg.LayerTypes) == 0 {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.LayerTypes {
			cfg.LayerTypes[i] = "full_attention"
		}
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return Config{}, fmt.Errorf("layer_types length %d does not match num_hidden_layers %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	for i, typ := range cfg.LayerTypes {
		switch strings.ToLower(typ) {
		case "full_attention":
		case "sliding_attention":
			if cfg.SlidingWindow <= 0 {
				return Config{}, fmt.Errorf("layer %d uses sliding_attention but sliding_window is not set", i)
			}
		default:
			return Config{}, fmt.Errorf("unsupported layer type %q", typ)
		}
	}
	cfg.Scale = float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	cfg.RopeScale = 1
	ropeParams := cfg.RopeParameters
	if ropeParams == nil {
		ropeParams = cfg.RopeScaling
	}
	if ropeParams != nil && strings.EqualFold(ropeParams.TypeName(), "yarn") {
		cfg.RopeFreqs, cfg.RopeScale = nn.BuildYarnRopeFreqs(int(cfg.HeadDim), cfg.RopeTheta, ropeParams)
	}
	return cfg, nil
}

func newModel(root *model.Root, target base.Model) (base.DraftModel, error) {
	if root == nil || root.Draft == nil {
		return nil, fmt.Errorf("draft metadata missing")
	}

	configPath := root.Draft.Config
	if configPath == "" {
		configPath = "draft/config.json"
	}
	configData, err := root.Manifest.ReadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("load dflash config: %w", err)
	}

	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}
	if target.NumLayers() < int(cfg.NumTargetLayers) {
		return nil, fmt.Errorf("dflash target expects %d layers, target has %d", cfg.NumTargetLayers, target.NumLayers())
	}
	for _, layerID := range cfg.DFlash.TargetLayerIDs {
		if layerID < 0 || layerID >= target.NumLayers() {
			return nil, fmt.Errorf("dflash target layer id %d out of range for %d-layer target", layerID, target.NumLayers())
		}
	}
	targetEmbeddings, ok := target.(base.MTPEmbeddingModel)
	if !ok {
		return nil, fmt.Errorf("dflash draft requires target token embeddings, got %T", target)
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

	prefix := root.Draft.TensorPrefix
	if prefix == "" {
		prefix = "draft."
	}

	m := &Model{
		Config:           &cfg,
		Layers:           make([]*Layer, cfg.NumHiddenLayers),
		target:           target,
		targetEmbeddings: targetEmbeddings,
		tensorPrefix:     prefix,
	}
	for i := range m.Layers {
		m.Layers[i] = &Layer{Attention: &Attention{}, MLP: &MLP{}}
	}
	return m, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := m.tensorPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	m.FC = linears.Make(prefix + "fc")
	if m.FC == nil {
		return fmt.Errorf("missing dflash fc weight")
	}
	if w := tensors[prefix+"hidden_norm.weight"]; w != nil {
		m.HiddenNorm = nn.NewRMSNorm(w, m.RMSNormEps)
	}
	if w := tensors[prefix+"norm.weight"]; w != nil {
		m.Norm = nn.NewRMSNorm(w, m.RMSNormEps)
	}
	if m.HiddenNorm == nil || m.Norm == nil {
		return fmt.Errorf("missing dflash norm weights")
	}

	for i := range m.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		layer := &Layer{
			Attention: &Attention{Sliding: strings.ToLower(m.LayerTypes[i]) == "sliding_attention"},
			MLP: &MLP{
				GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
			},
		}
		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttentionNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		layer.Attention.QProj = linears.Make(layerPrefix + ".self_attn.q_proj")
		layer.Attention.KProj = linears.Make(layerPrefix + ".self_attn.k_proj")
		layer.Attention.VProj = linears.Make(layerPrefix + ".self_attn.v_proj")
		layer.Attention.OProj = linears.Make(layerPrefix + ".self_attn.o_proj")
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}

		if layer.InputNorm == nil || layer.PostAttentionNorm == nil {
			return fmt.Errorf("dflash layer %d: missing layer norms", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil || layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("dflash layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("dflash layer %d: missing attention q/k norms", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("dflash layer %d: missing mlp projections", i)
		}

		m.Layers[i] = layer
	}
	return nil
}

func (m *Model) TargetLayerIDs() []int {
	return append([]int(nil), m.DFlash.TargetLayerIDs...)
}

func (m *Model) BlockSize() int {
	return int(m.BlockSizeValue)
}

func (m *Model) MaskTokenID() int32 {
	return m.DFlash.MaskTokenID
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, typ := range m.LayerTypes {
		if strings.ToLower(typ) == "sliding_attention" {
			// RotatingKVCache.View returns maxSize-1 tokens so assistant
			// paths can append the current query. DFlash uses that same
			// view for target context, so allocate one extra slot to expose
			// the draft model's sliding_window-1 context tokens.
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		} else {
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (m *Model) AppendContext(targetHidden *mlx.Array, caches []cache.Cache) {
	if targetHidden == nil || targetHidden.Dim(1) == 0 {
		return
	}
	hCtx := m.HiddenNorm.Forward(m.FC.Forward(targetHidden), m.RMSNormEps)
	offset := int32(0)
	if len(caches) > 0 && caches[0] != nil {
		offset = int32(caches[0].Offset())
	}
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, targetHidden.Dim(0), targetHidden.Dim(1)),
		SeqOffsets:   []int32{offset},
		SeqQueryLens: []int32{int32(targetHidden.Dim(1))},
	}
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	for i, layer := range m.Layers {
		if i >= len(caches) || caches[i] == nil {
			continue
		}
		layer.Attention.AppendContext(hCtx, b, positions, caches[i], m.Config)
	}
}

func (m *Model) Draft(inputIDs *mlx.Array, caches []cache.Cache) *mlx.Array {
	dims := inputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	offset := int32(0)
	if len(caches) > 0 && caches[0] != nil {
		offset = int32(caches[0].Offset())
	}
	b := &batch.Batch{
		InputIDs:     inputIDs,
		SeqOffsets:   []int32{offset},
		SeqQueryLens: []int32{L},
	}
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.targetEmbeddings.TokenEmbeddings(inputIDs)
	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	logits := m.target.Unembed(m.Norm.Forward(h, m.RMSNormEps))
	if m.FinalLogitSoftcapping != nil {
		cap := mlx.FromValue(*m.FinalLogitSoftcapping).AsType(logits.DType())
		logits = mlx.LogitSoftcap(logits, cap)
	}
	return logits
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.InputNorm.Forward(x, cfg.RMSNormEps), b, c, positions, B, L, cfg))
	return mlx.Add(h, l.MLP.Forward(l.PostAttentionNorm.Forward(h, cfg.RMSNormEps)))
}

func (a *Attention) AppendContext(xCtx *mlx.Array, b *batch.Batch, positions *mlx.Array, c cache.Cache, cfg *Config) {
	B, L := int32(xCtx.Dim(0)), int32(xCtx.Dim(1))
	k := a.KProj.Forward(xCtx)
	v := a.VProj.Forward(xCtx)

	k = mlx.Reshape(k, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	k = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), int(cfg.HeadDim), cfg.RopeScale)

	c.(cache.Attention).Update(b, k, v)
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	q := a.QProj.Forward(x)
	propK := a.KProj.Forward(x)
	propV := a.VProj.Forward(x)

	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	propK = mlx.Reshape(propK, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	propV = mlx.Reshape(propV, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	propK = a.KNorm.Forward(propK, cfg.RMSNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	propK = mlx.Transpose(propK, 0, 2, 1, 3)
	propV = mlx.Transpose(propV, 0, 2, 1, 3)

	q = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), int(cfg.HeadDim), cfg.RopeScale)
	propK = nn.ScaleRotaryPart(mlx.RoPEWithFreqs(propK, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, positions, cfg.RopeFreqs), int(cfg.HeadDim), cfg.RopeScale)

	k, v := propK, propV
	if viewer, ok := c.(cache.Viewer); ok {
		if history := viewer.View(b); history != nil {
			k = history.K().Concatenate(2, propK)
			v = history.V().Concatenate(2, propV)
		}
	}

	mask := nn.AttentionMask{}
	if a.Sliding {
		mask = nn.CausalMask()
		if int(cfg.SlidingWindow) > 0 && k.Dim(2) > int(cfg.SlidingWindow) {
			mask = mask.Intersect(nn.SlidingWindowMask(b, k.Dim(2), int(cfg.SlidingWindow), q.DType()))
		}
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, nn.WithKV(k, v, []int32{int32(k.Dim(2))}), nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.DownProj.Forward(mlx.SwiGLU(m.GateProj.Forward(x), m.UpProj.Forward(x)))
}
