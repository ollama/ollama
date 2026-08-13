// Package dflash implements the DFlash block-diffusion draft model:
// qwen3-shaped layers drafting a whole block per forward, conditioned on
// tapped target hidden states as key/value context.
package dflash

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

func init() {
	base.RegisterDraft("DFlashDraftModel", func(root *model.Root, target base.Model) (base.DraftModel, error) {
		return newModel(root, target, false)
	})
	base.RegisterDraft("DFlashLagunaForCausalLM", func(root *model.Root, target base.Model) (base.DraftModel, error) {
		return newModel(root, target, true)
	})
	base.RegisterDraft("MuseGlimmerAssistantModel", func(root *model.Root, target base.Model) (base.DraftModel, error) {
		return newModel(root, target, false)
	})
}

var _ base.BlockDraft = (*Model)(nil)

type Config struct {
	HiddenSize        int32
	NumHiddenLayers   int32
	NumAttentionHeads int32
	NumKeyValueHeads  int32
	HeadDim           int32
	RMSNormEps        float32
	RopeTheta         float32
	Scale             float32
	SlidingWindow     int32
	LayerTypes        []string

	BlockSize      int
	MaskTokenID    int32
	VocabSize      int32
	TargetLayerIDs []int

	// RopeInterleaved selects the draft's rotary pairing convention:
	// true pairs adjacent dims (torch view_as_complex over pairs, the glimmer
	// publisher convention); false pairs split halves (HF rotate_half, the
	// laguna convention). Defaults to false for backwards compatibility with
	// laguna drafts.
	RopeInterleaved bool

	// Causal, when set, overrides every layer's attention direction;
	// otherwise only sliding layers run causal.
	Causal *bool
}

// draftTarget is what dflash requires of its target beyond base.Model.
type draftTarget interface {
	base.Model

	// TokenEmbeddings is the raw table lookup; the draft has no table of
	// its own.
	TokenEmbeddings(ids *mlx.Array) *mlx.Array

	// RawLogits is the raw head projection, skipping any output decoration
	// the target's own Unembed applies.
	RawLogits(hidden *mlx.Array) *mlx.Array

	// SetAuxHiddenLayers taps layer outputs: id i means the hidden state
	// after layer i, the same convention as checkpoint target_layer_ids.
	SetAuxHiddenLayers(layers []int)

	// NumLayers is used to validate the config's tap ids.
	NumLayers() int
}

type Model struct {
	FC         nn.LinearLayer
	HiddenNorm *nn.RMSNorm
	Norm       *nn.RMSNorm
	Layers     []*Layer

	// AuxNorms, when shipped, normalize each target slice before fusion.
	AuxNorms []*nn.RMSNorm

	// ctxLayerNorm passes context rows through each layer's input norm, a
	// laguna convention that neither tensors nor config indicate.
	ctxLayerNorm bool

	*Config

	target draftTarget

	tensorPrefix string

	QuantGroupSize int
	QuantBits      int
	QuantMode      string
	TensorQuant    map[string]*model.TensorQuantInfo
}

type Layer struct {
	InputNorm    *nn.RMSNorm
	PostAttnNorm *nn.RMSNorm
	Attention    *Attention
	MLP          *MLP
	IsSliding    bool
	IsCausal     bool
}

// Attention holds a q projection and a fused k|v projection: split
// checkpoints are stacked at load, fused ones sliced. Context rows produce
// no queries, so the context path uses only KVProj.
type Attention struct {
	QProj  nn.LinearLayer
	KVProj nn.LinearLayer
	GProj  nn.LinearLayer
	OProj  nn.LinearLayer
	QNorm  *nn.RMSNorm
	KNorm  *nn.RMSNorm

	// ctxInputNorm applies the layer's input norm to context rows (laguna).
	ctxInputNorm *nn.RMSNorm
}

type MLP struct {
	// GateUpProj is gate|up stacked at load; SwiGLU splits the halves.
	GateUpProj nn.LinearLayer
	DownProj   nn.LinearLayer
}

func parseConfig(data []byte) (*Config, error) {
	var raw struct {
		HiddenSize        int32   `json:"hidden_size"`
		NumHiddenLayers   int32   `json:"num_hidden_layers"`
		NumAttentionHeads int32   `json:"num_attention_heads"`
		NumKeyValueHeads  int32   `json:"num_key_value_heads"`
		HeadDim           int32   `json:"head_dim"`
		RMSNormEps        float32 `json:"rms_norm_eps"`
		RopeTheta         float32 `json:"rope_theta"`
		RopeParameters    struct {
			RopeTheta float32 `json:"rope_theta"`
		} `json:"rope_parameters"`
		BlockSize    int `json:"block_size"`
		DFlashConfig struct {
			BlockSize       int    `json:"block_size"`
			MaskTokenID     *int32 `json:"mask_token_id"`
			TargetLayerIDs  []int  `json:"target_layer_ids"`
			NumTargetLayers int    `json:"num_target_layers"`
			Causal          *bool  `json:"causal"`
		} `json:"dflash_config"`
		NumTargetLayers int      `json:"num_target_layers"`
		VocabSize       int32    `json:"vocab_size"`
		LayerTypes      []string `json:"layer_types"`
		SlidingWindow   int32    `json:"sliding_window"`
		RopeInterleaved *bool    `json:"rope_interleaved"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse dflash config: %w", err)
	}

	cfg := &Config{
		HiddenSize:        raw.HiddenSize,
		NumHiddenLayers:   raw.NumHiddenLayers,
		NumAttentionHeads: raw.NumAttentionHeads,
		NumKeyValueHeads:  raw.NumKeyValueHeads,
		HeadDim:           raw.HeadDim,
		RMSNormEps:        raw.RMSNormEps,
		RopeTheta:         raw.RopeTheta,
		SlidingWindow:     raw.SlidingWindow,
		LayerTypes:        raw.LayerTypes,
		BlockSize:         raw.DFlashConfig.BlockSize,
		VocabSize:         raw.VocabSize,
		TargetLayerIDs:    raw.DFlashConfig.TargetLayerIDs,
		Causal:            raw.DFlashConfig.Causal,
	}
	if raw.RopeInterleaved != nil {
		cfg.RopeInterleaved = *raw.RopeInterleaved
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = raw.RopeParameters.RopeTheta
	}
	if cfg.BlockSize == 0 {
		cfg.BlockSize = raw.BlockSize
	}
	cfg.Scale = float32(math.Pow(float64(cfg.HeadDim), -0.5))

	if cfg.BlockSize < 2 {
		return nil, fmt.Errorf("dflash block size %d must be at least 2", cfg.BlockSize)
	}
	if raw.DFlashConfig.MaskTokenID == nil {
		return nil, fmt.Errorf("dflash config missing mask_token_id")
	}
	cfg.MaskTokenID = *raw.DFlashConfig.MaskTokenID
	if len(cfg.TargetLayerIDs) == 0 {
		return nil, fmt.Errorf("dflash config missing target_layer_ids")
	}
	for i, id := range cfg.TargetLayerIDs {
		if id < 0 || (i > 0 && id <= cfg.TargetLayerIDs[i-1]) {
			return nil, fmt.Errorf("dflash target_layer_ids must be ascending and non-negative")
		}
	}
	if n := max(raw.NumTargetLayers, raw.DFlashConfig.NumTargetLayers); n > 0 && cfg.TargetLayerIDs[len(cfg.TargetLayerIDs)-1] >= n {
		return nil, fmt.Errorf("dflash target layer %d out of range for %d target layers",
			cfg.TargetLayerIDs[len(cfg.TargetLayerIDs)-1], n)
	}
	if len(cfg.LayerTypes) == 0 {
		cfg.LayerTypes = make([]string, cfg.NumHiddenLayers)
		for i := range cfg.LayerTypes {
			cfg.LayerTypes[i] = "full_attention"
		}
	}
	if len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return nil, fmt.Errorf("dflash layer_types length %d != num_hidden_layers %d", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	for _, t := range cfg.LayerTypes {
		switch t {
		case "full_attention":
		case "sliding_attention":
			if cfg.SlidingWindow <= 0 {
				return nil, fmt.Errorf("dflash sliding_attention layers require sliding_window")
			}
		default:
			return nil, fmt.Errorf("unsupported dflash layer type %q", t)
		}
	}
	return cfg, nil
}

func newModel(root *model.Root, targetModel base.Model, ctxLayerNorm bool) (base.DraftModel, error) {
	if root == nil || root.Draft == nil {
		return nil, fmt.Errorf("draft metadata missing")
	}

	configPath := root.Draft.Config
	if configPath == "" {
		configPath = "draft/config.json"
	}
	configData, err := root.Manifest.ReadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("load draft config: %w", err)
	}
	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	target, ok := targetModel.(draftTarget)
	if !ok {
		return nil, fmt.Errorf("dflash draft is not supported with this target model")
	}

	var trained struct {
		NumTargetLayers int `json:"num_target_layers"`
		DFlashConfig    struct {
			NumTargetLayers int `json:"num_target_layers"`
		} `json:"dflash_config"`
	}
	_ = json.Unmarshal(configData, &trained)
	if n := max(trained.NumTargetLayers, trained.DFlashConfig.NumTargetLayers); n > 0 && n != target.NumLayers() {
		return nil, fmt.Errorf("dflash draft trained for %d target layers, target has %d", n, target.NumLayers())
	}
	if last := cfg.TargetLayerIDs[len(cfg.TargetLayerIDs)-1]; last >= target.NumLayers() {
		return nil, fmt.Errorf("dflash target layer %d out of range for %d target layers", last, target.NumLayers())
	}

	// The manifest can pair any draft with any target; probe the borrowed
	// table and head (static shapes, nothing evaluated) to verify the fit.
	emb := target.TokenEmbeddings(mlx.FromValues([]int32{0}, 1, 1))
	if w := emb.Dim(2); w != int(cfg.HiddenSize) {
		return nil, fmt.Errorf("dflash draft trained for hidden size %d, target has %d", cfg.HiddenSize, w)
	}
	vocab := target.RawLogits(emb).Dim(2)
	if cfg.VocabSize > 0 && int(cfg.VocabSize) != vocab {
		return nil, fmt.Errorf("dflash draft trained for a %d-token vocabulary, target has %d", cfg.VocabSize, vocab)
	}
	if cfg.MaskTokenID < 0 || int(cfg.MaskTokenID) >= vocab {
		return nil, fmt.Errorf("dflash mask token %d outside the target's %d-token vocabulary", cfg.MaskTokenID, vocab)
	}
	target.SetAuxHiddenLayers(cfg.TargetLayerIDs)

	tensorPrefix := root.Draft.TensorPrefix
	if tensorPrefix == "" {
		tensorPrefix = "draft."
	}

	m := &Model{
		Config:       cfg,
		target:       target,
		ctxLayerNorm: ctxLayerNorm,
		tensorPrefix: tensorPrefix,
		Layers:       make([]*Layer, cfg.NumHiddenLayers),
		TensorQuant:  root.AllTensorQuant(),
	}
	if qt := root.QuantType(); qt != "" {
		m.QuantGroupSize, m.QuantBits, m.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			m.QuantGroupSize = gs
		}
	}
	return m, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := m.tensorPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	if m.FC = linears.Make(prefix + "fc"); m.FC == nil {
		return fmt.Errorf("missing dflash fc weight")
	}
	for name, dst := range map[string]**nn.RMSNorm{
		"hidden_norm.weight": &m.HiddenNorm,
		"norm.weight":        &m.Norm,
	} {
		w := tensors[prefix+name]
		if w == nil {
			return fmt.Errorf("missing dflash %s", name)
		}
		*dst = nn.NewRMSNorm(w, m.RMSNormEps)
	}

	for i := 0; ; i++ {
		w := tensors[fmt.Sprintf("%saux_hidden_norms.%d.weight", prefix, i)]
		if w == nil {
			break
		}
		m.AuxNorms = append(m.AuxNorms, nn.NewRMSNorm(w, m.RMSNormEps))
	}
	if len(m.AuxNorms) > 0 && len(m.AuxNorms) != len(m.TargetLayerIDs) {
		return fmt.Errorf("dflash has %d aux hidden norms for %d target layers", len(m.AuxNorms), len(m.TargetLayerIDs))
	}

	for i := range m.Layers {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, i)
		layer := &Layer{
			IsSliding: m.LayerTypes[i] == "sliding_attention",
			Attention: &Attention{
				GProj: linears.Make(layerPrefix + ".self_attn.g_proj"),
				OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
			},
			MLP: &MLP{
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
			},
		}
		a := layer.Attention
		qDim := m.NumAttentionHeads * m.HeadDim
		kvDim := m.NumKeyValueHeads * m.HeadDim
		if fused := linears.Make(layerPrefix + ".self_attn.qkv_proj"); fused != nil {
			a.QProj = sliceLinearRows(fused, 0, qDim)
			a.KVProj = sliceLinearRows(fused, qDim, qDim+2*kvDim)
		} else if q := linears.Make(layerPrefix + ".self_attn.q_proj"); q != nil {
			k := linears.Make(layerPrefix + ".self_attn.k_proj")
			v := linears.Make(layerPrefix + ".self_attn.v_proj")
			if k != nil && v != nil {
				kv, err := stackLinears(k, v)
				if err != nil {
					return fmt.Errorf("dflash layer %d k|v: %w", i, err)
				}
				a.QProj, a.KVProj = q, kv
			}
		}
		if gate := linears.Make(layerPrefix + ".mlp.gate_proj"); gate != nil {
			if up := linears.Make(layerPrefix + ".mlp.up_proj"); up != nil {
				gu, err := stackLinears(gate, up)
				if err != nil {
					return fmt.Errorf("dflash layer %d gate|up: %w", i, err)
				}
				layer.MLP.GateUpProj = gu
			}
		}
		layer.IsCausal = layer.IsSliding
		if m.Causal != nil {
			layer.IsCausal = *m.Causal
		}
		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, m.RMSNormEps)
		}
		if m.ctxLayerNorm {
			layer.Attention.ctxInputNorm = layer.InputNorm
		}

		if a.QProj == nil || a.KVProj == nil || a.OProj == nil || a.QNorm == nil || a.KNorm == nil {
			return fmt.Errorf("dflash layer %d: missing attention weights", i)
		}
		if layer.MLP.GateUpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("dflash layer %d: missing mlp weights", i)
		}
		if layer.InputNorm == nil || layer.PostAttnNorm == nil {
			return fmt.Errorf("dflash layer %d: missing norm weights", i)
		}
		m.Layers[i] = layer
	}
	return nil
}

// stackLinears concatenates two linears along the output dimension. Quant
// groups run along the input dimension, so this is exact; per-tensor global
// scales are expanded to per-row so each half keeps its own.
func stackLinears(a, b nn.LinearLayer) (nn.LinearLayer, error) {
	if pa, ok := a.(*nn.Linear); ok {
		pb, ok := b.(*nn.Linear)
		if !ok {
			return nil, fmt.Errorf("stack linears: mixed plain and quantized parts")
		}
		return &nn.Linear{
			Weight: mlx.Concatenate([]*mlx.Array{pa.Weight, pb.Weight}, 0),
			Bias:   concatBias(pa.Bias, int32(pa.Weight.Dim(0)), pb.Bias, int32(pb.Weight.Dim(0))),
		}, nil
	}
	qa, ok := a.(*nn.QuantizedLinear)
	if !ok {
		return nil, fmt.Errorf("stack linears: unsupported layer type %T", a)
	}
	qb, ok := b.(*nn.QuantizedLinear)
	if !ok {
		return nil, fmt.Errorf("stack linears: mixed plain and quantized parts")
	}
	if qa.GroupSize != qb.GroupSize || qa.Bits != qb.Bits || qa.Mode != qb.Mode {
		return nil, fmt.Errorf("stack linears: quant mode mismatch %s/%d/%d vs %s/%d/%d",
			qa.Mode, qa.Bits, qa.GroupSize, qb.Mode, qb.Bits, qb.GroupSize)
	}
	if (qa.QBiases == nil) != (qb.QBiases == nil) {
		return nil, fmt.Errorf("stack linears: quant bias layout mismatch")
	}
	out := &nn.QuantizedLinear{
		Weight:    mlx.Concatenate([]*mlx.Array{qa.Weight, qb.Weight}, 0),
		Scales:    mlx.Concatenate([]*mlx.Array{qa.Scales, qb.Scales}, 0),
		GroupSize: qa.GroupSize,
		Bits:      qa.Bits,
		Mode:      qa.Mode,
	}
	if qa.QBiases != nil {
		out.QBiases = mlx.Concatenate([]*mlx.Array{qa.QBiases, qb.QBiases}, 0)
	}
	out.Bias = concatBias(qa.Bias, int32(qa.Scales.Dim(0)), qb.Bias, int32(qb.Scales.Dim(0)))
	if qa.GlobalScale != nil || qb.GlobalScale != nil {
		out.GlobalScale = mlx.Concatenate([]*mlx.Array{
			perRowGlobal(qa.GlobalScale, int32(qa.Scales.Dim(0))),
			perRowGlobal(qb.GlobalScale, int32(qb.Scales.Dim(0))),
		}, 0)
	}
	return out, nil
}

// perRowGlobal expands a per-tensor (or nil, meaning 1.0) global scale to a
// per-row vector; an already per-row scale passes through unchanged.
func perRowGlobal(g *mlx.Array, rows int32) *mlx.Array {
	ones := make([]float32, rows)
	for i := range ones {
		ones[i] = 1
	}
	v := mlx.FromValues(ones, int(rows))
	if g == nil {
		return v
	}
	return mlx.Mul(v, g)
}

func concatBias(a *mlx.Array, aRows int32, b *mlx.Array, bRows int32) *mlx.Array {
	if a == nil && b == nil {
		return nil
	}
	fill := func(bias *mlx.Array, rows int32, like *mlx.Array) *mlx.Array {
		if bias != nil {
			return bias
		}
		return mlx.ZerosF32([]int32{rows}).AsType(like.DType())
	}
	if a == nil {
		a = fill(nil, aRows, b)
	}
	if b == nil {
		b = fill(nil, bRows, a)
	}
	return mlx.Concatenate([]*mlx.Array{a, b}, 0)
}

// sliceLinearRows returns rows [start, stop) of l along the output dimension.
func sliceLinearRows(l nn.LinearLayer, start, stop int32) nn.LinearLayer {
	rows := func(t *mlx.Array) *mlx.Array {
		if t == nil {
			return nil
		}
		dims := t.Dims()
		starts := make([]int32, len(dims))
		stops := make([]int32, len(dims))
		for i, d := range dims {
			stops[i] = int32(d)
		}
		starts[0], stops[0] = start, stop
		return mlx.SliceStartStop(t, starts, stops)
	}
	switch q := l.(type) {
	case *nn.Linear:
		return &nn.Linear{Weight: rows(q.Weight), Bias: rows(q.Bias)}
	case *nn.QuantizedLinear:
		g := q.GlobalScale
		if g != nil && len(g.Dims()) == 1 && g.Dim(0) == q.Scales.Dim(0) {
			g = rows(g)
		}
		return &nn.QuantizedLinear{
			Weight:      rows(q.Weight),
			Scales:      rows(q.Scales),
			QBiases:     rows(q.QBiases),
			Bias:        rows(q.Bias),
			GlobalScale: g,
			GroupSize:   q.GroupSize,
			Bits:        q.Bits,
			Mode:        q.Mode,
		}
	}
	return nil
}

func (m *Model) BlockParams() (int, int32) { return m.BlockSize, m.MaskTokenID }

// NewCaches builds the per-layer context caches.
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

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.target.RawLogits(x)
}

// Forward writes b.Hidden's rows into each layer's context cache starting at
// SeqOffsets[0] and runs b.InputIDs as a block positioned after them; queries
// come from the block only. Either input may be absent: with no block the
// call just extends the context, with no context the block drafts from
// whatever is already cached.
func (m *Model) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	kv := draftCaches

	var hctx *mlx.Array
	nCtx := int32(0)
	if b.Hidden != nil {
		features := b.Hidden
		if len(m.AuxNorms) > 0 {
			slices := make([]*mlx.Array, len(m.AuxNorms))
			for i, norm := range m.AuxNorms {
				lo := int32(i) * m.HiddenSize
				slices[i] = norm.Forward(features.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(lo), int(lo+m.HiddenSize))), m.RMSNormEps)
			}
			features = mlx.Concatenate(slices, -1)
		}
		hctx = m.HiddenNorm.Forward(m.FC.Forward(features), m.RMSNormEps)
		nCtx = int32(b.Hidden.Dim(1))
	}
	ctxPositions := mlx.FromValues([]int32{b.SeqOffsets[0]}, 1)

	var h, blockPositions *mlx.Array
	var bb *batch.Batch
	var B, L int32
	if b.InputIDs != nil {
		dims := b.InputIDs.Dims()
		B, L = int32(dims[0]), int32(dims[1])
		h = m.target.TokenEmbeddings(b.InputIDs)
		blockStart := b.SeqOffsets[0] + nCtx
		bb = &batch.Batch{InputIDs: b.InputIDs, SeqOffsets: []int32{blockStart}, SeqQueryLens: b.SeqQueryLens}
		blockPositions = mlx.FromValues([]int32{blockStart}, 1)
	}

	for i, layer := range m.Layers {
		var ctxK, ctxV *mlx.Array
		if hctx != nil {
			ctxK, ctxV = layer.Attention.contextKV(hctx, ctxPositions, m.Config)
		}
		if h == nil {
			if ctxK != nil {
				kv[i].(cache.Attention).Update(b, ctxK, ctxV)
			}
			continue
		}
		var mask nn.AttentionMask
		// A sliding layer's window comes from its cache, not from this mask.
		if layer.IsCausal {
			mask = nn.CausalMask()
		}
		h = layer.Forward(h, ctxK, ctxV, kv[i], bb, blockPositions, mask, B, L, m.Config)
	}
	if h == nil {
		return nil, nil
	}
	hidden = m.Norm.Forward(h, m.RMSNormEps)
	return hidden, hidden
}

func (l *Layer) Forward(x, ctxK, ctxV *mlx.Array, c cache.Cache, bb *batch.Batch, positions *mlx.Array, mask nn.AttentionMask, B, L int32, cfg *Config) *mlx.Array {
	h := mlx.Add(x, l.Attention.Forward(l.InputNorm.Forward(x, cfg.RMSNormEps), ctxK, ctxV, c, bb, positions, mask, B, L, cfg))
	return mlx.Add(h, l.MLP.Forward(l.PostAttnNorm.Forward(h, cfg.RMSNormEps)))
}

// contextKV projects feature rows into the layer's context K/V.
func (a *Attention) contextKV(hctx *mlx.Array, positions *mlx.Array, cfg *Config) (k, v *mlx.Array) {
	if a.ctxInputNorm != nil {
		hctx = a.ctxInputNorm.Forward(hctx, cfg.RMSNormEps)
	}
	dims := hctx.Dims()
	B, S := int32(dims[0]), int32(dims[1])
	k, v = a.splitKV(a.KVProj.Forward(hctx), B, S, cfg)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), cfg.RopeInterleaved, cfg.RopeTheta, 1.0, positions)
	return k, v
}

func (a *Attention) splitKV(kv *mlx.Array, B, L int32, cfg *Config) (k, v *mlx.Array) {
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim
	k = mlx.Reshape(mlx.SliceStartStop(kv, []int32{0, 0, 0}, []int32{B, L, kvDim}), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v = mlx.Reshape(mlx.SliceStartStop(kv, []int32{0, 0, kvDim}, []int32{B, L, 2 * kvDim}), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	return k, v
}

func (a *Attention) Forward(x, ctxK, ctxV *mlx.Array, c cache.Cache, bb *batch.Batch, positions *mlx.Array, mask nn.AttentionMask, B, L int32, cfg *Config) *mlx.Array {
	q := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k, v := a.splitKV(a.KVProj.Forward(x), B, L, cfg)

	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), cfg.RopeInterleaved, cfg.RopeTheta, 1.0, positions)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), cfg.RopeInterleaved, cfg.RopeTheta, 1.0, positions)

	if ctxK != nil {
		k = ctxK.Concatenate(2, k)
		v = ctxV.Concatenate(2, v)
	}

	// Write the context and block K/V together: a rollback point between two
	// writes would force a wrapped rotating cache to copy its window out.
	hist := c.(cache.Attention).Update(bb, k, v)

	out := nn.ScaledDotProductAttention(bb, q, cfg.Scale, nn.WithKVHistory(hist), nn.WithMask(mask))
	if a.GProj != nil {
		// Per-head softplus output gate, applied before the head merge.
		gate := mlx.ExpandDims(mlx.SoftplusF32(a.GProj.Forward(x)), -1)
		out = mlx.Mul(mlx.Transpose(out, 0, 2, 1, 3), gate)
		out = mlx.Reshape(out, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	} else {
		out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	}
	return a.OProj.Forward(out)
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gu := m.GateUpProj.Forward(x)
	dims := gu.Dims()
	B, L, half := int32(dims[0]), int32(dims[1]), int32(dims[2])/2
	gate := mlx.SliceStartStop(gu, []int32{0, 0, 0}, []int32{B, L, half})
	up := mlx.SliceStartStop(gu, []int32{0, 0, half}, []int32{B, L, 2 * half})
	return m.DownProj.Forward(mlx.SwiGLU(gate, up))
}
