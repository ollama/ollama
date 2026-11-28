// Package olala implements OVHai's Olala v1.0 (OlalaForCausalLM) for MLX.
//
// Hybrid model: Mamba-3 MIMO layers ("M") and differential tensor-product
// attention layers ("V"), a 256-expert latent MoE on every layer, and a
// geodesic (rotation) residual update instead of pre-norms.
//
// Ported from the plain-MLX reference path of OVHai's mlx-engine fork
// (https://github.com/OVHai-LLM/mlx-engine, MIT license:
// mlx_engine/models/olala/model.py and kernels.py), which expects the
// checkpoint layout produced by its converter: projections that read the same
// input are already merged (mixer.in_proj, mixer.in_proj_small, mlp.in_cat,
// mlp.out_cat) and the routed experts are stacked.
package olala

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("OlalaForCausalLM", newModel)
}

// m3Chunk is the chunk length (tokens) of the chunked Mamba-3 prefill.
const m3Chunk = 32

// m3MaxPiece is the largest number of tokens one chunked Mamba-3 call handles.
const m3MaxPiece = 256

// attnScoreBudget bounds the f32 attention scores materialized at once
// (bytes). A variable so tests can exercise query chunking on small models.
var attnScoreBudget = 128 * 1024 * 1024

type Config struct {
	VocabSize             int32   `json:"vocab_size"`
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	LayersConfig          string  `json:"layers_config"`
	NormEpsilon           float32 `json:"norm_epsilon"`
	ZeroCenteredGamma     bool    `json:"zero_centered_gamma"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumSignalHeadsDiff    int32   `json:"num_signal_heads_diff"`
	HeadDim               int32   `json:"head_dim"`
	TPARank               int32   `json:"tpa_rank"`
	SoftcapAttn           float32 `json:"softcap_attn"`
	SlwWsize              int32   `json:"slw_wsize"`
	UseCompletedP         bool    `json:"use_completed_p"`
	ZeroCenteredGate      bool    `json:"zero_centered_gate"`
	MambaDState           int32   `json:"mamba_d_state"`
	MambaHeadDim          int32   `json:"mamba_headdim"`
	MambaMimoDim          int32   `json:"mamba_mimo_dim"`
	MambaNGroups          int32   `json:"mamba_ngroups"`
	NumRoutedExperts      int32   `json:"moe_num_routed_experts"`
	NumActiveExperts      int32   `json:"moe_num_active_experts"`
	RoutedInputDim        int32   `json:"moe_routed_input_dim"`
	RoutedScalingFactor   float32 `json:"moe_routed_scaling_factor"`
	NormalizeEmbedNGPT    bool    `json:"normalize_embeddings_ngpt"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`

	// derived
	dInner, mH, mP, mN, mR, mA int32
	nStatic                    int32
	nSig, nNoise, snr          int32
	attnScale, gateBias        float32
}

func parseConfig(data []byte) (Config, error) {
	cfg := Config{
		HiddenSize:          1536,
		NormEpsilon:         1e-6,
		ZeroCenteredGamma:   true,
		NumAttentionHeads:   48,
		NumSignalHeadsDiff:  36,
		HeadDim:             128,
		TPARank:             4,
		SoftcapAttn:         150,
		SlwWsize:            32768,
		UseCompletedP:       true,
		ZeroCenteredGate:    true,
		MambaDState:         128,
		MambaHeadDim:        64,
		MambaMimoDim:        4,
		MambaNGroups:        1,
		NumRoutedExperts:    256,
		NumActiveExperts:    6,
		RoutedInputDim:      384,
		RoutedScalingFactor: 2.45,
		NormalizeEmbedNGPT:  true,
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	if strings.Trim(cfg.LayersConfig, "MV") != "" {
		return cfg, fmt.Errorf("olala: unsupported layer types in %q", cfg.LayersConfig)
	}
	if int32(len(cfg.LayersConfig)) != cfg.NumHiddenLayers {
		return cfg, fmt.Errorf("olala: layers_config does not match num_hidden_layers")
	}
	if cfg.MambaNGroups != 1 {
		return cfg, fmt.Errorf("olala: mamba_ngroups != 1 is not supported")
	}

	cfg.dInner = 2 * cfg.HiddenSize
	cfg.mP = cfg.MambaHeadDim
	cfg.mH = cfg.dInner / cfg.mP
	cfg.mN = cfg.MambaDState
	cfg.mR = cfg.MambaMimoDim
	split := cfg.mN / 2
	split -= split % 2
	cfg.mA = split / 2
	cfg.nStatic = 2*cfg.dInner + 3*cfg.mH

	cfg.nSig = cfg.NumSignalHeadsDiff
	if cfg.nSig == 0 {
		cfg.nSig = cfg.NumAttentionHeads / 2
	}
	cfg.nNoise = cfg.NumAttentionHeads - cfg.nSig
	cfg.snr = cfg.nSig / cfg.nNoise
	if cfg.UseCompletedP {
		cfg.attnScale = 1 / float32(cfg.HeadDim)
	} else {
		cfg.attnScale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	}
	if cfg.ZeroCenteredGate {
		cfg.gateBias = 1.15
	}
	return cfg, nil
}

type Model struct {
	cfg    *Config
	tok    *tokenizer.Tokenizer
	Embed  nn.EmbeddingLayer
	LmHead nn.LinearLayer
	Layers []*Layer
	// shiftSlot[i] is the cache index of attention layer i's token-shift state
	shiftSlot []int
	// Fast holds the weights of the fused decode path, nil when unavailable.
	Fast []*fastLayer
}

type Layer struct {
	typ       byte
	idx       int
	Mamba     *Mamba3
	Attn      *DiffTPAttention
	MixerProj nn.LinearLayer
	GeoMixS   *mlx.Array
	GeoMixB   *mlx.Array
	GeoMlpS   *mlx.Array
	GeoMlpB   *mlx.Array
	Moe       *MoE
}

type Mamba3 struct {
	InProj nn.LinearLayer
	BBias  *mlx.Array // (H,R,N) f32
	CBias  *mlx.Array
	WB, WC *mlx.Array // (N) f32, zero-centered applied
	MimoX  *mlx.Array // (H,R,P) f32
	MimoZ  *mlx.Array
	OutMim *mlx.Array
	DtBias *mlx.Array // (H) f32
	D      *mlx.Array // (H) f32
}

type DiffTPAttention struct {
	InProj      nn.LinearLayer
	InProjSmall nn.LinearLayer
	QNorm       *mlx.Array // (Dh), zero-centered applied, f32
	KNorm       *mlx.Array
	Scaler      *mlx.Array // (Hq) f32
}

type MoE struct {
	GateW      *mlx.Array // (E, D) f32
	ExpertBias *mlx.Array // (E) f32
	InCat      nn.LinearLayer
	OutCat     nn.LinearLayer
	Up, Down   expertWeights
}

type expertWeights struct {
	// Raw is the stacked (E, out, in) weight as stored; W is what GatherMM /
	// GatherQMM read.
	Raw               *mlx.Array
	W, Scales, Biases *mlx.Array
	GroupSize, Bits   int
	Mode              string
	Quantized         bool
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
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant = root.Quantization()
	tok, err := root.LoadTokenizer(configData)
	if err != nil {
		return nil, err
	}

	return newModelFromConfig(cfg, tok), nil
}

func newModelFromConfig(cfg Config, tok *tokenizer.Tokenizer) *Model {
	m := &Model{cfg: &cfg, tok: tok}
	m.Layers = make([]*Layer, cfg.NumHiddenLayers)
	m.shiftSlot = make([]int, cfg.NumHiddenLayers)
	next := int(cfg.NumHiddenLayers)
	for i := range m.Layers {
		m.Layers[i] = &Layer{typ: cfg.LayersConfig[i], idx: i}
		m.shiftSlot[i] = -1
		if cfg.LayersConfig[i] == 'V' {
			m.shiftSlot[i] = next
			next++
		}
	}
	return m
}

func f32(a *mlx.Array) *mlx.Array {
	if a == nil {
		return nil
	}
	return a.AsType(mlx.DTypeFloat32)
}

func zc(w *mlx.Array, on bool) *mlx.Array {
	w = f32(w)
	if w != nil && on {
		return mlx.AddScalar(w, 1)
	}
	return w
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := m.cfg
	linears := model.NewLinearFactory(tensors, cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)

	need := func(name string) (*mlx.Array, error) {
		t := tensors[name]
		if t == nil {
			return nil, fmt.Errorf("missing tensor %s", name)
		}
		return t, nil
	}
	needLinear := func(path string) (nn.LinearLayer, error) {
		l := linears.Make(path)
		if l == nil {
			return nil, fmt.Errorf("missing linear %s", path)
		}
		return l, nil
	}

	m.Embed = model.MakeEmbeddingLayer(tensors, "model.embedding", cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant)
	if m.Embed == nil {
		return fmt.Errorf("missing model.embedding.weight")
	}
	if cfg.TieWordEmbeddings {
		if e, ok := m.Embed.(interface{ AsLinear() nn.LinearLayer }); ok {
			m.LmHead = e.AsLinear()
		}
	} else {
		m.LmHead = linears.Make("lm_head")
	}
	if m.LmHead == nil {
		return fmt.Errorf("missing lm_head.weight")
	}

	var err error
	for i, l := range m.Layers {
		p := fmt.Sprintf("model.layers.%d", i)
		get := func(name string) *mlx.Array {
			if err != nil {
				return nil
			}
			var t *mlx.Array
			t, err = need(p + "." + name)
			return t
		}
		lin := func(name string) nn.LinearLayer {
			if err != nil {
				return nil
			}
			var t nn.LinearLayer
			t, err = needLinear(p + "." + name)
			return t
		}

		l.MixerProj = lin("mixer_proj")
		// The runner remaps "<x>.scale" / "<x>.bias" pairs to quantization
		// companions "<x>_scale" / "<x>_qbias".
		l.GeoMixS = f32(get("geodesic_mixer_scale"))
		l.GeoMixB = f32(get("geodesic_mixer_qbias"))
		l.GeoMlpS = f32(get("geodesic_mlp_scale"))
		l.GeoMlpB = f32(get("geodesic_mlp_qbias"))

		if l.typ == 'M' {
			l.Mamba = &Mamba3{
				InProj: lin("mixer.in_proj"),
				BBias:  f32(get("mixer.B_bias")),
				CBias:  f32(get("mixer.C_bias")),
				WB:     zc(get("mixer.B_norm.norm.weight"), cfg.ZeroCenteredGamma),
				WC:     zc(get("mixer.C_norm.norm.weight"), cfg.ZeroCenteredGamma),
				MimoX:  f32(get("mixer.in_proj_mimo_x")),
				MimoZ:  f32(get("mixer.in_proj_mimo_z")),
				OutMim: f32(get("mixer.out_proj_mimo")),
				DtBias: f32(get("mixer.dt_bias")),
				D:      f32(get("mixer.D")),
			}
		} else {
			l.Attn = &DiffTPAttention{
				InProj:      lin("mixer.in_proj"),
				InProjSmall: lin("mixer.in_proj_small"),
				QNorm:       zc(get("mixer.q_norm.norm.weight"), cfg.ZeroCenteredGamma),
				KNorm:       zc(get("mixer.k_norm.norm.weight"), cfg.ZeroCenteredGamma),
				Scaler:      f32(get("mixer.softmax_scaler")),
			}
		}

		l.Moe = &MoE{
			GateW:      f32(get("mlp.moe_gate.weight")),
			ExpertBias: f32(get("mlp.expert_bias")),
			InCat:      lin("mlp.in_cat"),
			OutCat:     lin("mlp.out_cat"),
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		for _, name := range []string{p + ".mlp.experts.experts.weight", p + ".mlp.experts.output_experts.weight"} {
			if tensors[name] == nil {
				return fmt.Errorf("layer %d: missing tensor %s", i, name)
			}
		}
		l.Moe.Up = loadExperts(tensors, cfg, p+".mlp.experts.experts.weight")
		l.Moe.Down = loadExperts(tensors, cfg, p+".mlp.experts.output_experts.weight")
	}
	m.Fast = m.buildFast()
	return nil
}

// loadExperts reads a stacked (E, out, in) expert matrix. Quantized affine
// 4/8-bit weights stay packed for GatherQMM; anything else is dequantized and
// transposed to (E, in, out) for GatherMM.
func loadExperts(tensors map[string]*mlx.Array, cfg *Config, key string) expertWeights {
	w := tensors[key]
	if w == nil {
		return expertWeights{}
	}
	scales := tensors[key+"_scale"]
	if scales == nil {
		return expertWeights{Raw: w, W: model.TransposeExpertWeightForGatherMM(w)}
	}
	qbias := tensors[key+"_qbias"]
	groupSize, bits, mode := model.ResolveLinearQuantParams(cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant, key, w, scales)
	if model.SupportsGatherQMM(mode, bits) {
		return expertWeights{Raw: w, W: w, Scales: scales, Biases: qbias, GroupSize: groupSize, Bits: bits, Mode: mode, Quantized: true}
	}
	deq := mlx.Dequantize(w, scales, qbias, groupSize, bits, mode, nil)
	return expertWeights{W: model.TransposeExpertWeightForGatherMM(deq)}
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// MaxContextLength is capped at the attention window: the reference path does
// not implement the sliding window beyond it.
func (m *Model) MaxContextLength() int {
	n := int(m.cfg.SlwWsize)
	if mp := int(m.cfg.MaxPositionEmbeddings); mp > 0 && mp < n {
		n = mp
	}
	return n
}

// NewCaches returns one slot per layer, then one token-shift slot per
// attention layer. M layers keep their state in a RecurrentCache whose delta
// state (B,H,1,SW) packs, per head, [SSM state S (P*N) | rotated keys of the
// last token (R*N) | raw x of the last token (P) | rotary angles (A)], the
// layout the fused decode kernel reads. Token-shift slots keep the previous
// token's raw k|v in the conv state.
func (m *Model) NewCaches() []cache.Cache {
	cfg := m.cfg
	caches := make([]cache.Cache, 0, len(m.Layers)+8)
	for _, l := range m.Layers {
		if l.typ == 'M' {
			caches = append(caches, cache.NewRecurrentCache(1, 1, cfg.mH, 1, cfg.mP*cfg.mN+cfg.mR*cfg.mN+cfg.mP+cfg.mA))
		} else {
			caches = append(caches, cache.NewKVCache())
		}
	}
	for _, l := range m.Layers {
		if l.typ == 'V' {
			caches = append(caches, cache.NewRecurrentCache(1, 2*cfg.nNoise*cfg.HeadDim, 1, 1, 1))
		}
	}
	return caches
}

func (m *Model) computeDType() mlx.DType {
	return mlx.DTypeBFloat16
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Layer) (hidden, auxHidden *mlx.Array) {
	cfg := m.cfg
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	dtype := m.computeDType()

	if m.canDecodeFused(b, caches) {
		h := m.decodeFused(b, caches)
		return h, h
	}

	x := f32(m.Embed.Forward(b.InputIDs))
	if cfg.NormalizeEmbedNGPT {
		n := mlx.Maximum(mlx.Sum(mlx.Mul(x, x), -1, true).Sqrt(), mlx.FromValue(float32(1e-12)))
		x = mlx.Div(x, n)
	}

	pos := make([]float32, 0, B*L)
	for i := range B {
		off := int32(0)
		if int(i) < len(b.SeqOffsets) {
			off = b.SeqOffsets[i]
		}
		for j := range L {
			pos = append(pos, float32(off+j))
		}
	}
	positions := mlx.FromValues(pos, int(B), int(L))

	slot := func(i int) cache.Layer {
		if caches == nil || i < 0 || i >= len(caches) {
			return nil
		}
		return caches[i]
	}

	for i, l := range m.Layers {
		h := x.AsType(dtype)
		var y *mlx.Array
		if l.typ == 'M' {
			y = l.Mamba.forward(h, b, slot(i), cfg)
		} else {
			y = l.Attn.forward(h, b, slot(i), slot(m.shiftSlot[i]), positions, cfg)
		}
		y = l.MixerProj.Forward(y.AsType(dtype))
		x = geodesic(x, y, l.GeoMixS, l.GeoMixB, l.idx)
		y = l.Moe.forward(x.AsType(dtype), cfg)
		x = geodesic(x, y, l.GeoMlpS, l.GeoMlpB, l.idx)
	}
	return x, x
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.LmHead.Forward(x.AsType(m.computeDType()))
}

// ---------------------------------------------------------------------------
// small helpers

func scalar(v float32) *mlx.Array { return mlx.FromValue(v) }

// sliceLast slices the last axis to [start, stop).
func sliceLast(a *mlx.Array, start, stop int) *mlx.Array {
	return sliceAxis(a, a.NumDims()-1, start, stop)
}

func sliceAxis(a *mlx.Array, axis, start, stop int) *mlx.Array {
	dims := a.Dims()
	starts := make([]int32, len(dims))
	stops := make([]int32, len(dims))
	for i, d := range dims {
		stops[i] = int32(d)
	}
	starts[axis], stops[axis] = int32(start), int32(stop)
	return mlx.SliceStartStop(a, starts, stops)
}

func swapLast(a *mlx.Array) *mlx.Array {
	n := a.NumDims()
	axes := make([]int, n)
	for i := range axes {
		axes[i] = i
	}
	axes[n-1], axes[n-2] = axes[n-2], axes[n-1]
	return mlx.Transpose(a, axes...)
}

func silu(a *mlx.Array) *mlx.Array { return mlx.Mul(a, mlx.Sigmoid(a)) }

// wrap brings angles back to [-pi, pi].
func wrap(theta *mlx.Array) *mlx.Array {
	twoPi := float32(2 * math.Pi)
	r := mlx.AddScalar(mlx.DivScalar(theta, twoPi), 0.5).FloorDivide(scalar(1))
	return mlx.Sub(theta, mlx.MulScalar(r, twoPi))
}

// rotate rotates pairs (i, i+2A) for i < A along the last axis.
func rotate(t, cos, sin *mlx.Array, A int) *mlx.Array {
	n := t.Dim(t.NumDims() - 1)
	t0 := sliceLast(t, 0, A)
	t1 := sliceLast(t, A, 2*A)
	t2 := sliceLast(t, 2*A, 3*A)
	t3 := sliceLast(t, 3*A, n)
	r0 := mlx.Sub(mlx.Mul(t0, cos), mlx.Mul(t2, sin))
	r2 := mlx.Add(mlx.Mul(t0, sin), mlx.Mul(t2, cos))
	return mlx.Concatenate([]*mlx.Array{r0, t1, r2, t3}, -1)
}

// geodesic rotates the residual x towards the update g (OlalaGeodesicNorm), in f32.
func geodesic(x, g, scale, bias *mlx.Array, layerIdx int) *mlx.Array {
	g = f32(g)
	xn2 := mlx.Maximum(mlx.Sum(mlx.Mul(x, x), -1, true), scalar(1e-12))
	proj := mlx.Div(mlx.Sum(mlx.Mul(x, g), -1, true), xn2)
	grad := mlx.Sub(g, mlx.Mul(proj, x))
	tn := mlx.Maximum(mlx.Sum(mlx.Mul(grad, grad), -1, true).Sqrt(), scalar(1e-8))
	r := mlx.Maximum(xn2.Sqrt(), scalar(1e-6))
	clamp := scalar(math.Pi / 4)
	theta := mlx.Minimum(mlx.Div(tn, r), clamp)
	theta = mlx.Minimum(mlx.DivScalar(mlx.Add(mlx.Mul(theta, scale), bias), float32(layerIdx+1)), clamp)
	return mlx.Add(mlx.Mul(x, mlx.Cos(theta)), mlx.Mul(mlx.Mul(grad, mlx.Div(r, tn)), mlx.Sin(theta)))
}

// segments cuts a forward of length L at the recurrent cache's snapshot splits.
func segments(c cache.Layer, L int) []int {
	bounds := []int{0}
	if rc, ok := c.(cache.Recurrent); ok {
		bounds = append(bounds, rc.SnapshotSplits(L)...)
	}
	return append(bounds, L)
}

// ---------------------------------------------------------------------------
// Mamba-3 MIMO

type m3State struct {
	theta *mlx.Array // (B,H,A)
	s     *mlx.Array // (B,H,P,N)
	k     *mlx.Array // (B,H,R,N)
	x     *mlx.Array // (B,H,P)
}

type m3Inputs struct {
	x, z         *mlx.Array // (B,L,H,P)
	dt, la, trap *mlx.Array // (B,L,H)
	bn, cn       *mlx.Array // (B,L,R,N)
	angInc       *mlx.Array // (B,L,H,A)
}

func (m *Mamba3) forward(h *mlx.Array, b *batch.Batch, c cache.Layer, cfg *Config) *mlx.Array {
	dims := h.Dims()
	B, L := dims[0], dims[1]
	H, R, N, P, A := int(cfg.mH), int(cfg.mR), int(cfg.mN), int(cfg.mP), int(cfg.mA)

	out := m.InProj.Forward(h)
	proj := sliceLast(out, 0, int(cfg.nStatic))
	dyn := sliceLast(out, int(cfg.nStatic), out.Dim(2))

	var st m3State
	var conv *mlx.Array
	rc, hasCache := c.(cache.Recurrent)
	if hasCache {
		hist := rc.Get(b, mlx.DTypeFloat32)
		conv = hist.ConvState()
		st = unpackM3(hist.DeltaState(), cfg)
	} else {
		st = m3State{
			theta: mlx.Zeros(mlx.DTypeFloat32, B, H, A),
			s:     mlx.Zeros(mlx.DTypeFloat32, B, H, P, N),
			k:     mlx.Zeros(mlx.DTypeFloat32, B, H, R, N),
			x:     mlx.Zeros(mlx.DTypeFloat32, B, H, P),
		}
	}

	in := m.prep(proj, dyn, cfg)
	bounds := segments(c, L)
	var ys []*mlx.Array
	var convStates, deltaStates []*mlx.Array
	for i := 0; i+1 < len(bounds); i++ {
		s0, s1 := bounds[i], bounds[i+1]
		seg := in
		if s0 != 0 || s1 != L {
			seg = in.slice(s0, s1)
		}
		if s1-s0 == 1 {
			var y *mlx.Array
			y, st = m.step(seg, st, cfg)
			ys = append(ys, y)
		} else {
			// Bound the (L/T) x (T*R)^2 intermediate scores: long prompts run
			// as consecutive pieces of at most m3MaxPiece tokens.
			for p0 := 0; p0 < s1-s0; p0 += m3MaxPiece {
				p1 := min(s1-s0, p0+m3MaxPiece)
				piece := seg
				if p0 != 0 || p1 != s1-s0 {
					piece = seg.slice(p0, p1)
				}
				var y *mlx.Array
				if p1-p0 == 1 {
					y, st = m.step(piece, st, cfg)
				} else {
					y, st = m.chunked(piece, st, cfg)
				}
				ys = append(ys, y)
			}
		}
		if hasCache {
			convStates = append(convStates, conv)
			deltaStates = append(deltaStates, packM3(st, cfg))
		}
	}
	if hasCache {
		rc.Put(b, convStates, deltaStates)
	}
	y := ys[0]
	if len(ys) > 1 {
		y = mlx.Concatenate(ys, 1)
	}
	return y.AsType(h.DType())
}

// m3StateWidth is the packed per-head state width.
func m3StateWidth(cfg *Config) int {
	return int(cfg.mP*cfg.mN + cfg.mR*cfg.mN + cfg.mP + cfg.mA)
}

// unpackM3 splits a packed state (B,H,1,SW) into its parts.
func unpackM3(packed *mlx.Array, cfg *Config) m3State {
	B := int32(packed.Dim(0))
	H, R, N, P := cfg.mH, cfg.mR, cfg.mN, cfg.mP
	st := mlx.Reshape(packed, B, H, int32(m3StateWidth(cfg)))
	o1, o2, o3 := int(P*N), int(P*N+R*N), int(P*N+R*N+P)
	return m3State{
		s:     mlx.Reshape(sliceLast(st, 0, o1), B, H, P, N),
		k:     mlx.Reshape(sliceLast(st, o1, o2), B, H, R, N),
		x:     sliceLast(st, o2, o3),
		theta: sliceLast(st, o3, st.Dim(2)),
	}
}

// packM3 is the inverse of unpackM3.
func packM3(st m3State, cfg *Config) *mlx.Array {
	B := int32(st.s.Dim(0))
	H := cfg.mH
	packed := mlx.Concatenate([]*mlx.Array{
		mlx.Reshape(st.s, B, H, cfg.mP*cfg.mN),
		mlx.Reshape(st.k, B, H, cfg.mR*cfg.mN),
		st.x,
		st.theta,
	}, -1)
	return mlx.Reshape(packed, B, H, 1, int32(m3StateWidth(cfg)))
}

func (in m3Inputs) slice(s0, s1 int) m3Inputs {
	return m3Inputs{
		x: sliceAxis(in.x, 1, s0, s1), z: sliceAxis(in.z, 1, s0, s1),
		dt: sliceAxis(in.dt, 1, s0, s1), la: sliceAxis(in.la, 1, s0, s1), trap: sliceAxis(in.trap, 1, s0, s1),
		bn: sliceAxis(in.bn, 1, s0, s1), cn: sliceAxis(in.cn, 1, s0, s1),
		angInc: sliceAxis(in.angInc, 1, s0, s1),
	}
}

// prep splits the projections and discretizes, in f32.
func (m *Mamba3) prep(proj, dyn *mlx.Array, cfg *Config) m3Inputs {
	dims := proj.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	H, R, N, P := cfg.mH, cfg.mR, cfg.mN, int(cfg.mP)

	ph := f32(mlx.Reshape(proj, B, L, H, 2*int32(P)+3))
	z := sliceLast(ph, 0, P)
	x := sliceLast(ph, P, 2*P)
	dt := mlx.Softplus(mlx.Add(mlx.Squeeze(sliceLast(ph, 2*P, 2*P+1), -1), m.DtBias))
	a := mlx.Minimum(mlx.Neg(mlx.Softplus(mlx.Squeeze(sliceLast(ph, 2*P+1, 2*P+2), -1))), scalar(-1e-4))
	trap := mlx.Sigmoid(mlx.Squeeze(sliceLast(ph, 2*P+2, 2*P+3), -1))

	dynF := f32(dyn)
	rn := int(2 * R * N)
	bc := mlx.Reshape(sliceLast(dynF, 0, rn), B, L, 2*R, N)
	bc = mlx.Mul(bc, mlx.RSqrt(mlx.AddScalar(mlx.Mean(mlx.Mul(bc, bc), -1, true), cfg.NormEpsilon)))
	bn := mlx.Mul(sliceAxis(bc, 2, 0, int(R)), m.WB)
	cn := mlx.Mul(sliceAxis(bc, 2, int(R), int(2*R)), m.WC)
	ang := mlx.MulScalar(sliceLast(dynF, rn, dynF.Dim(2)).Tanh(), math.Pi) // (B,L,A)
	angInc := mlx.Mul(mlx.ExpandDims(ang, 2), mlx.ExpandDims(dt, -1))      // (B,L,H,A)

	return m3Inputs{x: x, z: z, dt: dt, la: mlx.Mul(a, dt), trap: trap, bn: bn, cn: cn, angInc: angInc}
}

// step processes one token per row.
func (m *Mamba3) step(in m3Inputs, st m3State, cfg *Config) (*mlx.Array, m3State) {
	B := in.x.Dim(0)
	H, P, A := int32(cfg.mH), int32(cfg.mP), int(cfg.mA)

	x := mlx.Squeeze(in.x, 1)       // (B,H,P)
	z := mlx.Squeeze(in.z, 1)       // (B,H,P)
	dt := mlx.Squeeze(in.dt, 1)     // (B,H)
	la := mlx.Squeeze(in.la, 1)     // (B,H)
	trap := mlx.Squeeze(in.trap, 1) // (B,H)

	theta := wrap(mlx.Add(st.theta, mlx.Squeeze(in.angInc, 1))) // (B,H,A)
	cos := mlx.ExpandDims(mlx.Cos(theta), 2)
	sin := mlx.ExpandDims(mlx.Sin(theta), 2)
	k := rotate(mlx.Add(mlx.ExpandDims(mlx.Squeeze(in.bn, 1), 1), m.BBias), cos, sin, A) // (B,H,R,N)
	q := rotate(mlx.Add(mlx.ExpandDims(mlx.Squeeze(in.cn, 1), 1), m.CBias), cos, sin, A)

	v := mlx.Mul(mlx.ExpandDims(x, 2), m.MimoX)        // (B,H,R,P)
	vPrev := mlx.Mul(mlx.ExpandDims(st.x, 2), m.MimoX) // (B,H,R,P)
	w := mlx.Mul(silu(mlx.Mul(mlx.ExpandDims(z, 2), m.MimoZ)), m.OutMim)

	alpha := mlx.Exp(la)
	gamma := mlx.Mul(trap, dt)
	beta := mlx.Mul(mlx.Mul(mlx.Sub(scalar(1), trap), dt), alpha)
	e2 := func(a *mlx.Array) *mlx.Array { return mlx.ExpandDims(mlx.ExpandDims(a, -1), -1) }

	u := mlx.Matmul(swapLast(v), k)            // (B,H,P,N)
	uPrev := mlx.Matmul(swapLast(vPrev), st.k) // (B,H,P,N)
	S := mlx.Add(mlx.Add(mlx.Mul(e2(alpha), st.s), mlx.Mul(e2(beta), uPrev)), mlx.Mul(e2(gamma), u))
	sq := swapLast(mlx.Matmul(S, swapLast(q))) // (B,H,R,P)

	d := mlx.Reshape(m.D, 1, H, 1, 1)
	y := mlx.Sum(mlx.Mul(w, mlx.Add(mlx.Mul(d, v), sq)), 2, false) // (B,H,P)
	y = mlx.Reshape(y, int32(B), 1, H*P)
	return y, m3State{theta: theta, s: S, k: k, x: x}
}

// chunked processes L > 1 tokens per row in chunks of m3Chunk.
func (m *Mamba3) chunked(in m3Inputs, st m3State, cfg *Config) (*mlx.Array, m3State) {
	dims := in.x.Dims()
	B, L := dims[0], dims[1]
	H, R, N, P, A := int(cfg.mH), int(cfg.mR), int(cfg.mN), int(cfg.mP), int(cfg.mA)
	T := m3Chunk
	i32 := func(v ...int) []int32 {
		out := make([]int32, len(v))
		for i, x := range v {
			out[i] = int32(x)
		}
		return out
	}
	reshape := func(a *mlx.Array, shape ...int) *mlx.Array { return mlx.Reshape(a, i32(shape...)...) }

	theta := wrap(mlx.Add(mlx.ExpandDims(st.theta, 1), in.angInc.Cumsum(1, false, true))) // (B,L,H,A)
	cos := mlx.ExpandDims(mlx.Cos(theta), 3)
	sin := mlx.ExpandDims(mlx.Sin(theta), 3)
	k := rotate(mlx.Add(mlx.ExpandDims(in.bn, 2), m.BBias), cos, sin, A) // (B,L,H,R,N)
	q := rotate(mlx.Add(mlx.ExpandDims(in.cn, 2), m.CBias), cos, sin, A)
	v := mlx.Mul(mlx.ExpandDims(in.x, 3), m.MimoX) // (B,L,H,R,P)
	w := mlx.Mul(silu(mlx.Mul(mlx.ExpandDims(in.z, 3), m.MimoZ)), m.OutMim)
	gamma := mlx.Mul(in.trap, in.dt)                     // (B,L,H)
	delta := mlx.Mul(mlx.Sub(scalar(1), in.trap), in.dt) // (B,L,H)
	la := in.la

	Lp := ((L + T - 1) / T) * T
	if pad := Lp - L; pad > 0 {
		padL := func(a *mlx.Array) *mlx.Array {
			shape := a.Dims()
			shape[1] = pad
			return mlx.Concatenate([]*mlx.Array{a, mlx.Zeros(a.DType(), shape...)}, 1)
		}
		k, q, v, w, la, gamma, delta = padL(k), padL(q), padL(v), padL(w), padL(la), padL(gamma), padL(delta)
	}
	Nc := Lp / T

	k = mlx.Transpose(reshape(k, B, Nc, T, H, R, N), 0, 3, 1, 2, 4, 5) // (B,H,Nc,T,R,N)
	q = mlx.Transpose(reshape(q, B, Nc, T, H, R, N), 0, 3, 1, 2, 4, 5)
	v = mlx.Transpose(reshape(v, B, Nc, T, H, R, P), 0, 3, 1, 2, 4, 5)
	w = mlx.Transpose(reshape(w, B, Nc, T, H, R, P), 0, 3, 1, 2, 4, 5)
	la = mlx.Transpose(reshape(la, B, Nc, T, H), 0, 3, 1, 2) // (B,H,Nc,T)
	gamma = mlx.Transpose(reshape(gamma, B, Nc, T, H), 0, 3, 1, 2)
	delta = mlx.Transpose(reshape(delta, B, Nc, T, H), 0, 3, 1, 2)

	cum := la.Cumsum(-1, false, true) // (B,H,Nc,T)
	deltaNext := mlx.Concatenate([]*mlx.Array{sliceLast(delta, 1, T), mlx.Zeros(mlx.DTypeFloat32, B, H, Nc, 1)}, -1)
	g := mlx.Add(gamma, deltaNext)

	// Mx[t, s] = exp(cum_t - cum_s) * g_s for s < t, gamma_t on the diagonal
	seg := mlx.Sub(mlx.ExpandDims(cum, -1), mlx.ExpandDims(cum, -2)) // (B,H,Nc,T,T)
	tri := mlx.Tri(int32(T), int32(T), -1)
	eye := mlx.Sub(mlx.Tri(int32(T), int32(T), 0), tri)
	Mx := mlx.Mul(mlx.Mul(tri, mlx.Exp(mlx.Mul(seg, tri))), mlx.ExpandDims(g, -2))
	Mx = mlx.Add(Mx, mlx.Mul(eye, mlx.ExpandDims(gamma, -1)))

	kf := reshape(k, B, H, Nc, T*R, N)
	qf := reshape(q, B, H, Nc, T*R, N)
	vf := reshape(v, B, H, Nc, T*R, P)
	scores := reshape(mlx.Matmul(qf, swapLast(kf)), B, H, Nc, T, R, T, R)
	scores = mlx.Mul(scores, mlx.ExpandDims(mlx.ExpandDims(Mx, 4), 6))
	intra := mlx.Matmul(reshape(scores, B, H, Nc, T*R, T*R), vf) // (B,H,Nc,TR,P)

	// per-chunk state increments
	e := mlx.Mul(mlx.Exp(mlx.Sub(sliceLast(cum, T-1, T), cum)), g) // (B,H,Nc,T)
	e = mlx.Concatenate([]*mlx.Array{sliceLast(e, 0, T-1), sliceLast(gamma, T-1, T)}, -1)
	ve := reshape(mlx.Mul(v, mlx.ExpandDims(mlx.ExpandDims(e, -1), -1)), B, H, Nc, T*R, P)
	dS := mlx.Matmul(swapLast(ve), kf)                     // (B,H,Nc,P,N)
	aC := mlx.Exp(mlx.Squeeze(sliceLast(cum, T-1, T), -1)) // (B,H,Nc)
	vLast := mlx.Squeeze(sliceAxis(v, 3, T-1, T), 3)       // (B,H,Nc,R,P)
	kLast := mlx.Squeeze(sliceAxis(k, 3, T-1, T), 3)       // (B,H,Nc,R,N)
	uLast := mlx.Matmul(swapLast(vLast), kLast)            // (B,H,Nc,P,N)
	vPrev0 := mlx.Mul(mlx.ExpandDims(st.x, 2), m.MimoX)    // (B,H,R,P)
	uPrev0 := mlx.Matmul(swapLast(vPrev0), st.k)           // (B,H,P,N)

	e2 := func(a *mlx.Array) *mlx.Array { return mlx.ExpandDims(mlx.ExpandDims(a, -1), -1) }
	S := st.s
	inits := make([]*mlx.Array, Nc)
	for c := range Nc {
		uPrev := uPrev0
		if c > 0 {
			uPrev = mlx.Squeeze(sliceAxis(uLast, 2, c-1, c), 2)
		}
		dc := mlx.Squeeze(mlx.Squeeze(sliceAxis(sliceLast(delta, 0, 1), 2, c, c+1), -1), -1) // (B,H)
		sEff := mlx.Add(S, mlx.Mul(e2(dc), uPrev))
		inits[c] = sEff
		ac := mlx.Squeeze(sliceLast(aC, c, c+1), -1) // (B,H)
		S = mlx.Add(mlx.Mul(e2(ac), sEff), mlx.Squeeze(sliceAxis(dS, 2, c, c+1), 2))
	}
	sInit := mlx.Stack(inits, 2) // (B,H,Nc,P,N)
	expCum := reshape(mlx.BroadcastTo(mlx.ExpandDims(mlx.Exp(cum), -1), i32(B, H, Nc, T, R)...), B, H, Nc, T*R)
	inter := mlx.Mul(mlx.Matmul(qf, swapLast(sInit)), mlx.ExpandDims(expCum, -1))
	outC := reshape(mlx.Add(intra, inter), B, H, Nc, T, R, P)

	d := mlx.Reshape(m.D, 1, int32(H), 1, 1, 1, 1)
	y := mlx.Sum(mlx.Mul(w, mlx.Add(mlx.Mul(d, v), outC)), 4, false) // (B,H,Nc,T,P)
	y = reshape(mlx.Transpose(y, 0, 2, 3, 1, 4), B, Lp, H*P)
	if Lp != L {
		y = sliceAxis(y, 1, 0, L)
	}

	kAll := reshape(mlx.Transpose(k, 0, 2, 3, 1, 4, 5), B, Lp, H, R, N)
	return y, m3State{
		theta: mlx.Squeeze(sliceAxis(theta, 1, L-1, L), 1),
		s:     S,
		k:     mlx.Squeeze(sliceAxis(kAll, 1, L-1, L), 1),
		x:     mlx.Squeeze(sliceAxis(in.x, 1, L-1, L), 1),
	}
}

// ---------------------------------------------------------------------------
// Differential tensor-product attention

func (a *DiffTPAttention) forward(h *mlx.Array, b *batch.Batch, kvc, shift cache.Layer, positions *mlx.Array, cfg *Config) *mlx.Array {
	dims := h.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	Hq, Hk, Dh, Rk := cfg.NumAttentionHeads, cfg.nNoise, cfg.HeadDim, cfg.TPARank

	big := f32(a.InProj.Forward(h))
	small := f32(a.InProjSmall.Forward(h))
	o1, o2, o3 := int(Hq*Dh), int((Hq+Rk)*Dh), int((Hq+2*Rk)*Dh)
	q := mlx.Reshape(sliceLast(big, 0, o1), B, L, Hq, Dh)
	bk := mlx.Reshape(sliceLast(big, o1, o2), B, L, Rk, Dh)
	bv := mlx.Reshape(sliceLast(big, o2, o3), B, L, Rk, Dh)
	gate := mlx.Reshape(sliceLast(big, o3, big.Dim(2)), B, L, cfg.nSig, Dh)

	s1, s2, s3, s4 := int(Hk*Rk), int(2*Hk*Rk), int(2*Hk*Rk+Hk), int(2*Hk*Rk+2*Hk)
	ak := mlx.Reshape(sliceLast(small, 0, s1), B, L, Hk, Rk)
	av := mlx.Reshape(sliceLast(small, s1, s2), B, L, Hk, Rk)
	alphaK := mlx.ExpandDims(mlx.Sigmoid(sliceLast(small, s2, s3)), -1) // (B,L,Hk,1)
	alphaV := mlx.ExpandDims(mlx.Sigmoid(sliceLast(small, s3, s4)), -1)
	lam := mlx.ExpandDims(mlx.ExpandDims(mlx.Sigmoid(sliceLast(small, s4, small.Dim(2))), -1), -1) // (B,L,Hn,1,1)

	k := mlx.DivScalar(mlx.Matmul(ak, bk), float32(Rk)) // (B,L,Hk,Dh)
	v := mlx.DivScalar(mlx.Matmul(av, bv), float32(Rk))

	// token shift: mix with the previous token's raw k / v
	kvw := int(Hk * Dh)
	var kLast, vLast *mlx.Array
	rc, hasShift := shift.(cache.Recurrent)
	if hasShift {
		hist := rc.Get(b, mlx.DTypeFloat32)
		conv := hist.ConvState() // (B,1,2*Hk*Dh)
		kLast = mlx.Reshape(sliceLast(conv, 0, kvw), B, 1, Hk, Dh)
		vLast = mlx.Reshape(sliceLast(conv, kvw, 2*kvw), B, 1, Hk, Dh)
		var convStates, deltaStates []*mlx.Array
		bounds := segments(shift, int(L))
		for _, e := range bounds[1:] {
			kk := mlx.Reshape(sliceAxis(k, 1, e-1, e), B, 1, int32(kvw))
			vv := mlx.Reshape(sliceAxis(v, 1, e-1, e), B, 1, int32(kvw))
			convStates = append(convStates, mlx.Concatenate([]*mlx.Array{kk, vv}, -1))
			deltaStates = append(deltaStates, hist.DeltaState())
		}
		rc.Put(b, convStates, deltaStates)
	} else {
		kLast = mlx.Zeros(mlx.DTypeFloat32, int(B), 1, int(Hk), int(Dh))
		vLast = mlx.Zeros(mlx.DTypeFloat32, int(B), 1, int(Hk), int(Dh))
	}
	kPrev, vPrev := kLast, vLast
	if L > 1 {
		kPrev = mlx.Concatenate([]*mlx.Array{kLast, sliceAxis(k, 1, 0, int(L)-1)}, 1)
		vPrev = mlx.Concatenate([]*mlx.Array{vLast, sliceAxis(v, 1, 0, int(L)-1)}, 1)
	}
	notFirst := mlx.ExpandDims(mlx.ExpandDims(positions.Greater(scalar(0)).AsType(mlx.DTypeFloat32), -1), -1) // (B,L,1,1)
	alphaK = mlx.Mul(alphaK, notFirst)
	alphaV = mlx.Mul(alphaV, notFirst)
	one := scalar(1)
	k = mlx.Add(mlx.Mul(alphaK, kPrev), mlx.Mul(mlx.Sub(one, alphaK), k))
	v = mlx.Add(mlx.Mul(alphaV, vPrev), mlx.Mul(mlx.Sub(one, alphaV), v))

	q = mlx.RMSNormFn(q, a.QNorm, cfg.NormEpsilon)
	k = mlx.RMSNormFn(k, a.KNorm, cfg.NormEpsilon)
	// scalable softmax: q *= s_h * log(min(pos + 1, window))
	logPos := mlx.Log(mlx.Minimum(mlx.AddScalar(positions, 1), scalar(float32(cfg.SlwWsize)))) // (B,L)
	q = mlx.Mul(q, mlx.ExpandDims(mlx.Mul(a.Scaler, mlx.ExpandDims(logPos, -1)), -1))

	dtype := h.DType()
	q = mlx.Transpose(q.AsType(dtype), 0, 2, 1, 3) // (B,Hq,L,Dh)
	k = mlx.Transpose(k.AsType(dtype), 0, 2, 1, 3) // (B,Hk,L,Dh)
	v = mlx.Transpose(v.AsType(dtype), 0, 2, 1, 3)

	var o *mlx.Array
	if kv, ok := kvc.(cache.Attention); ok {
		hist := kv.Update(b, k, v)
		if rows := hist.Rows(); rows != nil {
			outs := make([]*mlx.Array, len(rows))
			for i, row := range rows {
				qi := sliceAxis(q, 0, i, i+1)
				outs[i] = softcapAttention(qi, row.K(), row.V(), cfg)
			}
			o = mlx.Concatenate(outs, 0)
		} else {
			o = softcapAttention(q, hist.K(), hist.V(), cfg)
		}
	} else {
		o = softcapAttention(q, k, v, cfg)
	}

	o = mlx.Reshape(mlx.Transpose(o, 0, 2, 1, 3), B, L, cfg.nNoise, cfg.snr+1, Dh)
	sig := f32(sliceAxis(o, 3, 0, int(cfg.snr)))
	noise := f32(sliceAxis(o, 3, int(cfg.snr), int(cfg.snr)+1))
	o = mlx.Sub(sig, mlx.Mul(lam, noise))
	o = mlx.Mul(mlx.Reshape(o, B, L, cfg.nSig, Dh), silu(mlx.AddScalar(gate, cfg.gateBias)))
	return mlx.Reshape(o, B, L, cfg.nSig*Dh)
}

// softcapAttention computes causal attention with soft-capped logits. q:
// (B,Hq,L,D), k/v: (B,Hk,T,D) where the queries are the last L of T positions.
func softcapAttention(q, k, v *mlx.Array, cfg *Config) *mlx.Array {
	L := q.Dim(2)
	T := k.Dim(2)
	qc := max(1, attnScoreBudget/(4*q.Dim(0)*q.Dim(1)*T))
	if qc >= L {
		return softcapAttentionChunk(q, k, v, cfg, 0, L)
	}
	var outs []*mlx.Array
	for s0 := 0; s0 < L; s0 += qc {
		s1 := min(L, s0+qc)
		outs = append(outs, softcapAttentionChunk(sliceAxis(q, 2, s0, s1), k, v, cfg, s0, L))
	}
	return mlx.Concatenate(outs, 2)
}

func softcapAttentionChunk(q, k, v *mlx.Array, cfg *Config, q0, lTotal int) *mlx.Array {
	B, Hq, L, D := int32(q.Dim(0)), int32(q.Dim(1)), q.Dim(2), int32(q.Dim(3))
	Hk, T := int32(k.Dim(1)), k.Dim(2)
	G := Hq / Hk
	qg := mlx.Reshape(q, B, Hk, G, int32(L), D)
	s := mlx.MulScalar(f32(mlx.Matmul(qg, swapLast(mlx.ExpandDims(k, 2)))), cfg.attnScale) // (B,Hk,G,L,T)
	s = mlx.MulScalar(mlx.DivScalar(s, cfg.SoftcapAttn).Tanh(), cfg.SoftcapAttn)
	if L > 1 || T > lTotal {
		// causal: query i (absolute T-lTotal+q0+i) sees keys <= itself
		start := float64(T - lTotal + q0)
		qi := mlx.ExpandDims(mlx.Arange(start, start+float64(L), 1, mlx.DTypeInt32), -1)
		ki := mlx.ExpandDims(mlx.Arange(0, float64(T), 1, mlx.DTypeInt32), 0)
		mask := mlx.Where(ki.LessEqual(qi), scalar(0), scalar(float32(math.Inf(-1))))
		s = mlx.Add(s, mask)
	}
	p := mlx.SoftmaxAxis(s, -1, true).AsType(v.DType())
	o := mlx.Matmul(p, mlx.ExpandDims(v, 2)) // (B,Hk,G,L,D)
	return mlx.Reshape(o, B, Hq, int32(L), D)
}

// ---------------------------------------------------------------------------
// MoE

func (m *MoE) forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	K := cfg.NumActiveExperts

	logits := mlx.Matmul(f32(x), mlx.Transpose(m.GateW, 1, 0))
	probs, negSel := mlx.SigmoidRouter(logits, m.ExpertBias)
	inds := mlx.SliceStartStop(mlx.Argpartition(negSel, int(K)-1, -1), []int32{0, 0, 0}, []int32{B, L, K})
	scores := mlx.TakeAlongAxis(probs, inds, -1)
	scores = mlx.MulScalar(mlx.Div(scores, mlx.AddScalar(mlx.Sum(scores, -1, true), 1e-20)), cfg.RoutedScalingFactor)

	h := m.InCat.Forward(x)
	x0 := sliceLast(h, 0, int(cfg.RoutedInputDim))
	sh := sliceLast(h, int(cfg.RoutedInputDim), h.Dim(2))
	e := m.experts(x0, inds, scores).AsType(x.DType())
	return m.OutCat.Forward(mlx.Concatenate([]*mlx.Array{e, mlx.ReLUSquared(sh)}, -1))
}

func gather(ew expertWeights, x, idx *mlx.Array, sorted bool) *mlx.Array {
	if ew.Quantized {
		return mlx.GatherQMM(x, ew.W, ew.Scales, ew.Biases, nil, idx, true, ew.GroupSize, ew.Bits, ew.Mode, nil, sorted)
	}
	return mlx.GatherMM(x, ew.W, nil, idx, sorted)
}

// experts returns sum_k scores_k * expert_k(x) for the latent input x (B,L,d).
func (m *MoE) experts(x, inds, scores *mlx.Array) *mlx.Array {
	dims := x.Dims()
	B, L, width := int32(dims[0]), int32(dims[1]), int32(dims[2])
	K := int32(inds.Dim(2))

	xFlat := mlx.Reshape(x, B*L, 1, 1, width)
	idx := mlx.Reshape(inds, B*L, K)
	n := B * L * K
	doSort := B*L >= 64
	var invOrder *mlx.Array
	if doSort {
		all := mlx.Flatten(idx)
		order := mlx.Argsort(all, 0)
		invOrder = mlx.Argsort(order, 0)
		tokenOrder := mlx.FloorDivideScalar(order, K)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), tokenOrder, 0), 1)
		idx = mlx.Reshape(mlx.Take(all, order, 0), n, 1)
	}
	up := gather(m.Up, xFlat, idx, doSort)
	hidden := mlx.ReLUSquared(up)
	down := gather(m.Down, hidden, idx, doSort)
	y := mlx.GatherWeightedSum(mlx.Reshape(down, n, width), mlx.Reshape(f32(scores), B*L, K), invOrder)
	return mlx.Reshape(y, B, L, width)
}
