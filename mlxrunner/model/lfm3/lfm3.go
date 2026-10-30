// Package lfm3 implements Liquid's LFM3 hybrid text model for MLX.
package lfm3

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lfm3/lfm3audio"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

// This frequency keeps the nonrotary half of full-attention heads stationary.
const unrotatedRopeFrequency = 1e30

func init() {
	model.Register("Lfm3ForCausalLM", newModel)
	model.Register("Lfm3VlForConditionalGeneration", newVLModel)
}

type ropeConfig struct {
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
}

// Config describes the language model in either a text or a multimodal checkpoint.
type Config struct {
	HiddenSize            int32      `json:"hidden_size"`
	IntermediateSize      int32      `json:"intermediate_size"`
	NumHiddenLayers       int32      `json:"num_hidden_layers"`
	NumAttentionHeads     int32      `json:"num_attention_heads"`
	NumKeyValueHeads      int32      `json:"num_key_value_heads"`
	HeadDim               int32      `json:"head_dim"`
	VocabSize             int32      `json:"vocab_size"`
	NormEps               float32    `json:"norm_eps"`
	ConvLCache            int32      `json:"conv_L_cache"`
	LayerTypes            []string   `json:"layer_types"`
	SlidingWindow         int32      `json:"sliding_window"`
	MaxPositionEmbeddings int32      `json:"max_position_embeddings"`
	TieWordEmbeddings     bool       `json:"tie_word_embeddings"`
	UseSandwichNorm       bool       `json:"use_sandwich_norm"`
	FullRope              ropeConfig `json:"-"`
	SlidingRope           ropeConfig `json:"-"`

	QuantGroupSize int                               `json:"-"`
	QuantBits      int                               `json:"-"`
	QuantMode      string                            `json:"-"`
	TensorQuant    map[string]*model.TensorQuantInfo `json:"-"`
	Scale          float32                           `json:"-"`
	FullRopeFreqs  *mlx.Array                        `json:"-"`
	// UncachedAttention, when set, computes attention layers run without a
	// cache from rotated q, k and v; d1 decision checkpoints attend in packed
	// question sets.
	UncachedAttention func(q, k, v *mlx.Array, scale float32, sliding bool) *mlx.Array `json:"-"`
}

// Model holds a decoder with attention and short convolution layers.
type Model struct {
	EmbedTokens   nn.EmbeddingLayer
	EmbeddingNorm *nn.RMSNorm
	Layers        []*Layer
	LMHead        nn.LinearLayer
	Vision        *VisionModel
	Audio         *audioInput
	*Config
	tok *tokenizer.Tokenizer
	// prefix names the decoder tensors: "model." alone, "model.language_model."
	// in VL and Omni, "lfm.model." in Audio.
	prefix string
	// Suppress keeps chat decoding in text mode for speech-capable checkpoints.
	Suppress *mlx.Array
}

type Layer struct {
	LayerType        string
	OperatorNorm     *nn.RMSNorm
	PostOperatorNorm *nn.RMSNorm
	FFNNorm          *nn.RMSNorm
	PostFFNNorm      *nn.RMSNorm
	Attention        *Attention
	Conv             *ShortConv
	MLP              *MLP
}

type Attention struct {
	QProj    nn.LinearLayer
	KProj    nn.LinearLayer
	VProj    nn.LinearLayer
	OProj    nn.LinearLayer
	GateProj nn.LinearLayer
	QNorm    *nn.RMSNorm
	KNorm    *nn.RMSNorm
}

type ShortConv struct {
	InProj  nn.LinearLayer
	OutProj nn.LinearLayer
	Conv1D  *nn.Conv1d
}

type MLP struct {
	W1 nn.LinearLayer
	W2 nn.LinearLayer
	W3 nn.LinearLayer
}

func newModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	return newModelFromConfig(root, data, nil)
}

// newModelFromConfig builds the decoder. generationConfig replaces
// generation_config.json for checkpoints that do not ship one.
func newModelFromConfig(root *model.Root, data, generationConfig []byte) (*Model, error) {
	var source struct {
		Config
		RopeParameters    map[string]ropeConfig `json:"rope_parameters"`
		TieWordEmbeddings *bool                 `json:"tie_word_embeddings"`
		UseSandwichNorm   *bool                 `json:"use_sandwich_norm"`
	}
	if err := json.Unmarshal(data, &source); err != nil {
		return nil, fmt.Errorf("parse LFM3 config: %w", err)
	}
	if source.TieWordEmbeddings == nil || source.UseSandwichNorm == nil {
		return nil, fmt.Errorf("missing LFM3 tie_word_embeddings or use_sandwich_norm")
	}
	cfg := source.Config
	cfg.TieWordEmbeddings = *source.TieWordEmbeddings
	cfg.UseSandwichNorm = *source.UseSandwichNorm
	cfg.FullRope = source.RopeParameters["full_attention"]
	cfg.SlidingRope = source.RopeParameters["sliding_attention"]
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.VocabSize <= 0 || cfg.MaxPositionEmbeddings <= 0 {
		return nil, fmt.Errorf("invalid LFM3 dimensions: hidden=%d intermediate=%d vocab=%d context=%d", cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.MaxPositionEmbeddings)
	}
	if cfg.NumHiddenLayers <= 0 || len(cfg.LayerTypes) != int(cfg.NumHiddenLayers) {
		return nil, fmt.Errorf("LFM3 layer_types has %d entries for %d layers", len(cfg.LayerTypes), cfg.NumHiddenLayers)
	}
	if cfg.HeadDim <= 0 || cfg.NumAttentionHeads <= 0 || cfg.NumKeyValueHeads <= 0 || cfg.NumAttentionHeads%cfg.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("invalid LFM3 attention heads: query=%d kv=%d width=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim)
	}
	if cfg.ConvLCache <= 0 {
		return nil, fmt.Errorf("invalid conv_L_cache: %d", cfg.ConvLCache)
	}
	if cfg.NormEps <= 0 {
		return nil, fmt.Errorf("invalid LFM3 norm_eps: %g", cfg.NormEps)
	}
	if cfg.SlidingWindow <= 0 {
		return nil, fmt.Errorf("invalid sliding_window: %d", cfg.SlidingWindow)
	}
	for i, layerType := range cfg.LayerTypes {
		switch layerType {
		case "conv", "full_attention", "sliding_attention":
		default:
			return nil, fmt.Errorf("unsupported LFM3 layer type %q at %d", layerType, i)
		}
	}
	if cfg.FullRope.RopeTheta <= 0 || cfg.SlidingRope.RopeTheta <= 0 {
		return nil, fmt.Errorf("missing LFM3 rope theta")
	}
	if cfg.FullRope.PartialRotaryFactor <= 0 || cfg.FullRope.PartialRotaryFactor > 1 {
		return nil, fmt.Errorf("invalid full_attention partial_rotary_factor: %g", cfg.FullRope.PartialRotaryFactor)
	}
	// The sliding-attention config omits this field to request full-head RoPE.
	if factor := cfg.SlidingRope.PartialRotaryFactor; factor != 0 && factor != 1 {
		return nil, fmt.Errorf("unsupported sliding_attention partial_rotary_factor: %g", factor)
	}
	cfg.Scale = float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	if qt := root.QuantType(); qt != "" {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams(qt)
		if root.GroupSize() > 0 {
			cfg.QuantGroupSize = root.GroupSize()
		}
	} else {
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = model.QuantizationParams("")
	}
	cfg.TensorQuant = root.AllTensorQuant()

	// Full attention rotates only part of each head. Its frequencies still use
	// the complete head width as the exponent denominator.
	if rp := cfg.FullRope; rp.PartialRotaryFactor > 0 && rp.PartialRotaryFactor < 1 {
		frequencies := make([]float32, cfg.HeadDim/2)
		rotated := int(float32(cfg.HeadDim)*rp.PartialRotaryFactor) / 2
		if rotated == 0 {
			return nil, fmt.Errorf("full attention rotary fraction %g rotates no dimensions", rp.PartialRotaryFactor)
		}
		for i := range frequencies {
			if i < rotated {
				frequencies[i] = float32(math.Pow(float64(rp.RopeTheta), float64(2*i)/float64(cfg.HeadDim)))
			} else {
				frequencies[i] = unrotatedRopeFrequency
			}
		}
		cfg.FullRopeFreqs = mlx.FromValues(frequencies, len(frequencies))
	}

	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokConfig := &tokenizer.TokenizerConfig{ConfigJSON: data}
	tokConfig.GenerationConfigJSON = generationConfig
	if generationConfig == nil {
		if tokConfig.GenerationConfigJSON, err = root.Manifest.ReadConfig("generation_config.json"); err != nil {
			return nil, fmt.Errorf("load generation_config.json: %w", err)
		}
	}
	if tokConfig.TokenizerConfigJSON, err = root.Manifest.ReadConfig("tokenizer_config.json"); err != nil {
		return nil, fmt.Errorf("load tokenizer_config.json: %w", err)
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, tokConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	return &Model{Config: &cfg, tok: tok, Layers: make([]*Layer, cfg.NumHiddenLayers), prefix: "model."}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, m.prefix+"embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing %sembed_tokens.weight", m.prefix)
	}
	if w := tensors[m.prefix+"embedding_norm.weight"]; w != nil {
		m.EmbeddingNorm = nn.NewRMSNorm(w, m.NormEps)
	} else {
		return fmt.Errorf("missing %sembedding_norm.weight", m.prefix)
	}
	if m.TieWordEmbeddings {
		m.LMHead = m.EmbedTokens.AsLinear()
	} else {
		m.LMHead = linears.Make("lm_head")
		if m.LMHead == nil {
			return fmt.Errorf("missing lm_head.weight")
		}
	}

	for i, layerType := range m.LayerTypes {
		prefix := fmt.Sprintf("%slayers.%d", m.prefix, i)
		layer := &Layer{LayerType: layerType, MLP: &MLP{}}
		for _, norm := range []struct {
			name string
			dest **nn.RMSNorm
		}{
			{"operator_norm", &layer.OperatorNorm},
			{"post_operator_norm", &layer.PostOperatorNorm},
			{"ffn_norm", &layer.FFNNorm},
			{"post_ffn_norm", &layer.PostFFNNorm},
		} {
			if w := tensors[prefix+"."+norm.name+".weight"]; w != nil {
				*norm.dest = nn.NewRMSNorm(w, m.NormEps)
			}
		}
		if layer.OperatorNorm == nil || layer.FFNNorm == nil || (m.UseSandwichNorm && (layer.PostOperatorNorm == nil || layer.PostFFNNorm == nil)) {
			return fmt.Errorf("layer %d: missing normalization weight", i)
		}
		layer.MLP.W1 = linears.Make(prefix + ".feed_forward.w1")
		layer.MLP.W2 = linears.Make(prefix + ".feed_forward.w2")
		layer.MLP.W3 = linears.Make(prefix + ".feed_forward.w3")
		if layer.MLP.W1 == nil || layer.MLP.W2 == nil || layer.MLP.W3 == nil {
			return fmt.Errorf("layer %d: missing feed-forward projection", i)
		}

		if layerType == "conv" {
			convPrefix := prefix + ".conv"
			conv := &ShortConv{
				InProj:  linears.Make(convPrefix + ".in_proj"),
				OutProj: linears.Make(convPrefix + ".out_proj"),
			}
			w := tensors[convPrefix+".conv.weight"]
			if conv.InProj == nil || conv.OutProj == nil || w == nil {
				return fmt.Errorf("layer %d: missing short-convolution weight", i)
			}
			// PyTorch stores [channel, 1, kernel], MLX expects [channel, kernel, 1].
			conv.Conv1D = nn.NewConv1d(mlx.Transpose(w, 0, 2, 1), nil, 1, 0, 1, m.HiddenSize)
			layer.Conv = conv
		} else {
			attnPrefix := prefix + ".self_attn"
			attn := &Attention{
				QProj:    linears.Make(attnPrefix + ".q_proj"),
				KProj:    linears.Make(attnPrefix + ".k_proj"),
				VProj:    linears.Make(attnPrefix + ".v_proj"),
				OProj:    linears.Make(attnPrefix + ".o_proj"),
				GateProj: linears.Make(attnPrefix + ".gate_proj"),
			}
			if w := tensors[attnPrefix+".q_norm.weight"]; w != nil {
				attn.QNorm = nn.NewRMSNorm(w, m.NormEps)
			}
			if w := tensors[attnPrefix+".k_norm.weight"]; w != nil {
				attn.KNorm = nn.NewRMSNorm(w, m.NormEps)
			}
			if attn.QProj == nil || attn.KProj == nil || attn.VProj == nil || attn.OProj == nil || attn.GateProj == nil || attn.QNorm == nil || attn.KNorm == nil {
				return fmt.Errorf("layer %d: missing attention weight", i)
			}
			layer.Attention = attn
		}
		m.Layers[i] = layer
	}
	if m.Vision != nil {
		if err := m.Vision.LoadWeights(tensors); err != nil {
			return err
		}
	}
	if m.Audio != nil {
		enc, err := lfm3audio.LoadEncoder(tensors, m.Audio.cfg, linears)
		if err != nil {
			return err
		}
		m.Audio.Enc = enc
	}
	return nil
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	h := m.EmbedTokens.Forward(b.InputIDs)
	return m.forwardHidden(h, b, caches)
}

// uncachedLayerEvalTokens is the row length from which uncached rows are
// evaluated layer by layer; shorter rows skip the per-layer round trips.
const uncachedLayerEvalTokens = 4096

func (m *Model) forwardHidden(h *mlx.Array, b *batch.Batch, caches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	B, L := int32(b.InputIDs.Dim(0)), int32(b.InputIDs.Dim(1))
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	for i, layer := range m.Layers {
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		if c == nil && m.UncachedAttention != nil && L >= uncachedLayerEvalTokens {
			// A long uncached row in one graph holds several layers'
			// activations at once; layer by layer, peak memory stays at one.
			h = mlx.ScopedEval(func() []*mlx.Array {
				return []*mlx.Array{layer.Forward(h, b, c, positions, B, L, m.Config)}
			})[0]
			continue
		}
		h = layer.Forward(h, b, c, positions, B, L, m.Config)
	}
	h = m.EmbeddingNorm.Forward(h, m.NormEps)
	return h, h
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	logits := m.LMHead.Forward(x)
	if m.Suppress != nil {
		logits = logits.Add(m.Suppress.AsType(logits.DType()))
	}
	return logits
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) MaxContextLength() int { return int(m.MaxPositionEmbeddings) }

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, len(m.Layers))
	for i, layer := range m.Layers {
		switch layer.LayerType {
		case "conv":
			// RecurrentCache owns the convolution tail and snapshot history. Its
			// one-scalar delta slot is unused by LFM3's plain convolution.
			caches[i] = cache.NewRecurrentCache(m.ConvLCache-1, m.HiddenSize, 1, 1, 1)
		case "sliding_attention":
			caches[i] = cache.NewRotatingKVCache(int(m.SlidingWindow))
		default:
			caches[i] = cache.NewKVCache()
		}
	}
	return caches
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	normed := l.OperatorNorm.Forward(x, cfg.NormEps)
	var op *mlx.Array
	if l.Conv != nil {
		op = l.Conv.Forward(normed, b, c, B, L, cfg)
	} else {
		op = l.Attention.Forward(normed, b, c, positions, B, L, l.LayerType == "sliding_attention", cfg)
	}
	if cfg.UseSandwichNorm {
		op = l.PostOperatorNorm.Forward(op, cfg.NormEps)
	}
	h := mlx.Add(x, op)
	ffn := l.MLP.Forward(l.FFNNorm.Forward(h, cfg.NormEps))
	if cfg.UseSandwichNorm {
		ffn = l.PostFFNNorm.Forward(ffn, cfg.NormEps)
	}
	return mlx.Add(h, ffn)
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, sliding bool, cfg *Config) *mlx.Array {
	q := mlx.Transpose(mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	k := mlx.Transpose(mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	v := mlx.Transpose(mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim), 0, 2, 1, 3)
	q = a.QNorm.Forward(q, cfg.NormEps)
	k = a.KNorm.Forward(k, cfg.NormEps)
	rope := cfg.FullRope
	frequencies := cfg.FullRopeFreqs
	if sliding {
		rope = cfg.SlidingRope
		frequencies = nil
	}
	q = mlx.RoPEWithFreqs(q, int(cfg.HeadDim), false, rope.RopeTheta, 1, positions, frequencies)
	k = mlx.RoPEWithFreqs(k, int(cfg.HeadDim), false, rope.RopeTheta, 1, positions, frequencies)

	var out *mlx.Array
	if c == nil && cfg.UncachedAttention != nil {
		out = cfg.UncachedAttention(q, k, v, cfg.Scale, sliding)
	} else {
		var kv nn.SDPAOption
		mask := nn.CausalMask()
		if c != nil {
			kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
		} else {
			kv = nn.WithKV(k, v, b.SeqQueryLens)
			if sliding {
				mask = mask.Intersect(nn.SlidingWindowMask(b, int(L), int(cfg.SlidingWindow), q.DType()))
			}
		}
		out = nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask))
	}
	out = mlx.Transpose(out, 0, 2, 1, 3)
	gate := mlx.Sigmoid(a.GateProj.Forward(x))
	out = mlx.Mul(out, mlx.ExpandDims(gate, -1))
	out = mlx.Reshape(out, B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(out)
}

func (s *ShortConv) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *Config) *mlx.Array {
	projection := s.InProj.Forward(x)
	D := cfg.HiddenSize
	bInput := mlx.SliceStartStop(projection, []int32{0, 0, 0}, []int32{B, L, D})
	gate := mlx.SliceStartStop(projection, []int32{0, 0, D}, []int32{B, L, 2 * D})
	xInput := mlx.SliceStartStop(projection, []int32{0, 0, 2 * D}, []int32{B, L, 3 * D})
	convInput := mlx.Mul(bInput, xInput)
	tail := int(cfg.ConvLCache - 1)
	var opts []nn.RecurrentOption
	var recurrent *cache.RecurrentCache
	var delta *mlx.Array
	if c != nil {
		recurrent = c.(*cache.RecurrentCache)
		history := recurrent.Get(b, x.DType())
		delta = history.DeltaState()
		opts = append(opts, nn.WithRecurrentHistory(history))
		if splits := recurrent.SnapshotSplits(int(L)); len(splits) > 0 {
			opts = append(opts, nn.WithSnapshotSplits(splits))
		}
	} else {
		opts = append(opts, nn.WithRecurrentState(mlx.Zeros(x.DType(), int(B), tail, int(D)), nil))
	}
	convOut, states := nn.CausalConv1D(b, convInput, s.Conv1D, tail, opts...)
	if recurrent != nil {
		deltaStates := make([]*mlx.Array, len(states))
		for i := range deltaStates {
			deltaStates[i] = delta
		}
		recurrent.Put(b, states, deltaStates)
	}
	return s.OutProj.Forward(mlx.Mul(gate, convOut))
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.W2.Forward(nn.SwiGLU(m.W1, m.W3, x))
}
