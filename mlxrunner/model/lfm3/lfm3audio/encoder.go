package lfm3audio

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Long clips run the downsampler and attention in blocks so activations
// stay linear in clip length. Variables so tests can force small blocks.
var (
	downsampleBlockTokens int32 = 256
	attentionBlockTokens  int32 = 256
)

// Layout names the speech tensors in one checkpoint namespace.
type Layout struct {
	Downsample string
	Encoder    string
	Projector  [3]string // layer norm, first linear, second linear
}

var (
	// OmniLayout is the LFM3-Omni namespace.
	OmniLayout = Layout{
		Downsample: "model.audio_encoder.downsample.",
		Encoder:    "model.audio_encoder.lfm.",
		Projector:  [3]string{"model.audio_projector.layer_norm", "model.audio_projector.linear_1", "model.audio_projector.linear_2"},
	}
	// AudioLayout is the LFM3-Audio training-export namespace.
	AudioLayout = Layout{
		Downsample: "encoder.downsample.",
		Encoder:    "encoder.lfm.lfm.",
		Projector:  [3]string{"mlp.model.0", "mlp.model.1", "mlp.model.3"},
	}
)

// DetectLayout reports which namespace holds the speech encoder.
func DetectLayout(tensors map[string]*mlx.Array) (Layout, error) {
	for _, l := range []Layout{OmniLayout, AudioLayout} {
		if tensors[l.Downsample+"conv_in.weight"] != nil {
			return l, nil
		}
	}
	return Layout{}, fmt.Errorf("LFM3 speech encoder weights not found")
}

// Encoder maps log-mel features to language model embeddings.
type Encoder struct {
	Config     Config
	Downsample *Downsampler
	Layers     []*Layer
	Norm       *nn.RMSNorm
	Projector  *Projector
}

// Downsampler turns 8 frames of 128 mel bins into one encoder input with
// three stride-2 convolutions over the (mel, time) plane.
type Downsampler struct {
	ConvIn, ConvMid, ConvEnd *conv2d
	LinMid, LinEnd, LinOut   nn.LinearLayer
}

type conv2d struct {
	Weight *mlx.Array // [out, kH, kW, in/groups]
	Bias   *mlx.Array
	Groups int32
}

type Layer struct {
	OperatorNorm *nn.RMSNorm
	FFNNorm      *nn.RMSNorm
	Conv         *ShortConv
	Attention    *Attention
	W1, W2, W3   nn.LinearLayer
}

type ShortConv struct {
	InProj  nn.LinearLayer
	OutProj nn.LinearLayer
	Weight  *mlx.Array // [hidden, kernel, 1]
}

type Attention struct {
	QProj, KProj, VProj, OutProj nn.LinearLayer
	QNorm, KNorm                 *nn.RMSNorm
}

type Projector struct {
	Norm    *nn.LayerNorm
	Linear1 nn.LinearLayer
	Linear2 nn.LinearLayer
}

// LoadEncoder builds the encoder from either checkpoint namespace.
func LoadEncoder(tensors map[string]*mlx.Array, cfg Config, linears model.LinearFactory) (*Encoder, error) {
	layout, err := DetectLayout(tensors)
	if err != nil {
		return nil, err
	}
	e := &Encoder{Config: cfg}
	if e.Downsample, err = loadDownsampler(tensors, layout.Downsample, cfg); err != nil {
		return nil, err
	}

	required := func(name string, shape ...int) (*mlx.Array, error) {
		w := tensors[name]
		if w == nil {
			return nil, fmt.Errorf("missing LFM3 speech weight %s", name)
		}
		if len(shape) > 0 && !sameShape(w.Dims(), shape) {
			return nil, fmt.Errorf("LFM3 speech weight %s has shape %v, want %v", name, w.Dims(), shape)
		}
		return w, nil
	}
	linear := func(name string, out, in int32) (nn.LinearLayer, error) {
		l := linears.Make(name)
		if l == nil {
			return nil, fmt.Errorf("missing LFM3 speech weight %s.weight", name)
		}
		if l.OutputDim() != out {
			return nil, fmt.Errorf("LFM3 speech weight %s has %d outputs, want %d", name, l.OutputDim(), out)
		}
		if dense, ok := l.(*nn.Linear); ok && int32(dense.Weight.Dim(1)) != in {
			return nil, fmt.Errorf("LFM3 speech weight %s has %d inputs, want %d", name, dense.Weight.Dim(1), in)
		}
		return l, nil
	}
	norm := func(name string, width int32) (*nn.RMSNorm, error) {
		w, err := required(name+".weight", int(width))
		if err != nil {
			return nil, err
		}
		return nn.NewRMSNorm(w, cfg.NormEps), nil
	}

	d, ff := cfg.HiddenSize, cfg.IntermediateSize
	qWidth, kvWidth := cfg.NumAttentionHeads*cfg.HeadDim, cfg.NumKeyValueHeads*cfg.HeadDim
	e.Layers = make([]*Layer, len(cfg.LayerTypes))
	for i, kind := range cfg.LayerTypes {
		p := fmt.Sprintf("%slayers.%d.", layout.Encoder, i)
		l := &Layer{}
		if l.OperatorNorm, err = norm(p+"operator_norm", d); err != nil {
			return nil, err
		}
		if l.FFNNorm, err = norm(p+"ffn_norm", d); err != nil {
			return nil, err
		}
		if l.W1, err = linear(p+"feed_forward.w1", ff, d); err != nil {
			return nil, err
		}
		if l.W3, err = linear(p+"feed_forward.w3", ff, d); err != nil {
			return nil, err
		}
		if l.W2, err = linear(p+"feed_forward.w2", d, ff); err != nil {
			return nil, err
		}
		if kind == "conv" {
			c := &ShortConv{}
			if c.InProj, err = linear(p+"conv.in_proj", 3*d, d); err != nil {
				return nil, err
			}
			if c.OutProj, err = linear(p+"conv.out_proj", d, d); err != nil {
				return nil, err
			}
			w, err := required(p+"conv.conv.weight", int(d), 1, int(cfg.ConvKernel))
			if err != nil {
				return nil, err
			}
			c.Weight = mlx.Transpose(w, 0, 2, 1)
			l.Conv = c
		} else {
			a := &Attention{}
			if a.QProj, err = linear(p+"self_attn.q_proj", qWidth, d); err != nil {
				return nil, err
			}
			if a.KProj, err = linear(p+"self_attn.k_proj", kvWidth, d); err != nil {
				return nil, err
			}
			if a.VProj, err = linear(p+"self_attn.v_proj", kvWidth, d); err != nil {
				return nil, err
			}
			if a.OutProj, err = linear(p+"self_attn.out_proj", d, qWidth); err != nil {
				return nil, err
			}
			if a.QNorm, err = norm(p+"self_attn.q_layernorm", cfg.HeadDim); err != nil {
				return nil, err
			}
			if a.KNorm, err = norm(p+"self_attn.k_layernorm", cfg.HeadDim); err != nil {
				return nil, err
			}
			l.Attention = a
		}
		e.Layers[i] = l
	}
	if e.Norm, err = norm(layout.Encoder+"embedding_norm", d); err != nil {
		return nil, err
	}

	pr := &Projector{}
	w, err := required(layout.Projector[0]+".weight", int(d))
	if err != nil {
		return nil, err
	}
	b, err := required(layout.Projector[0]+".bias", int(d))
	if err != nil {
		return nil, err
	}
	pr.Norm = &nn.LayerNorm{Weight: w, Bias: b, Eps: cfg.ProjectorNormEps}
	if pr.Linear1, err = linear(layout.Projector[1], cfg.OutputSize, d); err != nil {
		return nil, err
	}
	if pr.Linear2, err = linear(layout.Projector[2], cfg.OutputSize, cfg.OutputSize); err != nil {
		return nil, err
	}
	e.Projector = pr
	return e, nil
}

func loadDownsampler(tensors map[string]*mlx.Array, prefix string, cfg Config) (*Downsampler, error) {
	ch := int(cfg.ConvChannels)
	get := func(name string, shape ...int) (*mlx.Array, error) {
		w := tensors[prefix+name]
		if w == nil {
			return nil, fmt.Errorf("missing LFM3 speech weight %s%s", prefix, name)
		}
		if !sameShape(w.Dims(), shape) {
			return nil, fmt.Errorf("LFM3 speech weight %s%s has shape %v, want %v", prefix, name, w.Dims(), shape)
		}
		return w, nil
	}
	conv := func(name string, in, groups int) (*conv2d, error) {
		w, err := get(name+".weight", ch, in/groups, 3, 3)
		if err != nil {
			return nil, err
		}
		b, err := get(name+".bias", ch)
		if err != nil {
			return nil, err
		}
		// PyTorch stores [out, in, kH, kW]; MLX wants channels last.
		return &conv2d{Weight: mlx.Transpose(w, 0, 2, 3, 1), Bias: b, Groups: int32(groups)}, nil
	}
	pointwise := func(name string, out, in int, shape ...int) (nn.LinearLayer, error) {
		w, err := get(name+".weight", shape...)
		if err != nil {
			return nil, err
		}
		b, err := get(name+".bias", out)
		if err != nil {
			return nil, err
		}
		return nn.NewLinear(mlx.Reshape(w, int32(out), int32(in)), b), nil
	}

	var d Downsampler
	var err error
	if d.ConvIn, err = conv("conv_in", 1, 1); err != nil {
		return nil, err
	}
	if d.ConvMid, err = conv("conv_mid", ch, ch); err != nil {
		return nil, err
	}
	if d.ConvEnd, err = conv("conv_end", ch, ch); err != nil {
		return nil, err
	}
	if d.LinMid, err = pointwise("lin_mid", ch, ch, ch, ch, 1, 1); err != nil {
		return nil, err
	}
	if d.LinEnd, err = pointwise("lin_end", ch, ch, ch, ch, 1, 1); err != nil {
		return nil, err
	}
	flat := ch * cfg.NumMelBins / 8
	if d.LinOut, err = pointwise("lin_out", int(cfg.HiddenSize), flat, int(cfg.HiddenSize), flat, 1); err != nil {
		return nil, err
	}
	return &d, nil
}

func sameShape(got []int, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// FeatureArray uploads features as the [mel, frames] encoder input.
func FeatureArray(f Features) *mlx.Array {
	return mlx.FromValues(f.Data, len(f.Data)/f.Frames, f.Frames)
}

// Forward encodes one clip's [mel, frames] features into [tokens, OutputSize]
// language model embeddings.
func (e *Encoder) Forward(features *mlx.Array) *mlx.Array {
	h := e.Encode(features)
	return e.Projector.Forward(h)
}

// Encode returns the encoder output [tokens, HiddenSize] before projection.
func (e *Encoder) Encode(features *mlx.Array) *mlx.Array {
	h := mlx.ExpandDims(e.Downsample.Forward(features), 0)
	for _, l := range e.Layers {
		h = l.Forward(h, &e.Config)
	}
	return mlx.Squeeze(e.Norm.Forward(h, e.Config.NormEps), 0)
}

// Forward maps [mel, frames] features to [ceil(frames/8), hidden].
func (d *Downsampler) Forward(features *mlx.Array) *mlx.Array {
	x := features.AsType(d.ConvIn.Weight.DType())
	mels, frames := int32(x.Dim(0)), int32(x.Dim(1))
	tokens := (frames + 7) / 8
	if tokens <= downsampleBlockTokens {
		return d.forward(x)
	}
	// Each output reads 7 frames either side, and zero padding at a cut
	// disturbs one more output per stage, so cuts carry 2 tokens of margin.
	const margin = 2
	var parts []*mlx.Array
	for start := int32(0); start < tokens; start += downsampleBlockTokens {
		end := min(start+downsampleBlockTokens, tokens)
		lo := max(0, start-margin)
		hi := min(frames, 8*(end+margin))
		chunk := mlx.SliceStartStop(x, []int32{0, 8 * lo}, []int32{mels, hi})
		out := d.forward(chunk)
		parts = append(parts, mlx.SliceStartStop(out, []int32{start - lo, 0}, []int32{end - lo, int32(out.Dim(1))}))
	}
	return mlx.Concatenate(parts, 0)
}

func (d *Downsampler) forward(x *mlx.Array) *mlx.Array {
	h := mlx.ExpandDims(mlx.ExpandDims(x, 0), 3) // [1, mel, frames, 1]
	h = mlx.SiLU(d.ConvIn.Forward(h))
	h = mlx.SiLU(d.LinMid.Forward(d.ConvMid.Forward(h)))
	h = mlx.SiLU(d.LinEnd.Forward(d.ConvEnd.Forward(h)))
	// [1, mel/8, tokens, ch] flattens channel-major: feature c*mel/8 + m.
	mels, tokens, ch := int32(h.Dim(1)), int32(h.Dim(2)), int32(h.Dim(3))
	h = mlx.Reshape(mlx.Transpose(h, 0, 2, 3, 1), tokens, ch*mels)
	return d.LinOut.Forward(h)
}

func (c *conv2d) Forward(x *mlx.Array) *mlx.Array {
	return mlx.Add(mlx.Conv2d(x, c.Weight, 2, 2, 1, 1, 1, 1, c.Groups), c.Bias)
}

func (l *Layer) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	normed := l.OperatorNorm.Forward(x, cfg.NormEps)
	var op *mlx.Array
	if l.Conv != nil {
		op = l.Conv.Forward(normed, cfg)
	} else {
		op = l.Attention.Forward(normed, cfg)
	}
	h := mlx.Add(x, op)
	return mlx.Add(h, l.W2.Forward(nn.SwiGLU(l.W1, l.W3, l.FFNNorm.Forward(h, cfg.NormEps))))
}

// Forward is LFM2's gated short convolution: B*x through a causal depthwise
// convolution, gated by C.
func (s *ShortConv) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	L, D := int32(x.Dim(1)), cfg.HiddenSize
	p := s.InProj.Forward(x)
	b := mlx.SliceStartStop(p, []int32{0, 0, 0}, []int32{1, L, D})
	c := mlx.SliceStartStop(p, []int32{0, 0, D}, []int32{1, L, 2 * D})
	in := mlx.SliceStartStop(p, []int32{0, 0, 2 * D}, []int32{1, L, 3 * D})
	y := mlx.PadConstant(mlx.Mul(b, in), []int{1}, []int{int(cfg.ConvKernel - 1)}, []int{0})
	y = mlx.Conv1d(y, s.Weight, nil, 1, 0, 1, D)
	return s.OutProj.Forward(mlx.Mul(c, y))
}

// Forward attends over a causal window of SlidingWindow tokens, the query
// included. The checkpoint calls these layers full attention; the encoder
// always applies the window.
func (a *Attention) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	L := int32(x.Dim(1))
	H, KV, hd := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim
	q := a.QNorm.Forward(mlx.Reshape(a.QProj.Forward(x), 1, L, H, hd), cfg.NormEps)
	k := a.KNorm.Forward(mlx.Reshape(a.KProj.Forward(x), 1, L, KV, hd), cfg.NormEps)
	v := mlx.Reshape(a.VProj.Forward(x), 1, L, KV, hd)
	q, k, v = mlx.Transpose(q, 0, 2, 1, 3), mlx.Transpose(k, 0, 2, 1, 3), mlx.Transpose(v, 0, 2, 1, 3)
	offset := mlx.FromValues([]int32{0}, 1)
	q = mlx.RoPEWithBase(q, int(hd), false, cfg.RopeTheta, 1, offset)
	k = mlx.RoPEWithBase(k, int(hd), false, cfg.RopeTheta, 1, offset)

	scale := float32(1 / math.Sqrt(float64(hd)))
	var parts []*mlx.Array
	for start := int32(0); start < L; start += attentionBlockTokens {
		end := min(start+attentionBlockTokens, L)
		kStart := max(0, start-cfg.SlidingWindow+1)
		qb := mlx.SliceStartStop(q, []int32{0, 0, start, 0}, []int32{1, H, end, hd})
		kb := mlx.SliceStartStop(k, []int32{0, 0, kStart, 0}, []int32{1, KV, end, hd})
		vb := mlx.SliceStartStop(v, []int32{0, 0, kStart, 0}, []int32{1, KV, end, hd})
		mask := windowMask(start, end, kStart, cfg.SlidingWindow, q.DType())
		parts = append(parts, mlx.FastScaledDotProductAttention(qb, kb, vb, scale, mlx.FastSDPANoWindow, "", mask))
	}
	out := parts[0]
	if len(parts) > 1 {
		out = mlx.Concatenate(parts, 2)
	}
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), 1, L, H*hd)
	return a.OutProj.Forward(out)
}

// windowMask allows key position k for query position q when
// q-window < k <= q, as an additive [1, 1, queries, keys] mask.
func windowMask(qStart, qEnd, kStart, window int32, dtype mlx.DType) *mlx.Array {
	q := mlx.Reshape(mlx.Arange(float64(qStart), float64(qEnd), 1, mlx.DTypeInt32), qEnd-qStart, 1)
	k := mlx.Reshape(mlx.Arange(float64(kStart), float64(qEnd), 1, mlx.DTypeInt32), 1, qEnd-kStart)
	blocked := mlx.FromValue(float32(math.Inf(-1)))
	open := mlx.FromValue(float32(0))
	tooOld := k.LessEqual(mlx.Sub(q, mlx.FromValues([]int32{window}, 1)))
	mask := mlx.Where(k.Greater(q), blocked, mlx.Where(tooOld, blocked, open))
	return mlx.Reshape(mask, 1, 1, qEnd-qStart, qEnd-kStart).AsType(dtype)
}

// Forward is LayerNorm, Linear, exact GELU, Linear. The GELU runs in
// float32 and rounds once, as PyTorch does for bfloat16 inputs.
func (p *Projector) Forward(x *mlx.Array) *mlx.Array {
	h := p.Linear1.Forward(p.Norm.Forward(x))
	h = mlx.GELU(h.AsType(mlx.DTypeFloat32)).AsType(h.DType())
	return p.Linear2.Forward(h)
}
