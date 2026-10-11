package pockettts

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx"
)

// Mimi is the streaming decoder half of the Mimi codec: latent frames in,
// 24 kHz samples out.
type Mimi struct {
	OutputProj  *mlx.Array // [latent, dim]
	Upsample    *ConvTr
	Transformer *Transformer
	Context     int
	Decoder     []*SeanetLayer
}

// SeanetLayer is one decoder stage; a layer with no weights is an ELU.
type SeanetLayer struct {
	Conv   *Conv
	ConvTr *ConvTr
	Res    []*Conv
}

// Conv is a causal stride-1 convolution. Weight is [out, kernel, in].
type Conv struct {
	Weight   *mlx.Array
	Bias     *mlx.Array
	Dilation int
}

// ConvTr is a transposed convolution whose kernel is twice its stride, as
// every Mimi upsampler's is. Weight is [in, kernel*out], or [kernel, channels]
// when depthwise.
type ConvTr struct {
	Weight    *mlx.Array
	Bias      *mlx.Array
	Stride    int
	Depthwise bool
}

func loadMimi(tensors map[string]*mlx.Array, cfg *config) (*Mimi, error) {
	t := cfg.Mimi.Transformer
	m := &Mimi{Context: t.Context}
	if w := tensors["mimi.quantizer.output_proj.weight"]; w != nil {
		m.OutputProj = mlx.Transpose(mlx.Squeeze(w, 2), 1, 0)
	}
	if w := tensors["mimi.upsample.convtr.convtr.weight"]; w != nil {
		m.Upsample = &ConvTr{Weight: mlx.Transpose(mlx.Squeeze(w, 1), 1, 0), Stride: w.Dim(2) / 2, Depthwise: true}
	}
	if m.OutputProj == nil || m.Upsample == nil {
		return nil, fmt.Errorf("missing mimi projection weights")
	}
	var err error
	if m.Transformer, err = loadTransformer(tensors, "mimi.decoder_transformer.transformer", t.NumLayers, t.NumHeads, t.MaxPeriod); err != nil {
		return nil, err
	}

	const prefix = "mimi.decoder.model."
	last := -1
	for name := range tensors {
		if rest, ok := strings.CutPrefix(name, prefix); ok {
			index, _, ok := strings.Cut(rest, ".")
			if i, err := strconv.Atoi(index); ok && err == nil {
				last = max(last, i)
			}
		}
	}
	dilation := 1
	for i := 0; i <= last; i++ {
		p := prefix + strconv.Itoa(i)
		l := &SeanetLayer{}
		switch {
		case tensors[p+".conv.weight"] != nil:
			l.Conv = conv(tensors, p+".conv", 1)
		case tensors[p+".convtr.weight"] != nil:
			w := tensors[p+".convtr.weight"]
			in, out, k := w.Dim(0), w.Dim(1), w.Dim(2)
			l.ConvTr = &ConvTr{
				Weight: mlx.Reshape(mlx.Transpose(w, 0, 2, 1), int32(in), int32(k*out)),
				Bias:   tensors[p+".convtr.bias"],
				Stride: k / 2,
			}
			dilation = 1
		case tensors[p+".block.1.conv.weight"] != nil:
			if tensors[p+".block.3.conv.weight"] == nil {
				return nil, fmt.Errorf("missing residual weights for %s", p)
			}
			l.Res = []*Conv{conv(tensors, p+".block.1.conv", dilation), conv(tensors, p+".block.3.conv", 1)}
			dilation *= cfg.Mimi.Seanet.DilationBase
		}
		m.Decoder = append(m.Decoder, l)
	}
	if last < 0 {
		return nil, fmt.Errorf("missing mimi decoder weights")
	}
	return m, nil
}

func conv(tensors map[string]*mlx.Array, prefix string, dilation int) *Conv {
	return &Conv{Weight: mlx.Transpose(tensors[prefix+".weight"], 0, 2, 1), Bias: tensors[prefix+".bias"], Dilation: dilation}
}

// mimiState is one stream's decoder state: convolution context, transposed
// convolution overlap, and the transformer's attention window.
type mimiState struct {
	scope  *mlx.Scope
	arrays []*mlx.Array
	next   int
	window []*windowCache
	pos    int
}

func (m *Mimi) newState() *mimiState {
	s := &mimiState{scope: mlx.NewScope()}
	for range m.Transformer.Layers {
		s.window = append(s.window, &windowCache{state: s, context: m.Context})
	}
	return s
}

func (s *mimiState) close() { s.scope.Close() }

// carry returns the state slot for the next stateful op, created from init
// on the stream's first frame, and a function to store its new value.
func (s *mimiState) carry(init func() *mlx.Array) (*mlx.Array, func(*mlx.Array)) {
	if s.next == len(s.arrays) {
		a := init()
		s.scope.Attach(a)
		s.arrays = append(s.arrays, a)
	}
	a := s.arrays[s.next]
	s.next++
	return a, a.Set
}

// decode turns normalized latents [1, T, latent] into samples [1, T*frame, 1].
func (m *Mimi) decode(s *mimiState, latents *mlx.Array) *mlx.Array {
	s.next = 0
	x := latents.Matmul(m.OutputProj)
	x = m.Upsample.forward(s, x)
	steps := x.Dim(1)
	x = m.Transformer.forward(x, s.pos, windowAttention(s.window))
	s.pos += steps
	for _, l := range m.Decoder {
		switch {
		case l.Conv != nil:
			x = l.Conv.forward(s, x)
		case l.ConvTr != nil:
			x = l.ConvTr.forward(s, x)
		case l.Res != nil:
			x = mlx.Add(x, l.Res[1].forward(s, elu(l.Res[0].forward(s, elu(x)))))
		default:
			x = elu(x)
		}
	}
	return x
}

func (c *Conv) forward(s *mimiState, x *mlx.Array) *mlx.Array {
	if pad := (c.Weight.Dim(1) - 1) * c.Dilation; pad > 0 {
		prev, store := s.carry(func() *mlx.Array { return mlx.Zeros(x.DType(), 1, pad, x.Dim(2)) })
		x = prev.Concatenate(1, x)
		store(x.Slice(mlx.Slice(), mlx.Slice(x.Dim(1)-pad, x.Dim(1)), mlx.Slice()))
	}
	return mlx.Conv1d(x, c.Weight, c.Bias, 1, 0, int32(c.Dilation), 1)
}

// forward computes every kernel tap with one matmul, then overlap-adds the
// second half of each step onto the first half of the next.
func (c *ConvTr) forward(s *mimiState, x *mlx.Array) *mlx.Array {
	T, S := x.Dim(1), c.Stride
	var z *mlx.Array
	if c.Depthwise {
		z = mlx.Mul(mlx.ExpandDims(x, 2), c.Weight)
	} else {
		z = mlx.Reshape(x.Matmul(c.Weight), 1, int32(T), int32(2*S), -1)
	}
	C := z.Dim(3)
	head := z.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, S), mlx.Slice())
	tail := z.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(S, 2*S), mlx.Slice())
	prev, store := s.carry(func() *mlx.Array { return mlx.Zeros(x.DType(), 1, 1, S, C) })
	shifted := prev.Concatenate(1, tail.Slice(mlx.Slice(), mlx.Slice(0, T-1), mlx.Slice(), mlx.Slice()))
	store(tail.Slice(mlx.Slice(), mlx.Slice(T-1, T), mlx.Slice(), mlx.Slice()))
	y := mlx.Reshape(mlx.Add(head, shifted), 1, int32(T*S), int32(C))
	if c.Bias != nil {
		y = mlx.Add(y, c.Bias)
	}
	return y
}

var elu = mlx.Compile1("ELU", func(x *mlx.Array) *mlx.Array {
	zero := mlx.FromValue[float32](0).AsType(x.DType())
	return mlx.Where(x.Greater(zero), x, mlx.AddScalar(mlx.Exp(mlx.Minimum(x, zero)), -1))
}, mlx.Shapeless())

// windowCache keeps the last context-1 keys and values of one Mimi layer.
type windowCache struct {
	state   *mimiState
	context int
	k, v    *mlx.Array
	n       int
}

func windowAttention(w []*windowCache) []attention {
	a := make([]attention, len(w))
	for i := range w {
		a[i] = w[i]
	}
	return a
}

func (c *windowCache) attend(q, k, v *mlx.Array, scale float32) *mlx.Array {
	L, n := k.Dim(2), c.n
	if c.k != nil {
		k = c.k.Concatenate(2, k)
		v = c.v.Concatenate(2, v)
	}
	K := n + L
	keep := min(c.context-1, K)
	c.n = keep
	ks := k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(K-keep, K), mlx.Slice())
	vs := v.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(K-keep, K), mlx.Slice())
	if c.k == nil {
		c.k, c.v = ks.Clone(), vs.Clone()
		c.state.scope.Attach(c.k, c.v)
	} else {
		c.k.Set(ks)
		c.v.Set(vs)
	}

	// Query i sits at key index n+i and sees keys in (n+i-context, n+i].
	mask := make([]float32, L*K)
	for i := range L {
		for j := range K {
			if j > n+i || j <= n+i-c.context {
				mask[i*K+j] = float32(math.Inf(-1))
			}
		}
	}
	return mlx.FastScaledDotProductAttention(q, k, v, scale, "", mlx.FromValues(mask, 1, 1, L, K).AsType(q.DType()))
}
