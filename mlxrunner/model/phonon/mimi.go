package phonon

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/cache"
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
	t := cfg.Mimi
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
	if m.Transformer, err = loadTransformer(tensors, "mimi.decoder_transformer.transformer", t.NumLayers, t.NumHeads, t.MaxPeriod, t.GELU); err != nil {
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
			dilation *= cfg.Mimi.DilationBase
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
	scope     *mlx.Scope
	conv      map[*Conv]*mlx.Array
	transpose map[*ConvTr]*mlx.Array
	attention []cache.Attention
	pos       int
}

func (m *Mimi) newState() *mimiState {
	s := &mimiState{
		scope:     mlx.NewScope(),
		conv:      make(map[*Conv]*mlx.Array),
		transpose: make(map[*ConvTr]*mlx.Array),
	}
	for range m.Transformer.Layers {
		s.attention = append(s.attention, cache.NewRotatingKVCache(m.Context))
	}
	return s
}

func (s *mimiState) close() {
	for _, c := range s.attention {
		c.Free()
	}
	s.scope.Close()
}

// decode turns normalized latents [1, T, latent] into samples [1, T*frame, 1].
func (m *Mimi) decode(s *mimiState, latents *mlx.Array) *mlx.Array {
	x := latents.Matmul(m.OutputProj)
	x = m.Upsample.forward(s, x)
	steps := x.Dim(1)
	x = m.Transformer.forward(x, s.pos, s.attention)
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
		prev := s.conv[c]
		if prev == nil {
			prev = mlx.Zeros(x.DType(), 1, pad, x.Dim(2))
			s.scope.Attach(prev)
			s.conv[c] = prev
		}
		x = prev.Concatenate(1, x)
		prev.Set(x.Slice(mlx.Slice(), mlx.Slice(x.Dim(1)-pad, x.Dim(1)), mlx.Slice()))
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
	prev := s.transpose[c]
	if prev == nil {
		prev = mlx.Zeros(x.DType(), 1, 1, S, C)
		s.scope.Attach(prev)
		s.transpose[c] = prev
	}
	shifted := prev.Concatenate(1, tail.Slice(mlx.Slice(), mlx.Slice(0, T-1), mlx.Slice(), mlx.Slice()))
	prev.Set(tail.Slice(mlx.Slice(), mlx.Slice(T-1, T), mlx.Slice(), mlx.Slice()))
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
