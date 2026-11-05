package lfm3speech

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// stackConfig describes an LFM3 layer stack: sandwich-normed short
// convolutions and sliding-window gated attention, as in the codec decoder.
type stackConfig struct {
	Hidden, Heads, KVHeads, HeadDim int32
	ConvKernel                      int32
	Window                          int
	NormEps, RopeTheta, Scale       float32
}

// Layer is one LFM3 block; exactly one of Conv and Attn is set.
type Layer struct {
	OperatorNorm, PostOperatorNorm *nn.RMSNorm
	FFNNorm, PostFFNNorm           *nn.RMSNorm
	Attn                           *Attention
	Conv                           *ShortConv
	MLP                            *MLP
}

// Attention is LFM3 attention: per-head q/k RMSNorm, RoPE, and a sigmoid
// gate per head on the output.
type Attention struct {
	QProj, KProj, VProj, OProj, GateProj nn.LinearLayer
	QNorm, KNorm                         *nn.RMSNorm
}

// ShortConv is the gated causal depthwise convolution.
type ShortConv struct {
	InProj, OutProj nn.LinearLayer
	Conv1D          *nn.Conv1d
}

// MLP is the SwiGLU feed-forward.
type MLP struct {
	W1, W2, W3 nn.LinearLayer
}

func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	return m.W2.Forward(nn.SwiGLU(m.W1, m.W3, x))
}

// forward runs the block over x [B, L, hidden]; c may be nil for a whole
// sequence without state.
func (l *Layer) forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *stackConfig) *mlx.Array {
	normed := l.OperatorNorm.Forward(x, cfg.NormEps)
	var op *mlx.Array
	if l.Conv != nil {
		op = l.Conv.forward(normed, b, c, B, L, cfg)
	} else {
		op = l.Attn.forward(normed, b, c, positions, B, L, cfg)
	}
	h := mlx.Add(x, l.PostOperatorNorm.Forward(op, cfg.NormEps))
	ffn := l.PostFFNNorm.Forward(l.MLP.Forward(l.FFNNorm.Forward(h, cfg.NormEps)), cfg.NormEps)
	return mlx.Add(h, ffn)
}

// forward is causal sliding-window attention with full-head RoPE.
func (a *Attention) forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *stackConfig) *mlx.Array {
	heads := func(proj nn.LinearLayer, n int32) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(proj.Forward(x), B, L, n, cfg.HeadDim), 0, 2, 1, 3)
	}
	q := a.QNorm.Forward(heads(a.QProj, cfg.Heads), cfg.NormEps)
	k := a.KNorm.Forward(heads(a.KProj, cfg.KVHeads), cfg.NormEps)
	v := heads(a.VProj, cfg.KVHeads)
	q = mlx.RoPEWithBase(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions)
	k = mlx.RoPEWithBase(k, int(cfg.HeadDim), false, cfg.RopeTheta, 1, positions)

	var kv nn.SDPAOption
	mask := nn.CausalMask()
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		mask = mask.Intersect(nn.SlidingWindowMask(b, int(L), cfg.Window, q.DType()))
	}
	out := mlx.Transpose(nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask)), 0, 2, 1, 3)
	out = mlx.Mul(out, mlx.ExpandDims(mlx.Sigmoid(a.GateProj.Forward(x)), -1))
	return a.OProj.Forward(mlx.Reshape(out, B, L, cfg.Heads*cfg.HeadDim))
}

// forward splits in_proj into B, C, x; convolves B*x causally; gates by C.
func (s *ShortConv) forward(x *mlx.Array, b *batch.Batch, c cache.Cache, B, L int32, cfg *stackConfig) *mlx.Array {
	p := s.InProj.Forward(x)
	D := cfg.Hidden
	part := func(i int32) *mlx.Array {
		return mlx.SliceStartStop(p, []int32{0, 0, i * D}, []int32{B, L, (i + 1) * D})
	}
	input := mlx.Mul(part(0), part(2))
	tail := int(cfg.ConvKernel - 1)
	var opts []nn.RecurrentOption
	var recurrent *cache.RecurrentCache
	var delta *mlx.Array
	if c != nil {
		recurrent = c.(*cache.RecurrentCache)
		history := recurrent.Get(b, x.DType())
		delta = history.DeltaState()
		opts = append(opts, nn.WithRecurrentHistory(history))
	} else {
		opts = append(opts, nn.WithRecurrentState(mlx.Zeros(x.DType(), int(B), tail, int(D)), nil))
	}
	out, states := nn.CausalConv1D(b, input, s.Conv1D, tail, opts...)
	if recurrent != nil {
		deltas := make([]*mlx.Array, len(states))
		for i := range deltas {
			deltas[i] = delta
		}
		recurrent.Put(b, states, deltas)
	}
	return s.OutProj.Forward(mlx.Mul(part(1), out))
}
