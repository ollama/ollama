package ripple

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func (m *Model) Forward(b *batch.Batch, caches []cache.Layer) (hidden, auxHidden *mlx.Array) {
	if caches != nil && len(caches) < len(m.Layers) {
		panic("ripple: cache count must cover target layers")
	}
	h := scaleEmbedding(m.EmbedTokens.Forward(b.InputIDs), m.EmbedScale)
	seed := h
	positions := mlx.Add(mlx.Reshape(mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets)), int32(len(b.SeqOffsets)), 1), mlx.Arange(0, float64(b.InputIDs.Dim(1)), 1, mlx.DTypeInt32))
	cos, sin := rotaryTables(positions, m.HeadDim, m.RopeTheta, h.DType())
	var features []*mlx.Array
	for i, layer := range m.Layers {
		var c cache.Layer
		if caches != nil {
			c = caches[i]
		}
		h = layer.Forward(h, seed, b, c, cos, sin, m.LayerTypes[i] == "sliding_attention", m.Config)
		if next := len(features); next < len(m.auxHiddenLayers) && i == m.auxHiddenLayers[next] {
			features = append(features, h)
		}
	}
	h = rmsNorm(h, m.Norm, m.RMSNormEps)
	if len(features) > 0 {
		return h, mlx.Concatenate(features, -1)
	}
	return h, h
}

func (l *Layer) Forward(x, seed *mlx.Array, b *batch.Batch, c cache.Layer, cos, sin *mlx.Array, local bool, cfg *Config) *mlx.Array {
	u := rmsNorm(x, l.InputNorm, cfg.RMSNormEps)
	a := l.Attention.Forward(u, seed, b, c, cos, sin, local, cfg)
	h := mlx.Add(x, rmsNorm(a, l.PostAttentionNorm, cfg.RMSNormEps))
	f := l.MLP.Forward(rmsNorm(h, l.PreFFNNorm, cfg.RMSNormEps), cfg.FFNClampThreshold)
	return mlx.Add(h, rmsNorm(f, l.PostFFNNorm, cfg.RMSNormEps))
}

func (a *Attention) Forward(x, seed *mlx.Array, b *batch.Batch, c cache.Layer, cos, sin *mlx.Array, local bool, cfg *Config) *mlx.Array {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	q := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(blendValues(seed, x, a.ValueResLambda)), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	q = mlx.Transpose(rmsNorm(q, a.QNorm, cfg.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(rmsNorm(k, a.KNorm, cfg.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	if local {
		q, k = applyRotary(q, cos, sin), applyRotary(k, cos, sin)
	}
	mask := nn.CausalMask()
	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		if local {
			mask = mask.Intersect(nn.SlidingWindowMask(b, int(L), cfg.SlidingWindow, k.DType()))
		}
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(gatedAttention(out, a.GProj.Forward(x)))
}

// MLP is the clipped SwiGLU feed-forward block.
type MLP struct {
	GateProj, UpProj, DownProj nn.LinearLayer
}

var ffnActivation = mlx.Compile2("SwiGLUF32", func(gate, up *mlx.Array) *mlx.Array {
	f := gate.AsType(mlx.DTypeFloat32)
	silu := mlx.Mul(f, mlx.Sigmoid(f)).AsType(gate.DType())
	return mlx.Mul(silu, up)
}, mlx.Shapeless())

var clippedFFNActivation = mlx.Compile3("ClippedSwiGLUF32", func(gate, up, threshold *mlx.Array) *mlx.Array {
	gate = mlx.Minimum(mlx.Maximum(gate, mlx.FromValue(float32(math.Inf(-1)))), threshold).AsType(gate.DType())
	up = mlx.Minimum(mlx.Maximum(up, threshold.Negative()), threshold).AsType(up.DType())
	return ffnActivation(gate, up)
}, mlx.Shapeless())

func (m *MLP) Forward(x *mlx.Array, threshold *float32) *mlx.Array {
	gate, up := m.GateProj.Forward(x), m.UpProj.Forward(x)
	var activated *mlx.Array
	if threshold != nil && *threshold > 0 {
		activated = clippedFFNActivation(gate, up, mlx.FromValue(*threshold))
	} else {
		activated = ffnActivation(gate, up)
	}
	return m.DownProj.Forward(activated)
}

func rotaryTables(positions *mlx.Array, dim int32, theta float32, dtype mlx.DType) (*mlx.Array, *mlx.Array) {
	inv := make([]float32, dim/2)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(float64(theta), float64(float32(2*i)/float32(dim))))
	}
	freq := mlx.Mul(mlx.ExpandDims(positions.AsType(mlx.DTypeFloat32), -1), mlx.FromValues(inv, len(inv)))
	freq = mlx.Concatenate([]*mlx.Array{freq, freq}, -1)
	return mlx.Cos(freq).AsType(dtype), mlx.Sin(freq).AsType(dtype)
}

func applyRotary(x, cos, sin *mlx.Array) *mlx.Array {
	f := x.AsType(mlx.DTypeFloat32)
	half := x.Dim(-1) / 2
	first := f.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, half))
	second := f.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(half, x.Dim(-1)))
	rotated := mlx.Concatenate([]*mlx.Array{second.Negative(), first}, -1)
	c := mlx.ExpandDims(cos.AsType(mlx.DTypeFloat32), 1)
	s := mlx.ExpandDims(sin.AsType(mlx.DTypeFloat32), 1)
	return mlx.Add(mlx.Mul(f, c), mlx.Mul(rotated, s)).AsType(x.DType())
}

func rmsNorm(x, weight *mlx.Array, eps float32) *mlx.Array {
	// Round the normalized activation before multiplying the learned weight.
	return mlx.Mul(mlx.RMSNormFn(x, nil, eps), weight)
}

func scaleEmbedding(x *mlx.Array, scale *float64) *mlx.Array {
	if scale == nil {
		return x
	}
	return mlx.MulScalar(x.AsType(mlx.DTypeFloat32), float32(*scale)).AsType(x.DType())
}

func attentionGate(x *mlx.Array) *mlx.Array {
	return mlx.Sigmoid(x.AsType(mlx.DTypeFloat32)).AsType(x.DType())
}

var gatedAttention = mlx.Compile2("AttentionOutputGate", func(out, gate *mlx.Array) *mlx.Array {
	return mlx.Mul(out, attentionGate(gate))
}, mlx.Shapeless())

func blendValues(seed, x, lambda *mlx.Array) *mlx.Array {
	// Sigmoid rounds to the parameter dtype; lerp has one activation rounding.
	a := mlx.Sigmoid(lambda.AsType(mlx.DTypeFloat32)).AsType(lambda.DType()).AsType(x.DType()).AsType(mlx.DTypeFloat32)
	return interpolateValues(seed, x, a)
}

var interpolateValues = mlx.Compile3("ValueBlend", func(seed, x, a *mlx.Array) *mlx.Array {
	start, end := seed.AsType(mlx.DTypeFloat32), x.AsType(mlx.DTypeFloat32)
	delta := mlx.Sub(end, start)
	low := mlx.Add(start, mlx.Mul(a, delta))
	high := mlx.Sub(end, mlx.Mul(mlx.Sub(mlx.FromValue(float32(1)), a), delta))
	return mlx.Where(a.Less(mlx.FromValue(float32(0.5))), low, high).AsType(x.DType())
}, mlx.Shapeless())
