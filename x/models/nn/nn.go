//go:build mlx

package nn

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Layer is the interface for neural network layers with a Forward method.
type Layer interface {
	Forward(x *mlx.Array) *mlx.Array
}

// LinearLayer is an interface for linear layers (both regular and quantized).
type LinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
	OutputDim() int32
}

// Linear applies an affine transformation: y = x @ W.T + b
type Linear struct {
	Weight *mlx.Array
	Bias   *mlx.Array
}

func NewLinear(weight *mlx.Array, bias *mlx.Array) *Linear {
	return &Linear{Weight: weight, Bias: bias}
}

func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	w := l.Weight.Transpose(1, 0)
	if l.Bias != nil && l.Bias.Valid() {
		return l.Bias.Addmm(x, w, 1.0, 1.0)
	}
	return x.Matmul(w)
}

func (l *Linear) OutputDim() int32 {
	return int32(l.Weight.Dim(0))
}

// QuantizedLinear applies an affine transformation using quantized weights.
type QuantizedLinear struct {
	Weight    *mlx.Array // Quantized weight data
	Scales    *mlx.Array // Scale factors for dequantization
	QBiases   *mlx.Array // Quantization biases (nil for nvfp4)
	Bias      *mlx.Array // Layer bias [output_dims] or nil
	GroupSize int
	Bits      int
	Mode      string
}

func NewQuantizedLinear(weight *mlx.Array, bias *mlx.Array, groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(weight, groupSize, bits, mode)
	if qbiases != nil {
		mlx.Eval(qw, scales, qbiases)
	} else {
		mlx.Eval(qw, scales)
	}
	return &QuantizedLinear{
		Weight:    qw,
		Scales:    scales,
		QBiases:   qbiases,
		Bias:      bias,
		GroupSize: groupSize,
		Bits:      bits,
		Mode:      mode,
	}
}

func (ql *QuantizedLinear) Forward(x *mlx.Array) *mlx.Array {
	out := mlx.QuantizedMatmul(x, ql.Weight, ql.Scales, ql.QBiases, true, ql.GroupSize, ql.Bits, ql.Mode)
	if ql.Bias != nil && ql.Bias.Valid() {
		out = out.Add(ql.Bias)
	}
	return out
}

func (ql *QuantizedLinear) OutputDim() int32 {
	return int32(ql.Weight.Dim(0))
}

// RMSNorm represents an RMS normalization layer.
type RMSNorm struct {
	Weight *mlx.Array
	Eps    float32
}

func NewRMSNorm(weight *mlx.Array, eps float32) *RMSNorm {
	return &RMSNorm{Weight: weight, Eps: eps}
}

func (rn *RMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	if eps == 0 {
		eps = rn.Eps
	}
	return mlx.RMSNormFn(x, rn.Weight, eps)
}

// Embedding represents an embedding layer.
type Embedding struct {
	Weight *mlx.Array
}

func NewEmbedding(weight *mlx.Array) *Embedding {
	return &Embedding{Weight: weight}
}

func (e *Embedding) Forward(indices *mlx.Array) *mlx.Array {
	return e.Weight.TakeAxis(indices, 0)
}

// LayerNorm represents a standard layer normalization layer (with bias).
type LayerNorm struct {
	Weight *mlx.Array
	Bias   *mlx.Array
	Eps    float32
}

func (ln *LayerNorm) Forward(x *mlx.Array) *mlx.Array {
	eps := ln.Eps
	if eps == 0 {
		eps = 1e-5
	}
	mean := mlx.Mean(x, -1, true)
	centered := x.Subtract(mean)
	variance := mlx.Mean(centered.Multiply(centered), -1, true)
	normalized := centered.Multiply(mlx.RSqrt(mlx.AddScalar(variance, eps)))
	out := normalized.Multiply(ln.Weight)
	if ln.Bias != nil && ln.Bias.Valid() {
		out = out.Add(ln.Bias)
	}
	return out
}

// MultiLinearLayer is an interface for per-head linear layers.
type MultiLinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
}

// MultiLinear performs per-head linear projections.
// Weight shape: [num_heads, output_dims, input_dims]
type MultiLinear struct {
	Weight *mlx.Array
}

func NewMultiLinear(weight *mlx.Array) *MultiLinear {
	return &MultiLinear{Weight: weight}
}

func (ml *MultiLinear) Forward(x *mlx.Array) *mlx.Array {
	wT := ml.Weight.Transpose(0, 2, 1)
	return x.Matmul(wT)
}

// RepeatKV repeats K/V tensors for grouped query attention.
func RepeatKV(x *mlx.Array, repeatFactor int32) *mlx.Array {
	if repeatFactor == 1 {
		return x
	}
	shape := x.Dims()
	x = x.ExpandDims(2)
	reps := []int32{1, 1, repeatFactor, 1, 1}
	x = mlx.Tile(x, reps)
	return mlx.Reshape(x, int32(shape[0]), int32(shape[1])*repeatFactor, int32(shape[2]), int32(shape[3]))
}

// ApplyCausalMask applies causal (lower triangular) mask to attention scores.
func ApplyCausalMask(scores *mlx.Array) *mlx.Array {
	shape := scores.Dims()
	seqLen := int32(shape[2])
	mask := mlx.Tri(seqLen, seqLen, 0)
	negInf := mlx.NewScalarArray(float32(-1e9))
	mask = mask.ExpandDims(0).ExpandDims(0)
	return mlx.Where(mask, scores, negInf)
}

// ApplyCausalMaskWithOffset applies causal mask for cached attention.
func ApplyCausalMaskWithOffset(scores *mlx.Array, offset int32) *mlx.Array {
	if offset == 0 {
		return ApplyCausalMask(scores)
	}
	shape := scores.Dims()
	queryLen := int32(shape[2])
	keyLen := int32(shape[3])
	mask := mlx.Tri(queryLen, keyLen, int(offset))
	negInf := mlx.NewScalarArray(float32(-1e9))
	mask = mask.ExpandDims(0).ExpandDims(0)
	return mlx.Where(mask, scores, negInf)
}
