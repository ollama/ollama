//go:build mlx

// Package nn provides neural network layer types.
package nn

import "github.com/ollama/ollama/x/imagegen/mlx"

// Layer is the interface for neural network layers with a Forward method.
type Layer interface {
	Forward(x *mlx.Array) *mlx.Array
}

// LinearLayer is an interface for linear layers (both regular and quantized).
// This allows swapping between Linear and QuantizedLinear at runtime.
type LinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
	OutputDim() int32 // Returns the output dimension of the layer
}

// Linear applies an affine transformation: y = x @ W.T + b
// Weight is stored as [out_features, in_features], matching PyTorch/MLX convention.
type Linear struct {
	Weight *mlx.Array `weight:"weight"`          // [out_features, in_features]
	Bias   *mlx.Array `weight:"bias,optional"`   // [out_features] or nil
}

// NewLinear creates a linear layer.
// Weight should be [out_features, in_features].
func NewLinear(weight *mlx.Array, bias *mlx.Array) *Linear {
	return &Linear{Weight: weight, Bias: bias}
}

// NewQuantizedLinear creates a quantized linear layer directly from bf16 weights.
// Quantizes the weight immediately and evaluates to break lazy dependencies.
func NewQuantizedLinear(weight *mlx.Array, bias *mlx.Array, groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(weight, groupSize, bits, mode)
	// Eval immediately so bf16 weight can be freed
	mlx.Eval(qw, scales, qbiases)
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

// Forward applies the linear transformation: x @ W.T + bias
func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	w := mlx.Transpose(l.Weight, 1, 0)
	if l.Bias != nil {
		return mlx.AddMM(l.Bias, x, w, 1.0, 1.0)
	}
	return mlx.Linear(x, w)
}

// OutputDim returns the output dimension of the linear layer.
func (l *Linear) OutputDim() int32 {
	return l.Weight.Shape()[0]
}

// ToQuantized converts this Linear to a QuantizedLinear.
func (l *Linear) ToQuantized(groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(l.Weight, groupSize, bits, mode)
	return &QuantizedLinear{
		Weight:    qw,
		Scales:    scales,
		QBiases:   qbiases,
		Bias:      l.Bias,
		GroupSize: groupSize,
		Bits:      bits,
		Mode:      mode,
	}
}

// QuantizedLinear applies an affine transformation using quantized weights.
// Equivalent to mlx.nn.QuantizedLinear.
type QuantizedLinear struct {
	Weight    *mlx.Array // Quantized weight data
	Scales    *mlx.Array // Scale factors for dequantization
	QBiases   *mlx.Array // Quantization biases (NOT layer bias)
	Bias      *mlx.Array // Layer bias [output_dims] or nil
	GroupSize int
	Bits      int
	Mode      string
}

// Forward applies the quantized linear transformation.
func (ql *QuantizedLinear) Forward(x *mlx.Array) *mlx.Array {
	out := mlx.QuantizedMatmul(x, ql.Weight, ql.Scales, ql.QBiases, true, ql.GroupSize, ql.Bits, ql.Mode)
	if ql.Bias != nil {
		out = mlx.Add(out, ql.Bias)
	}
	return out
}

// OutputDim returns the output dimension of the quantized linear layer.
// For mxfp8/mxfp4, quantized weight shape is [out_features, in_features / group_size].
// The output dimension is the first dimension of the weight.
func (ql *QuantizedLinear) OutputDim() int32 {
	return ql.Weight.Shape()[0]
}

// RMSNorm represents an RMS normalization layer.
type RMSNorm struct {
	Weight *mlx.Array `weight:"weight"`
	Eps    float32    // optional: used if Forward called with eps=0
}

// NewRMSNorm creates an RMSNorm layer (for models not using weight loader).
func NewRMSNorm(weight *mlx.Array, eps float32) *RMSNorm {
	return &RMSNorm{Weight: weight, Eps: eps}
}

// Forward applies RMS normalization. If eps=0, uses stored Eps.
func (rn *RMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	if eps == 0 {
		eps = rn.Eps
	}
	return mlx.RMSNorm(x, rn.Weight, eps)
}

// Embedding represents an embedding layer.
type Embedding struct {
	Weight *mlx.Array `weight:"weight"`
}

// NewEmbedding creates an embedding layer.
func NewEmbedding(weight *mlx.Array) *Embedding {
	return &Embedding{Weight: weight}
}

// Forward looks up embeddings by indices.
func (e *Embedding) Forward(indices *mlx.Array) *mlx.Array {
	return mlx.Take(e.Weight, indices, 0)
}

// RepeatKV repeats K/V tensors for grouped query attention
// x: [B, num_kv_heads, S, head_dim] -> [B, num_heads, S, head_dim]
func RepeatKV(x *mlx.Array, repeatFactor int32) *mlx.Array {
	if repeatFactor == 1 {
		return x
	}
	shape := x.Shape()
	// [B, num_kv_heads, S, head_dim] -> [B, num_kv_heads, 1, S, head_dim]
	x = mlx.ExpandDims(x, 2)
	// Repeat along the new axis
	reps := []int32{1, 1, repeatFactor, 1, 1}
	x = mlx.Tile(x, reps)
	// Reshape: [B, num_kv_heads, repeat, S, head_dim] -> [B, num_kv_heads * repeat, S, head_dim]
	return mlx.Reshape(x, shape[0], shape[1]*repeatFactor, shape[2], shape[3])
}

// ApplyCausalMask applies causal (lower triangular) mask to attention scores
func ApplyCausalMask(scores *mlx.Array) *mlx.Array {
	// scores: [B, num_heads, S, S]
	shape := scores.Shape()
	seqLen := shape[2]

	// Create causal mask: 1 for positions to keep, 0 for positions to mask
	mask := mlx.Tri(seqLen, seqLen, 0)

	// Where mask is 0, set score to -inf
	negInf := mlx.NewScalarArray(float32(-1e9))

	// Broadcast mask to match scores shape
	mask = mlx.ExpandDims(mlx.ExpandDims(mask, 0), 0) // [1, 1, S, S]

	// Use where: if mask > 0, keep scores, else -inf
	return mlx.Where(mask, scores, negInf)
}

// ApplyCausalMaskWithOffset applies causal mask for cached attention
// scores: [B, num_heads, queryLen, keyLen] where keyLen = cacheLen + queryLen
// offset: the starting position of the new queries (i.e., cache length)
func ApplyCausalMaskWithOffset(scores *mlx.Array, offset int32) *mlx.Array {
	if offset == 0 {
		return ApplyCausalMask(scores)
	}

	shape := scores.Shape()
	queryLen := shape[2]
	keyLen := shape[3]

	// For cached attention, new queries can attend to all cached keys plus
	// new keys up to and including their position.
	mask := mlx.Tri(queryLen, keyLen, int(offset))

	negInf := mlx.NewScalarArray(float32(-1e9))
	mask = mlx.ExpandDims(mlx.ExpandDims(mask, 0), 0) // [1, 1, queryLen, keyLen]

	return mlx.Where(mask, scores, negInf)
}

// LayerNorm represents a standard layer normalization layer (with bias).
type LayerNorm struct {
	Weight *mlx.Array `weight:"weight"`
	Bias   *mlx.Array `weight:"bias"`
	Eps    float32
}

// Forward applies layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
func (ln *LayerNorm) Forward(x *mlx.Array) *mlx.Array {
	eps := ln.Eps
	if eps == 0 {
		eps = 1e-5
	}
	// Compute mean and variance along last dimension
	mean := mlx.Mean(x, -1, true)
	centered := mlx.Sub(x, mean)
	variance := mlx.Mean(mlx.Mul(centered, centered), -1, true)
	normalized := mlx.Mul(centered, mlx.RSqrt(mlx.AddScalar(variance, eps)))

	// Scale and shift
	out := mlx.Mul(normalized, ln.Weight)
	if ln.Bias != nil {
		out = mlx.Add(out, ln.Bias)
	}
	return out
}
