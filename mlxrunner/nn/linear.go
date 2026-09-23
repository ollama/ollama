package nn

import "github.com/ollama/ollama/mlx"

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
	if bias != nil && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
	}
	return &Linear{Weight: weight, Bias: bias}
}

func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	w := l.Weight.Transpose(1, 0)
	if l.Bias != nil {
		return l.Bias.Addmm(x, w, 1.0, 1.0)
	}
	return x.Matmul(w)
}

func (l *Linear) OutputDim() int32 {
	return int32(l.Weight.Dim(0))
}

// QuantizedLinear applies an affine transformation using quantized weights.
type QuantizedLinear struct {
	Weight      *mlx.Array // Quantized weight data
	Scales      *mlx.Array // Scale factors for dequantization
	QBiases     *mlx.Array // Quantization biases (nil for nvfp4)
	Bias        *mlx.Array // Layer bias [output_dims] or nil
	GlobalScale *mlx.Array // Per-tensor or per-row global scale for double-scale nvfp4 (nil for standard)
	GroupSize   int
	Bits        int
	Mode        string
}

func NewQuantizedLinear(weight *mlx.Array, bias *mlx.Array, groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(weight, groupSize, bits, mode)
	if qbiases != nil {
		mlx.Eval(qw, scales, qbiases)
	} else {
		mlx.Eval(qw, scales)
	}
	if bias != nil && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
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
	// Double-scale nvfp4 (e.g., NVIDIA ModelOpt) applies the per-tensor or
	// per-row global scale inside QuantizedMatmul.
	out := mlx.QuantizedMatmul(x, ql.Weight, ql.Scales, ql.QBiases, true, ql.GroupSize, ql.Bits, ql.Mode, ql.GlobalScale)
	if ql.Bias != nil {
		bias := ql.Bias
		if bias.DType() != out.DType() {
			bias = bias.AsType(out.DType())
		}
		out = out.Add(bias)
	}
	return out
}

func (ql *QuantizedLinear) OutputDim() int32 {
	return int32(ql.Weight.Dim(0))
}
