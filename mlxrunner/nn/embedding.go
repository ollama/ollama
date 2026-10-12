package nn

import "github.com/ollama/ollama/mlx"

// EmbeddingLayer is an interface for embedding layers that can also expose a
// tied-output projection when the model reuses embedding weights as the LM head.
type EmbeddingLayer interface {
	Forward(indices *mlx.Array) *mlx.Array
	AsLinear() LinearLayer
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

func (e *Embedding) AsLinear() LinearLayer {
	return NewLinear(e.Weight, nil)
}

// QuantizedEmbedding performs row-wise embedding lookup from affine/nvfp4/etc.
// packed weights and dequantizes only the selected rows.
type QuantizedEmbedding struct {
	Weight      *mlx.Array
	Scales      *mlx.Array
	QBiases     *mlx.Array
	GlobalScale *mlx.Array // Per-tensor global scale for double-scale nvfp4 (nil for standard)
	GroupSize   int
	Bits        int
	Mode        string
}

func (qe *QuantizedEmbedding) Forward(indices *mlx.Array) *mlx.Array {
	weight := qe.Weight.TakeAxis(indices, 0)
	scales := qe.Scales.TakeAxis(indices, 0)
	var qbiases *mlx.Array
	if qe.QBiases != nil {
		qbiases = qe.QBiases.TakeAxis(indices, 0)
	}
	return mlx.Dequantize(weight, scales, qbiases, qe.GroupSize, qe.Bits, qe.Mode, qe.GlobalScale)
}

func (qe *QuantizedEmbedding) AsLinear() LinearLayer {
	return &QuantizedLinear{
		Weight:      qe.Weight,
		Scales:      qe.Scales,
		QBiases:     qe.QBiases,
		GlobalScale: qe.GlobalScale,
		GroupSize:   qe.GroupSize,
		Bits:        qe.Bits,
		Mode:        qe.Mode,
	}
}
