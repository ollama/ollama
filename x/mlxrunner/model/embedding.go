package model

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// MakeEmbeddingLayer constructs an embedding layer from a tensor map.
//
// For quantized tensors (path.weight + path.weight_scale), it returns a
// QuantizedEmbedding using the same quant metadata path that linear layers use.
// For non-quantized tensors, it returns a standard dense embedding.
func MakeEmbeddingLayer(
	tensors map[string]*mlx.Array,
	path string,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
) nn.EmbeddingLayer {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		groupSize, bits, mode := ResolveLinearQuantParams(
			defaultGroupSize,
			defaultBits,
			defaultMode,
			tensorQuant,
			path+".weight",
			w,
			scales,
		)

		// Check for per-tensor global scale (NVIDIA double-scale nvfp4).
		// NVIDIA ModelOpt stores this as "weight_scale_2"; our import
		// pipeline maps it to "weight.global_scale".
		globalScale := tensors[path+".weight.global_scale"]
		if globalScale == nil {
			globalScale = tensors[path+".weight_scale_2"]
		}

		return &nn.QuantizedEmbedding{
			Weight:      w,
			Scales:      scales,
			QBiases:     qbiases,
			GlobalScale: globalScale,
			GroupSize:   groupSize,
			Bits:        bits,
			Mode:        mode,
		}
	}

	return nn.NewEmbedding(w)
}
