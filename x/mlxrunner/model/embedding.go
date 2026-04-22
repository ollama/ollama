package model

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// MakeEmbeddingLayer constructs an embedding layer from a tensor map.
//
// For quantized tensors it returns a QuantizedEmbedding using the same quant
// metadata path that linear layers use. For non-quantized tensors it returns
// a standard dense embedding.
//
// Two scale/qbias naming conventions are recognised — see MakeLinearLayer for
// the rationale:
//   - Ollama-native dot-child singular: "<path>.weight_scale" / "<path>.weight_qbias"
//   - mlx-lm sibling plural: "<path>.scales" / "<path>.biases"
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
	if scales == nil {
		scales = tensors[path+".scales"]
	}
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		if qbiases == nil {
			qbiases = tensors[path+".biases"]
		}
		groupSize, bits, mode := ResolveLinearQuantParams(
			defaultGroupSize,
			defaultBits,
			defaultMode,
			tensorQuant,
			path+".weight",
			w,
			scales,
		)

		return nn.NewQuantizedEmbedding(w, scales, qbiases, groupSize, bits, mode)
	}

	return nn.NewEmbedding(w)
}
