//go:build mlx

package model

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// LinearFactory builds linear layers using shared tensor maps and quant defaults.
type LinearFactory struct {
	tensors          map[string]*mlx.Array
	defaultGroupSize int
	defaultBits      int
	defaultMode      string
	tensorQuant      map[string]*TensorQuantInfo
}

// NewLinearFactory creates a reusable constructor for model linear layers.
func NewLinearFactory(
	tensors map[string]*mlx.Array,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
) LinearFactory {
	return LinearFactory{
		tensors:          tensors,
		defaultGroupSize: defaultGroupSize,
		defaultBits:      defaultBits,
		defaultMode:      defaultMode,
		tensorQuant:      tensorQuant,
	}
}

// Make constructs a linear layer at path.
func (f LinearFactory) Make(path string) nn.LinearLayer {
	return MakeLinearLayer(
		f.tensors,
		path,
		f.defaultGroupSize,
		f.defaultBits,
		f.defaultMode,
		f.tensorQuant,
	)
}

// MakeLinearLayer constructs a linear layer from a tensor map.
//
// For quantized tensors (path.weight + path.weight_scale), it resolves per-tensor
// quant params via TensorQuant metadata (with shape-based affine fallback).
// For non-quantized tensors, it returns a standard nn.Linear.
func MakeLinearLayer(
	tensors map[string]*mlx.Array,
	path string,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
) nn.LinearLayer {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		bias := tensors[path+".bias"]

		groupSize, bits, mode := ResolveLinearQuantParams(
			defaultGroupSize,
			defaultBits,
			defaultMode,
			tensorQuant,
			path+".weight",
			w,
			scales,
		)

		return &nn.QuantizedLinear{
			Weight:    w,
			Scales:    scales,
			QBiases:   qbiases,
			Bias:      bias,
			GroupSize: groupSize,
			Bits:      bits,
			Mode:      mode,
		}
	}

	bias := tensors[path+".bias"]
	return nn.NewLinear(w, bias)
}
