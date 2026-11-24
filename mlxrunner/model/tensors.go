package model

import "github.com/ollama/ollama/mlx"

// TensorAny returns the first of keys present in tensors, and that key.
func TensorAny(tensors map[string]*mlx.Array, keys ...string) (*mlx.Array, string) {
	for _, k := range keys {
		if v := tensors[k]; v != nil {
			return v, k
		}
	}
	return nil, ""
}

// FreeTensorKeys removes the keys that hold a tensor, skipping empty keys.
func FreeTensorKeys(tensors map[string]*mlx.Array, keys ...string) {
	for _, k := range keys {
		if k == "" {
			continue
		}
		if t := tensors[k]; t != nil {
			delete(tensors, k)
		}
	}
}

// SupportsGatherQMM reports whether MLX has a GatherQMM kernel for the
// quantization mode and bit width.
func SupportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return bits == 4 || bits == 8
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

// StackAndClone stacks parts along a new leading axis into a fresh array, or
// returns nil when there are no parts.
func StackAndClone(parts []*mlx.Array) *mlx.Array {
	if len(parts) == 0 {
		return nil
	}
	return mlx.Stack(parts, 0).Clone()
}

// TransposeExpertWeightForGatherMM returns stacked [E, out, in] expert
// weights as the [E, in, out] layout GatherMM consumes. The result is a
// transposed view built once at load, so the forward path adds no transpose
// per call; the weight is not copied. Other ranks are returned unchanged.
func TransposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || w.NumDims() != 3 {
		return w
	}
	return mlx.Transpose(w, 0, 2, 1).Clone()
}

// FuseExpertStacks concatenates two expert stacks along axis into a fresh
// array, or returns nil when either is missing.
func FuseExpertStacks(a, b *mlx.Array, axis int) *mlx.Array {
	if a == nil || b == nil {
		return nil
	}
	return mlx.Concatenate([]*mlx.Array{a, b}, axis).Clone()
}
