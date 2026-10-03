package model

import "github.com/ollama/ollama/mlx"

// Import rewrites every vendor spelling to ".global_scale"; "_scale_2" is
// ModelOpt's own name, reached when a checkpoint skips import.
var globalScaleSuffixes = []string{".global_scale", "_scale_2"}

// These scale the activations, never the weight, but are freed alongside it.
var activationScaleSuffixes = []string{".input_global_scale", ".input_scale"}

// ReadGlobalScale returns a weight's NVFP4 global scale in MLX's
// representation, and the companion keys the caller should release. Candidate
// keys are tried in order, so pass the resolved tensor key before any base.
func ReadGlobalScale(tensors map[string]*mlx.Array, weightKeys ...string) (*mlx.Array, []string) {
	var found *mlx.Array
	var consumed []string
	for _, key := range weightKeys {
		if key == "" {
			continue
		}
		for _, suffix := range globalScaleSuffixes {
			scale, ok := tensors[key+suffix]
			if !ok || scale == nil {
				continue
			}
			if found == nil {
				found = scale
			}
			consumed = append(consumed, key+suffix)
		}
		for _, suffix := range activationScaleSuffixes {
			if _, ok := tensors[key+suffix]; ok {
				consumed = append(consumed, key+suffix)
			}
		}
	}
	return ToMLXGlobalScale(found), consumed
}

// ToMLXGlobalScale converts a checkpoint multiplier into the representation
// every global scale is held in once loaded. Shape is flattened too: a scalar
// ships as either [] or [1], and stacking a mix of the two fails.
func ToMLXGlobalScale(globalScale *mlx.Array) *mlx.Array {
	if globalScale == nil {
		return nil
	}
	flat := mlx.Reshape(globalScale.AsType(mlx.DTypeFloat32), int32(globalScale.Size()))
	return mlx.MulScalar(flat, mlx.Nvfp4MaxProduct)
}

// PrepareGatherQMMGlobalScale broadcasts an already-converted global scale
// into the one-entry-per-expert bank gather_qmm wants. Materialized dense: the
// kernel indexes it by raw offset, and a broadcast view is one element of
// storage.
func PrepareGatherQMMGlobalScale(globalScale *mlx.Array, numExperts int) *mlx.Array {
	if globalScale == nil {
		return nil
	}
	return mlx.Contiguous(mlx.BroadcastTo(globalScale, int32(numExperts)), false)
}

// GatherQMMIdentityScale is the scale that leaves an expert bank unscaled,
// for rows folded into a scaled bank without a scale of their own.
func GatherQMMIdentityScale() *mlx.Array {
	return mlx.FromValues([]float32{mlx.Nvfp4MaxProduct}, 1)
}

// SameGlobalScales reports whether two prepared banks hold the same scale for
// every expert, which is what lets two projections share one fused bank.
func SameGlobalScales(a, b *mlx.Array) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a == b {
		return true
	}
	if a.Size() != b.Size() {
		return false
	}
	mlx.Eval(a, b)
	aValues, bValues := a.Floats(), b.Floats()
	for i := range aValues {
		if aValues[i] != bValues[i] {
			return false
		}
	}
	return true
}
