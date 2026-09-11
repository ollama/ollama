package model

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/quant"
)

// PrepareGatherQMMGlobalScale converts a checkpoint's NVFP4 multiplier into
// the one-float32-per-expert form gather_qmm wants, broadcasting a
// checkpoint-wide scalar to the expert count. Materialized dense: the kernel
// indexes it by raw offset, and a broadcast view is one element of storage.
func PrepareGatherQMMGlobalScale(globalScale *mlx.Array, numExperts int) *mlx.Array {
	if globalScale == nil {
		return nil
	}
	globalScale = mlx.Reshape(globalScale.AsType(mlx.DTypeFloat32), int32(globalScale.Size()))
	globalScale = mlx.BroadcastTo(globalScale, int32(numExperts))
	return mlx.Contiguous(mlx.MulScalar(globalScale, mlx.Nvfp4MaxProduct), false)
}

// GatherQMMIdentityScale is the scale that leaves an expert bank unscaled,
// for rows folded into a scaled bank without a scale of their own.
func GatherQMMIdentityScale() *mlx.Array {
	return mlx.NewScalarArray(float32(mlx.Nvfp4MaxProduct))
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

// QuantizationParams returns default groupSize, bits, and mode for a
// quantization type. The values live in the shared x/quant package so the
// importer, the runtime loader, and `ollama show` agree on them.
func QuantizationParams(quantization string) (groupSize, bits int, mode string) {
	return quant.Params(quantization)
}

// TensorQuantParams resolves quant params for a tensor using per-tensor metadata
// when available, otherwise falling back to the provided model defaults.
func TensorQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
) (groupSize, bits int, mode string, fromTensor bool) {
	if tensorQuant != nil {
		if tq := tensorQuant[tensorName]; tq != nil {
			groupSize, bits, mode = QuantizationParams(tq.QuantType)
			if tq.GroupSize > 0 {
				groupSize = tq.GroupSize
			}
			return groupSize, bits, mode, true
		}
	}
	return defaultGroupSize, defaultBits, defaultMode, false
}

// ResolveLinearQuantParams resolves quantization params for a quantized linear
// tensor, preferring per-tensor metadata and falling back to shape-based
// inference for affine packed tensors.
func ResolveLinearQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
	weight, scales *mlx.Array,
) (groupSize, bits int, mode string) {
	groupSize, bits, mode, fromTensor := TensorQuantParams(
		defaultGroupSize,
		defaultBits,
		defaultMode,
		tensorQuant,
		tensorName,
	)

	if mode == "affine" {
		if inferredGroupSize, inferredBits, ok := InferAffineQuantParamsFromShapes(weight, scales, bits); ok {
			if !fromTensor || groupSize == 0 || bits == 0 {
				groupSize = inferredGroupSize
				bits = inferredBits
			}
		}
	}

	return groupSize, bits, mode
}

// InferAffineQuantParamsFromShapes infers (groupSize,bits) for affine quantized
// tensors from packed weight and scale shapes.
func InferAffineQuantParamsFromShapes(weight, scales *mlx.Array, hintBits int) (groupSize, bits int, ok bool) {
	if weight == nil || scales == nil {
		return 0, 0, false
	}

	weightShape := weight.Dims()
	scaleShape := scales.Dims()
	if len(weightShape) == 0 || len(scaleShape) == 0 {
		return 0, 0, false
	}

	weightCols := weightShape[len(weightShape)-1]
	scalesCols := scaleShape[len(scaleShape)-1]
	if weightCols <= 0 || scalesCols <= 0 {
		return 0, 0, false
	}

	groupSize4 := weightCols * 8 / scalesCols
	groupSize8 := weightCols * 4 / scalesCols

	switch {
	case groupSize4 == 32:
		return 32, 4, true
	case groupSize8 == 64:
		return 64, 8, true
	case groupSize4 == 64 && groupSize8 == 32:
		if hintBits == 8 {
			return 32, 8, true
		}
		if hintBits == 4 {
			return 64, 4, true
		}
	}

	if isCommonGroupSize(groupSize4) && !isCommonGroupSize(groupSize8) {
		return groupSize4, 4, true
	}
	if isCommonGroupSize(groupSize8) && !isCommonGroupSize(groupSize4) {
		return groupSize8, 8, true
	}

	return 0, 0, false
}

func isCommonGroupSize(v int) bool {
	switch v {
	case 16, 32, 64, 128:
		return true
	default:
		return false
	}
}
