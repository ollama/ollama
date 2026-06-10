package model

import (
	"log/slog"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// QuantizationParams returns default groupSize, bits, and mode for a quantization type.
func QuantizationParams(quantization string) (groupSize, bits int, mode string) {
	switch strings.ToUpper(quantization) {
	case "NVFP4":
		return 16, 4, "nvfp4"
	case "MXFP4":
		return 32, 4, "mxfp4"
	case "FP4", "Q4", "INT4":
		return 64, 4, "affine"
	case "MXFP8":
		return 32, 8, "mxfp8"
	case "FP8", "Q8", "INT8":
		return 64, 8, "affine"
	case "":
		return 0, 0, ""
	default:
		return 32, 8, "affine"
	}
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
// inference for affine packed tensors. Metadata that is inconsistent with the
// packed shapes (e.g. mixed-precision MLX checkpoints that quantize individual
// tensors at a different width than the model-level default) is overridden by
// shape-based inference.
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

	if mode == "affine" && !affineParamsMatchShapes(weight, scales, groupSize, bits) {
		inferredGroupSize, inferredBits, ok := 0, 0, false

		// Per-tensor metadata that fails shape validation typically has the
		// right group size and a stale bit width (mlx-lm mixed-precision
		// overrides usually change bits only), so resolve the bit width from
		// the recorded group size before falling back to generic heuristics.
		// Overrides that change the group size as well can still be
		// mis-inferred here; any (groupSize,bits) pair with the same packed
		// ratio is indistinguishable by shape alone.
		if fromTensor {
			if weightCols, scalesCols, colsOK := affinePackedCols(weight, scales); colsOK {
				if b, bOK := affineBitsForGroupSize(weightCols, scalesCols, groupSize); bOK {
					inferredGroupSize, inferredBits, ok = groupSize, b, true
				}
			}
		}
		if !ok {
			inferredGroupSize, inferredBits, ok = InferAffineQuantParamsFromShapes(weight, scales, groupSize, bits)
		}

		switch {
		case ok && groupSize > 0 && bits > 0:
			slog.Warn("recorded quantization params are inconsistent with packed shapes; re-inferred",
				"tensor", tensorName,
				"recorded_group_size", groupSize, "recorded_bits", bits,
				"group_size", inferredGroupSize, "bits", inferredBits)
		case !ok:
			slog.Warn("quantization params are inconsistent with packed shapes and could not be re-inferred; keeping recorded params",
				"tensor", tensorName, "group_size", groupSize, "bits", bits)
		}
		if ok {
			groupSize = inferredGroupSize
			bits = inferredBits
		}
	}

	return groupSize, bits, mode
}

func affinePackedCols(weight, scales *mlx.Array) (weightCols, scalesCols int, ok bool) {
	if weight == nil || scales == nil {
		return 0, 0, false
	}

	weightShape := weight.Dims()
	scaleShape := scales.Dims()
	if len(weightShape) == 0 || len(scaleShape) == 0 {
		return 0, 0, false
	}

	weightCols = weightShape[len(weightShape)-1]
	scalesCols = scaleShape[len(scaleShape)-1]
	if weightCols <= 0 || scalesCols <= 0 {
		return 0, 0, false
	}
	return weightCols, scalesCols, true
}

// affineParamsMatchShapes reports whether (groupSize,bits) is consistent with
// the packed weight and scale shapes.
func affineParamsMatchShapes(weight, scales *mlx.Array, groupSize, bits int) bool {
	weightCols, scalesCols, ok := affinePackedCols(weight, scales)
	if !ok {
		return false
	}
	return affineParamsMatchCols(weightCols, scalesCols, groupSize, bits)
}

// affineParamsMatchCols reports whether (groupSize,bits) is consistent with
// the packed weight and scale column counts: a uint32-packed affine weight
// holds 32/bits values per element, so weightCols*32 == scalesCols*groupSize*bits.
func affineParamsMatchCols(weightCols, scalesCols, groupSize, bits int) bool {
	if weightCols <= 0 || scalesCols <= 0 || groupSize <= 0 || bits <= 0 {
		return false
	}
	return weightCols*32 == scalesCols*groupSize*bits
}

// affineBitsForGroupSize returns the bit width implied by the packed column
// counts for a known group size. The packed ratio weightCols*32/scalesCols
// equals bits*groupSize, so a known group size uniquely determines the bit
// width.
func affineBitsForGroupSize(weightCols, scalesCols, groupSize int) (bits int, ok bool) {
	if weightCols <= 0 || scalesCols <= 0 || groupSize <= 0 {
		return 0, false
	}
	if (weightCols*32)%scalesCols != 0 {
		return 0, false
	}
	ratio := weightCols * 32 / scalesCols
	if ratio%groupSize != 0 {
		return 0, false
	}
	bits = ratio / groupSize
	if !isSupportedAffineBits(bits) {
		return 0, false
	}
	return bits, true
}

// InferAffineQuantParamsFromShapes infers (groupSize,bits) for affine quantized
// tensors from packed weight and scale shapes.
func InferAffineQuantParamsFromShapes(weight, scales *mlx.Array, hintGroupSize, hintBits int) (groupSize, bits int, ok bool) {
	weightCols, scalesCols, ok := affinePackedCols(weight, scales)
	if !ok {
		return 0, 0, false
	}
	return inferAffineQuantParamsFromCols(weightCols, scalesCols, hintGroupSize, hintBits)
}

// inferAffineQuantParamsFromCols infers (groupSize,bits) from packed weight
// and scale column counts.
func inferAffineQuantParamsFromCols(weightCols, scalesCols, hintGroupSize, hintBits int) (groupSize, bits int, ok bool) {
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

	// Mixed-precision checkpoints may quantize individual tensors at widths
	// the heuristics above do not cover (e.g. 6-bit experts in a 4-bit model).
	if hintGroupSize > 0 {
		if b, ok := affineBitsForGroupSize(weightCols, scalesCols, hintGroupSize); ok {
			return hintGroupSize, b, true
		}
	}

	return 0, 0, false
}

// SupportsGatherQMM reports whether the bundled MLX Metal kernels implement
// gather_qmm for the given quantization mode and bit width. MLX v0.31.2
// instantiates affine gather_qmm kernels for bits {2,3,4,5,6,8} and group
// sizes {32,64,128} (mlx/backend/metal/kernels/quantized.metal,
// instantiate_quantized_all); mxfp8 is 8-bit only, nvfp4 and mxfp4 are 4-bit
// only.
func SupportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return isSupportedAffineBits(bits)
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

// isSupportedAffineBits reports whether the bundled MLX kernels are
// instantiated for an affine bit width (see SupportsGatherQMM).
func isSupportedAffineBits(bits int) bool {
	switch bits {
	case 2, 3, 4, 5, 6, 8:
		return true
	default:
		return false
	}
}

func isCommonGroupSize(v int) bool {
	switch v {
	case 16, 32, 64, 128:
		return true
	default:
		return false
	}
}
