package model

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
)

// QuantDefaults are a model's default quantization parameters and its
// per-tensor overrides, used to resolve each expert bank's parameters.
type QuantDefaults struct {
	GroupSize   int
	Bits        int
	Mode        string
	TensorQuant map[string]*TensorQuantInfo
}

// ExpertBank is one projection of an MoE layer for every expert, as stored:
// a [experts, out, in] weight, packed when quantized, with its companions.
// It makes no policy decisions; callers choose whether to keep it quantized,
// dequantize it, or transpose it for GatherMM.
type ExpertBank struct {
	Weight      *mlx.Array
	Scales      *mlx.Array // nil when the bank is dense
	Biases      *mlx.Array // affine quantization biases, or nil
	GlobalScale *mlx.Array // [experts], MLX representation, or nil
	GroupSize   int
	Bits        int
	Mode        string

	Key  string   // the stacked tensor's key, or expert 0's key
	keys []string // the tensors the bank was read from
}

// Quantized reports whether the bank holds packed weights with scales.
func (b *ExpertBank) Quantized() bool {
	return b.Scales != nil
}

// Dense returns the bank as a dense [experts, out, in] array, dequantizing it
// and applying its global scale when it has them.
func (b *ExpertBank) Dense() *mlx.Array {
	if b.Quantized() {
		return mlx.Dequantize(b.Weight, b.Scales, b.Biases, b.GroupSize, b.Bits, b.Mode, b.GlobalScale)
	}
	if b.GlobalScale == nil {
		return b.Weight
	}
	// Divide out MLX's representation while still in float32: these
	// multipliers are ~1e-3, so narrowing first would cost most of the mantissa.
	scale := mlx.DivScalar(b.GlobalScale, mlx.Nvfp4MaxProduct)
	if scale.DType() != b.Weight.DType() {
		scale = scale.AsType(b.Weight.DType())
	}
	for scale.NumDims() < b.Weight.NumDims() {
		scale = mlx.ExpandDims(scale, -1)
	}
	return mlx.Mul(b.Weight, scale)
}

// Release drops the tensors the bank was read from. Call it once the bank, or
// whatever the caller derived from it, holds everything it needs.
func (b *ExpertBank) Release(tensors map[string]*mlx.Array) {
	for _, key := range b.keys {
		delete(tensors, key)
	}
}

// SupportsGatherQMM reports whether MLX's gather_qmm kernels run a mode and
// bit width; other quantized banks must be dequantized for GatherMM.
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

// TransposeForGatherMM returns a dense [experts, out, in] bank as the
// [experts, in, out] array GatherMM multiplies by.
func TransposeForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || w.NumDims() != 3 {
		return w
	}
	return mlx.Transpose(w, 0, 2, 1).Clone()
}

// ReadExpertBank finds one projection's expert bank under the first of
// prefixes that has it, in whichever form it was stored:
//
//   - stacked: "<prefix>.<proj>.weight", or "<prefix>.<proj>" when the
//     publisher names the stacked tensor without ".weight";
//   - per expert: "<prefix>.<i>.<proj>.weight" (or without ".weight") for
//     every i below experts, stacked here.
//
// Companions follow the runner's names: "<key>_scale", "<key>_qbias", and a
// global scale under "<key>.global_scale" or "<key>_scale_2". It returns nil
// and no error when no prefix has the projection, and an error when a bank is
// present but incomplete or inconsistent.
func ReadExpertBank(tensors map[string]*mlx.Array, quant QuantDefaults, experts int, proj string, prefixes ...string) (*ExpertBank, error) {
	for _, prefix := range prefixes {
		base := prefix + "." + proj
		for _, key := range []string{base + ".weight", base} {
			if w := tensors[key]; w != nil {
				return readStackedExpertBank(tensors, quant, experts, key, w)
			}
		}
		first := prefix + ".0." + proj
		for _, suffix := range []string{".weight", ""} {
			if tensors[first+suffix] != nil {
				return readPerExpertBank(tensors, quant, experts, prefix, proj, suffix)
			}
		}
	}
	return nil, nil
}

func readStackedExpertBank(tensors map[string]*mlx.Array, quant QuantDefaults, experts int, key string, w *mlx.Array) (*ExpertBank, error) {
	if w.NumDims() != 3 || (experts > 0 && w.Dim(0) != experts) {
		return nil, fmt.Errorf("expert bank %s has shape %v, want [%d, out, in]", key, w.Dims(), experts)
	}
	b := &ExpertBank{Weight: w, Key: key, keys: []string{key}}
	globalScale, globalKeys := ReadGlobalScale(tensors, key, key+".weight")
	b.keys = append(b.keys, globalKeys...)
	if globalScale != nil {
		if globalScale.Size() != 1 && globalScale.Size() != w.Dim(0) {
			return nil, fmt.Errorf("expert bank %s has %d global scales for %d experts", key, globalScale.Size(), w.Dim(0))
		}
		b.GlobalScale = PrepareGatherQMMGlobalScale(globalScale, w.Dim(0))
	}
	if scales := tensors[key+"_scale"]; scales != nil {
		b.Scales = scales
		b.keys = append(b.keys, key+"_scale")
		if biases := tensors[key+"_qbias"]; biases != nil {
			b.Biases = biases
			b.keys = append(b.keys, key+"_qbias")
		}
		b.GroupSize, b.Bits, b.Mode = ResolveLinearQuantParams(quant.GroupSize, quant.Bits, quant.Mode, quant.TensorQuant, key, w, scales)
	}
	return b, nil
}

func readPerExpertBank(tensors map[string]*mlx.Array, quant QuantDefaults, experts int, prefix, proj, suffix string) (*ExpertBank, error) {
	if experts <= 0 {
		return nil, fmt.Errorf("expert bank %s.N.%s: per-expert tensors need the expert count", prefix, proj)
	}
	var weights, scales, biases, globals []*mlx.Array
	b := &ExpertBank{}
	for i := range experts {
		key := fmt.Sprintf("%s.%d.%s%s", prefix, i, proj, suffix)
		w := tensors[key]
		if w == nil {
			return nil, fmt.Errorf("expert bank %s.N.%s: expert %d of %d is missing", prefix, proj, i, experts)
		}
		if i == 0 {
			b.Key = key
		}
		b.keys = append(b.keys, key)
		weights = append(weights, w)

		scale, bias := tensors[key+"_scale"], tensors[key+"_qbias"]
		if i > 0 && (scale != nil) != (len(scales) > 0) {
			return nil, fmt.Errorf("expert bank %s.N.%s: %d of %d experts have scales", prefix, proj, len(scales), experts)
		}
		globalScale, globalKeys := ReadGlobalScale(tensors, key, key+".weight")
		b.keys = append(b.keys, globalKeys...)
		if scale != nil {
			b.keys = append(b.keys, key+"_scale")
			scales = append(scales, scale)
			if bias != nil {
				b.keys = append(b.keys, key+"_qbias")
				biases = append(biases, bias)
			}
			gs, bits, mode := ResolveLinearQuantParams(quant.GroupSize, quant.Bits, quant.Mode, quant.TensorQuant, key, w, scale)
			if i == 0 {
				b.GroupSize, b.Bits, b.Mode = gs, bits, mode
			} else if gs != b.GroupSize || bits != b.Bits || mode != b.Mode {
				return nil, fmt.Errorf("expert bank %s.N.%s: expert %d is %s/%d-bit/group %d, expert 0 is %s/%d-bit/group %d", prefix, proj, i, mode, bits, gs, b.Mode, b.Bits, b.GroupSize)
			}
		}
		if i > 0 && !slices.Equal(w.Dims(), weights[0].Dims()) {
			return nil, fmt.Errorf("expert bank %s.N.%s: expert %d has shape %v, expert 0 has %v", prefix, proj, i, w.Dims(), weights[0].Dims())
		}
		if globalScale != nil {
			if globalScale.Size() != 1 {
				return nil, fmt.Errorf("expert bank %s.N.%s: expert %d has %d global scales, want 1", prefix, proj, i, globalScale.Size())
			}
			globals = append(globals, globalScale)
		}
	}
	for _, part := range []struct {
		name string
		n    int
	}{{"scales", len(scales)}, {"quantization biases", len(biases)}, {"global scales", len(globals)}} {
		if part.n != 0 && part.n != experts {
			return nil, fmt.Errorf("expert bank %s.N.%s: %d of %d experts have %s", prefix, proj, part.n, experts, part.name)
		}
	}

	b.Weight = mlx.Stack(weights, 0).Clone()
	if len(scales) > 0 {
		b.Scales = mlx.Stack(scales, 0).Clone()
	}
	if len(biases) > 0 {
		b.Biases = mlx.Stack(biases, 0).Clone()
	}
	if len(globals) > 0 {
		b.GlobalScale = mlx.Reshape(mlx.Stack(globals, 0).Clone(), int32(experts))
	}
	return b, nil
}
