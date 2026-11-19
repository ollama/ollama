package model

import (
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
)

// ExpertProjection keeps a packed expert bank and its quantization companions
// together. Unquantized weights are transposed once at load for GatherMM.
type ExpertProjection struct {
	Weight      *mlx.Array
	Scales      *mlx.Array
	Biases      *mlx.Array
	GlobalScale *mlx.Array
	GroupSize   int
	Bits        int
	Mode        string
}

func (p *ExpertProjection) Forward(x, indices *mlx.Array, sorted bool) *mlx.Array {
	if p.Scales != nil {
		return mlx.GatherQMM(x, p.Weight, p.Scales, p.Biases,
			nil, indices, true, p.GroupSize, p.Bits, p.Mode, p.GlobalScale, sorted)
	}
	return mlx.GatherMM(x, p.Weight, nil, indices, sorted)
}

func supportsGatherQMM(mode string, bits int) bool {
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

// LoadExpertProjection resolves a packed, out-major expert projection.
func LoadExpertProjection(tensors map[string]*mlx.Array, base string, group, bits int, mode string, quant map[string]*TensorQuantInfo) (*ExpertProjection, error) {
	p := &ExpertProjection{}
	var key string
	for _, candidate := range []string{base, base + ".weight"} {
		if p.Weight = tensors[candidate]; p.Weight != nil {
			key = candidate
			break
		}
	}
	if p.Weight == nil {
		return nil, fmt.Errorf("missing packed expert tensor %q", base)
	}
	if p.Weight.NumDims() != 3 || p.Weight.Size() == 0 {
		return nil, fmt.Errorf("packed expert tensor %q needs nonempty [experts, output, input] weights, got %v", key, p.Weight.Dims())
	}
	p.Scales = tensors[key+"_scale"]
	if p.Scales == nil {
		if p.Weight.DType() == mlx.DTypeUint32 {
			return nil, fmt.Errorf("packed expert tensor %q is missing its quantization scales", key)
		}
		p.Weight = mlx.Transpose(p.Weight, 0, 2, 1)
	} else {
		p.Biases = tensors[key+"_qbias"]
		p.GroupSize, p.Bits, p.Mode = ResolveLinearQuantParams(group, bits, mode, quant, key, p.Weight, p.Scales)
		if !supportsGatherQMM(p.Mode, p.Bits) {
			return nil, fmt.Errorf("packed expert tensor %q uses unsupported quantization mode=%q bits=%d", key, p.Mode, p.Bits)
		}
		weightShape, scaleShape := p.Weight.Dims(), p.Scales.Dims()
		if p.Weight.DType() != mlx.DTypeUint32 || len(scaleShape) != 3 ||
			!slices.Equal(weightShape[:2], scaleShape[:2]) || scaleShape[2] <= 0 || p.GroupSize <= 0 ||
			int64(weightShape[2])*32 != int64(scaleShape[2])*int64(p.GroupSize)*int64(p.Bits) {
			return nil, fmt.Errorf("packed expert tensor %q has incompatible weight %v and scale %v shapes for group %d and %d bits",
				key, weightShape, scaleShape, p.GroupSize, p.Bits)
		}
		if p.Biases != nil && !slices.Equal(p.Biases.Dims(), scaleShape) {
			return nil, fmt.Errorf("packed expert tensor %q has quantization bias shape %v, want %v", key, p.Biases.Dims(), scaleShape)
		}
	}
	var err error
	p.GlobalScale, err = loadExpertGlobalScale(tensors, base, key, p.Weight.Dim(0), p.Mode)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func loadExpertGlobalScale(tensors map[string]*mlx.Array, base, key string, experts int, mode string) (*mlx.Array, error) {
	scale, _ := ReadGlobalScale(tensors, key, base, base+".weight")
	if scale == nil {
		return nil, nil
	}
	if mode != "nvfp4" {
		return nil, fmt.Errorf("expert tensor %q has a global scale but uses mode=%q", base, mode)
	}
	if scale.Size() != 1 && scale.Size() != experts {
		return nil, fmt.Errorf("expert tensor %q has %d global scales for %d experts", base, scale.Size(), experts)
	}
	return PrepareGatherQMMGlobalScale(scale, experts), nil
}
