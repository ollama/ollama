package create

import (
	"fmt"
	"io"

	"github.com/ollama/ollama/fs/safetensors"
)

// applyByteTransform produces a TensorSpec's output tensor from its resolved
// source tensors using only byte-level (non-MLX) operations. The MLX transform
// (decode_fp8) and quantization are handled separately by the MLX writer path.
func applyByteTransform(ts TensorSpec, sources []*safetensors.TensorData) (*safetensors.TensorData, error) {
	switch ts.Transform {
	case TransformNone:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform none expects 1 source, got %d", len(sources))
		}
		return sources[0].WithName(ts.Name), nil

	case TransformRepackFP4, TransformRepackFP8, TransformRelabelU8:
		// These relabel the header (dtype, and for packed weights the last
		// dimension); the bytes are unchanged, so the reader is reused.
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform %s expects 1 source, got %d", ts.Transform, len(sources))
		}
		td := sources[0].WithName(ts.Name)
		if ts.OutDtype != "" {
			td.Dtype = ts.OutDtype
		}
		if ts.OutShape != nil {
			td.Shape = append([]int32(nil), ts.OutShape...)
		}
		return td, nil

	case TransformExpandE8M0:
		if len(sources) != 1 || sources[0].Dtype != "F8_E8M0" || len(sources[0].Shape) != 2 ||
			len(ts.OutShape) != 2 || ts.OutDtype != "U8" ||
			ts.OutShape[0] != sources[0].Shape[0]*32 || ts.OutShape[1] != sources[0].Shape[1] ||
			sources[0].Size != int64(sources[0].Shape[0])*int64(sources[0].Shape[1]) {
			return nil, fmt.Errorf("expand_e8m0 requires a 32-row E8M0 block scale and matching U8 output")
		}
		if sources[0].Size > 64<<20 {
			return nil, fmt.Errorf("expand_e8m0 scale %s exceeds 64 MiB", sources[0].Name)
		}
		in := make([]byte, sources[0].Size)
		if _, err := io.ReadFull(sources[0].Reader(), in); err != nil {
			return nil, fmt.Errorf("expand_e8m0 read %s: %w", sources[0].Name, err)
		}
		cols := int(sources[0].Shape[1])
		out := make([]byte, len(in)*32)
		for row := range int(sources[0].Shape[0]) {
			for repeat := range 32 {
				copy(out[(row*32+repeat)*cols:], in[row*cols:(row+1)*cols])
			}
		}
		return safetensors.NewTensorDataFromBytes(ts.Name, "U8", ts.OutShape, out), nil

	case TransformScalarF32:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform scalar_f32 expects 1 source, got %d", len(sources))
		}
		return validateScalarFloat32TensorData(sources[0], ts.Name)

	case TransformF32:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform f32 expects 1 source, got %d", len(sources))
		}
		return validateFloat32TensorData(sources[0], ts.Name)

	case TransformReciprocalF32:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform reciprocal_f32 expects 1 source, got %d", len(sources))
		}
		return invertFloat32TensorData(sources[0], ts.Name)

	case TransformStackExperts:
		return stackExpertTensors(ts.Name, ts.OutDtype, ts.OutShape, sources)

	default:
		return nil, fmt.Errorf("transform %q requires the MLX writer path", ts.Transform)
	}
}

// stackExpertTensors concatenates per-expert tensors (in the given order) into
// one [experts, ...] tensor. Row-major layout means the stacked bytes are
// exactly the per-expert byte blocks back to back.
func stackExpertTensors(name, dtype string, shape []int32, sources []*safetensors.TensorData) (*safetensors.TensorData, error) {
	if len(sources) == 0 {
		return nil, fmt.Errorf("stack_experts expects at least one source")
	}
	for i, s := range sources {
		if s.Dtype != sources[0].Dtype {
			return nil, fmt.Errorf("stack_experts source %d dtype %s != %s", i, s.Dtype, sources[0].Dtype)
		}
	}
	return safetensors.ConcatTensorData(name, dtype, shape, sources)
}
