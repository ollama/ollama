package create

import (
	"bytes"
	"fmt"
	"io"

	"github.com/ollama/ollama/x/safetensors"
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

	case TransformRepackFP4, TransformRelabelU8:
		// Both relabel the header (dtype, and for the fp4 repack the last
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

	case TransformScalarF32:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform scalar_f32 expects 1 source, got %d", len(sources))
		}
		return validateScalarFloat32TensorData(sources[0], ts.Name)

	case TransformReciprocalF32:
		if len(sources) != 1 {
			return nil, fmt.Errorf("transform reciprocal_f32 expects 1 source, got %d", len(sources))
		}
		return invertScalarFloat32TensorData(sources[0], ts.Name)

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
	var buf bytes.Buffer
	for i, s := range sources {
		if s.Dtype != sources[0].Dtype {
			return nil, fmt.Errorf("stack_experts source %d dtype %s != %s", i, s.Dtype, sources[0].Dtype)
		}
		if _, err := io.Copy(&buf, s.Reader()); err != nil {
			return nil, fmt.Errorf("stack_experts read source %d (%s): %w", i, s.Name, err)
		}
	}
	return safetensors.NewTensorDataFromBytes(name, dtype, append([]int32(nil), shape...), buf.Bytes()), nil
}
