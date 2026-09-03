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

	case TransformConcatAxis1:
		if len(sources) != 2 {
			return nil, fmt.Errorf("transform %s expects 2 sources, got %d", ts.Transform, len(sources))
		}
		return concatAxis1(ts.Name, ts.OutShape, sources[0], sources[1])

	default:
		return nil, fmt.Errorf("transform %q requires the MLX writer path", ts.Transform)
	}
}

// concatAxis1 concatenates two 3-D source tensors [A, B, C] along axis 1,
// producing [A, 2*B, C]. In row-major layout each outer slab has B*C elements;
// the output slab is the lo-half slab followed immediately by the hi-half slab.
// outShape must be [A, 2*B, C].
func concatAxis1(name string, outShape []int32, lo, hi *safetensors.TensorData) (*safetensors.TensorData, error) {
	if len(outShape) != 3 {
		return nil, fmt.Errorf("concat_axis1 requires a 3-D output shape, got %v", outShape)
	}
	if lo.Dtype != hi.Dtype {
		return nil, fmt.Errorf("concat_axis1: lo dtype %s != hi dtype %s", lo.Dtype, hi.Dtype)
	}
	elemSize, err := DTypeSize(lo.Dtype)
	if err != nil {
		return nil, fmt.Errorf("concat_axis1: %w", err)
	}
	A, twoB, C := int(outShape[0]), int(outShape[1]), int(outShape[2])
	B := twoB / 2
	halfBytes := B * C * elemSize
	slabBytes := twoB * C * elemSize

	loBytes, err := io.ReadAll(lo.Reader())
	if err != nil {
		return nil, fmt.Errorf("concat_axis1 read lo (%s): %w", lo.Name, err)
	}
	hiBytes, err := io.ReadAll(hi.Reader())
	if err != nil {
		return nil, fmt.Errorf("concat_axis1 read hi (%s): %w", hi.Name, err)
	}
	if len(loBytes) != A*halfBytes {
		return nil, fmt.Errorf("concat_axis1: lo %s has %d bytes, expected %d for shape [%d %d %d] dtype %s",
			lo.Name, len(loBytes), A*halfBytes, A, B, C, lo.Dtype)
	}
	if len(hiBytes) != A*halfBytes {
		return nil, fmt.Errorf("concat_axis1: hi %s has %d bytes, expected %d for shape [%d %d %d] dtype %s",
			hi.Name, len(hiBytes), A*halfBytes, A, B, C, hi.Dtype)
	}

	out := make([]byte, A*slabBytes)
	for a := range A {
		dstOff := a * slabBytes
		srcOff := a * halfBytes
		copy(out[dstOff:], loBytes[srcOff:srcOff+halfBytes])
		copy(out[dstOff+halfBytes:], hiBytes[srcOff:srcOff+halfBytes])
	}
	return safetensors.NewTensorDataFromBytes(name, lo.Dtype, append([]int32(nil), outShape...), out), nil
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
