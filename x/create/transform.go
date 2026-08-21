package create

import (
	"bytes"
	"fmt"
	"io"
	"slices"
	"strings"

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

	case TransformRepackFP4, TransformRelabelU8, TransformRelabelU32:
		// These transforms relabel the header (dtype, and for the fp4 repack the last
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

	case TransformInt4SymmetricQBias:
		return int4SymmetricQBiasTensor(ts.Name, ts.OutDtype, ts.OutShape, sources)

	case TransformBlockFP8GroupScales:
		return blockFP8GroupScalesTensor(ts.Name, ts.OutDtype, ts.OutShape, sources)

	default:
		return nil, fmt.Errorf("transform %q requires the MLX writer path", ts.Transform)
	}
}

// int4SymmetricQBiasTensor converts compressed-tensors' symmetric INT4 scales
// into MLX affine biases. compressed-tensors stores signed q in the packed word
// as q+8; MLX reconstructs scale*packed_q+bias, hence bias=-8*scale. Scaling by
// a power of two is exact for BF16/F16/F32 values that remain in range.
func int4SymmetricQBiasTensor(name, dtype string, shape []int32, sources []*safetensors.TensorData) (*safetensors.TensorData, error) {
	if len(sources) == 0 {
		return nil, fmt.Errorf("int4_symmetric_qbias expects at least one scale source")
	}
	base := sources[0]
	if dtype == "" {
		dtype = base.Dtype
	}
	if shape == nil {
		if len(sources) == 1 {
			shape = append([]int32(nil), base.Shape...)
		} else {
			shape = append([]int32{int32(len(sources))}, base.Shape...)
		}
	}

	var buf bytes.Buffer
	for i, source := range sources {
		if !strings.EqualFold(source.Dtype, base.Dtype) || !slices.Equal(source.Shape, base.Shape) {
			return nil, fmt.Errorf("int4_symmetric_qbias source %d layout %s %v != %s %v", i, source.Dtype, source.Shape, base.Dtype, base.Shape)
		}
		raw, err := io.ReadAll(source.Reader())
		if err != nil {
			return nil, fmt.Errorf("int4_symmetric_qbias read source %d (%s): %w", i, source.Name, err)
		}
		values, err := DecodeFloatTensor(source.Dtype, raw)
		if err != nil {
			return nil, fmt.Errorf("int4_symmetric_qbias decode source %d (%s): %w", i, source.Name, err)
		}
		for j := range values {
			values[j] *= -8
		}
		encoded, err := EncodeFloatTensor(dtype, values)
		if err != nil {
			return nil, fmt.Errorf("int4_symmetric_qbias encode source %d (%s): %w", i, source.Name, err)
		}
		buf.Write(encoded)
	}
	return safetensors.NewTensorDataFromBytes(name, dtype, append([]int32(nil), shape...), buf.Bytes()), nil
}

// blockFP8GroupScalesTensor expands UE8M0 128x128 block-scale exponent bytes
// into mxfp8 per-group scale bytes: out[..., r, g] = src[..., r/128, g*32/128].
// Every 32-value group along the last axis lies inside exactly one 128-column
// block, so the expansion is a pure replication and the conversion is exact.
// Sources are per-expert block scales in expert order (a single source for a
// plain 2D weight); shape is the output [(experts,) rows, cols/32].
func blockFP8GroupScalesTensor(name, dtype string, shape []int32, sources []*safetensors.TensorData) (*safetensors.TensorData, error) {
	const blockRows, blockCols, groupSize = 128, 128, 32
	if len(sources) == 0 {
		return nil, fmt.Errorf("blockfp8_group_scales expects at least one scale source")
	}
	if len(shape) < 2 {
		return nil, fmt.Errorf("blockfp8_group_scales output shape %v must have rank >= 2", shape)
	}
	if dtype == "" {
		dtype = "U8"
	}
	rows := int(shape[len(shape)-2])
	groups := int(shape[len(shape)-1])
	lead := 1
	for _, d := range shape[:len(shape)-2] {
		lead *= int(d)
	}
	cols := groups * groupSize
	sr := (rows + blockRows - 1) / blockRows
	sc := (cols + blockCols - 1) / blockCols
	if lead%len(sources) != 0 {
		return nil, fmt.Errorf("blockfp8_group_scales output shape %v does not evenly cover %d scale sources", shape, len(sources))
	}
	perSource := lead / len(sources)

	out := make([]byte, 0, lead*rows*groups)
	for i, source := range sources {
		if !isE8M0Dtype(source.Dtype) {
			return nil, fmt.Errorf("blockfp8_group_scales source %d (%s) has dtype %s, want an UE8M0 exponent tensor", i, source.Name, source.Dtype)
		}
		wantShape := []int32{int32(sr), int32(sc)}
		if perSource > 1 || len(source.Shape) == 3 {
			wantShape = append([]int32{int32(perSource)}, wantShape...)
		}
		if !slices.Equal(source.Shape, wantShape) {
			return nil, fmt.Errorf("blockfp8_group_scales source %d (%s) has shape %v, want %v for output %v", i, source.Name, source.Shape, wantShape, shape)
		}
		raw, err := io.ReadAll(source.Reader())
		if err != nil {
			return nil, fmt.Errorf("blockfp8_group_scales read source %d (%s): %w", i, source.Name, err)
		}
		if len(raw) != perSource*sr*sc {
			return nil, fmt.Errorf("blockfp8_group_scales source %d (%s) has %d bytes, want %d", i, source.Name, len(raw), perSource*sr*sc)
		}
		for s := 0; s < perSource; s++ {
			blockScales := raw[s*sr*sc : (s+1)*sr*sc]
			for r := 0; r < rows; r++ {
				blockRow := blockScales[(r/blockRows)*sc : (r/blockRows)*sc+sc]
				for g := 0; g < groups; g++ {
					out = append(out, blockRow[g*groupSize/blockCols])
				}
			}
		}
	}
	return safetensors.NewTensorDataFromBytes(name, dtype, append([]int32(nil), shape...), out), nil
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
