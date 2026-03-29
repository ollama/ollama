package create

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/d4l3k/go-bfloat16"
	"github.com/x448/float16"
)

// DTypeSize returns the byte size of a single element for the given dtype string.
func DTypeSize(dtype string) (int, error) {
	switch strings.ToUpper(dtype) {
	case "BF16", "F16":
		return 2, nil
	case "F32", "U32", "I32":
		return 4, nil
	case "F64":
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

// DecodeFloatTensor decodes raw bytes into []float32 according to the given dtype.
func DecodeFloatTensor(dtype string, raw []byte) ([]float32, error) {
	switch strings.ToUpper(dtype) {
	case "BF16":
		return bfloat16.DecodeFloat32(raw), nil
	case "F16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("invalid f16 byte length %d", len(raw))
		}
		values := make([]float32, len(raw)/2)
		for i := range values {
			values[i] = float16.Frombits(binary.LittleEndian.Uint16(raw[i*2:])).Float32()
		}
		return values, nil
	case "F32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("invalid f32 byte length %d", len(raw))
		}
		values := make([]float32, len(raw)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return values, nil
	case "F64":
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("invalid f64 byte length %d", len(raw))
		}
		values := make([]float32, len(raw)/8)
		for i := range values {
			values[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return values, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

// EncodeFloatTensor encodes []float32 into raw bytes according to the given dtype.
func EncodeFloatTensor(dtype string, values []float32) ([]byte, error) {
	switch strings.ToUpper(dtype) {
	case "BF16":
		return bfloat16.EncodeFloat32(values), nil
	case "F16":
		out := make([]byte, len(values)*2)
		for i, v := range values {
			binary.LittleEndian.PutUint16(out[i*2:], float16.Fromfloat32(v).Bits())
		}
		return out, nil
	case "F32":
		out := make([]byte, len(values)*4)
		for i, v := range values {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		return out, nil
	case "F64":
		out := make([]byte, len(values)*8)
		for i, v := range values {
			binary.LittleEndian.PutUint64(out[i*8:], math.Float64bits(float64(v)))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func sourceQuantType(mode string, bits int) string {
	switch strings.ToLower(mode) {
	case "affine":
		switch bits {
		case 4:
			return "int4"
		case 8:
			return "int8"
		}
	case "nvfp4":
		return "nvfp4"
	case "mxfp8":
		return "mxfp8"
	case "mxfp4":
		return "mxfp4"
	}
	return ""
}
