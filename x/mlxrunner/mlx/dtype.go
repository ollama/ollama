//go:build mlx

package mlx

// #include "generated.h"
import "C"

type DType int

func (t DType) String() string {
	switch t {
	case DTypeBool:
		return "BOOL"
	case DTypeUint8:
		return "U8"
	case DTypeUint16:
		return "U16"
	case DTypeUint32:
		return "U32"
	case DTypeUint64:
		return "U64"
	case DTypeInt8:
		return "I8"
	case DTypeInt16:
		return "I16"
	case DTypeInt32:
		return "I32"
	case DTypeInt64:
		return "I64"
	case DTypeFloat16:
		return "F16"
	case DTypeFloat32:
		return "F32"
	case DTypeFloat64:
		return "F64"
	case DTypeBFloat16:
		return "BF16"
	case DTypeComplex64:
		return "C64"
	default:
		return "Unknown"
	}
}

func (t *DType) UnmarshalJSON(b []byte) error {
	switch string(b) {
	case `"BOOL"`:
		*t = DTypeBool
	case `"U8"`:
		*t = DTypeUint8
	case `"U16"`:
		*t = DTypeUint16
	case `"U32"`:
		*t = DTypeUint32
	case `"U64"`:
		*t = DTypeUint64
	case `"I8"`:
		*t = DTypeInt8
	case `"I16"`:
		*t = DTypeInt16
	case `"I32"`:
		*t = DTypeInt32
	case `"I64"`:
		*t = DTypeInt64
	case `"F16"`:
		*t = DTypeFloat16
	case `"F64"`:
		*t = DTypeFloat64
	case `"F32"`:
		*t = DTypeFloat32
	case `"BF16"`:
		*t = DTypeBFloat16
	case `"C64"`:
		*t = DTypeComplex64
	default:
		return nil
	}
	return nil
}

const (
	DTypeBool      DType = C.MLX_BOOL
	DTypeUint8     DType = C.MLX_UINT8
	DTypeUint16    DType = C.MLX_UINT16
	DTypeUint32    DType = C.MLX_UINT32
	DTypeUint64    DType = C.MLX_UINT64
	DTypeInt8      DType = C.MLX_INT8
	DTypeInt16     DType = C.MLX_INT16
	DTypeInt32     DType = C.MLX_INT32
	DTypeInt64     DType = C.MLX_INT64
	DTypeFloat16   DType = C.MLX_FLOAT16
	DTypeFloat32   DType = C.MLX_FLOAT32
	DTypeFloat64   DType = C.MLX_FLOAT64
	DTypeBFloat16  DType = C.MLX_BFLOAT16
	DTypeComplex64 DType = C.MLX_COMPLEX64
)
