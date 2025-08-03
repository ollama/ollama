package gguf

import (
	"log/slog"
	"strings"
)

type TensorInfo struct {
	Name   string
	Offset uint64
	Shape  []uint64
	Type   TensorType
}

func (ti TensorInfo) Valid() bool {
	return ti.Name != "" && ti.NumBytes() > 0
}

func (ti TensorInfo) NumValues() int64 {
	var numItems int64 = 1
	for _, dim := range ti.Shape {
		numItems *= int64(dim)
	}
	return numItems
}

// NumBytes returns the number of bytes in the tensor.
func (ti TensorInfo) NumBytes() int64 {
	return int64(float64(ti.NumValues()) * ti.Type.NumBytes())
}

func (ti TensorInfo) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("name", ti.Name),
		slog.Int64("offset", int64(ti.Offset)),
		slog.Any("shape", ti.Shape),
		slog.Int64("num_values", ti.NumValues()),
		slog.Int64("num_bytes", ti.NumBytes()),
		slog.Any("type", ti.Type),
	)
}

type TensorType uint32

const (
	TensorTypeF32 TensorType = iota
	TensorTypeF16
	TensorTypeQ4_0
	TensorTypeQ4_1

	// unexported // unused in gguf
	tensorTypeQ4_2
	tensorTypeQ4_3

	TensorTypeQ5_0
	TensorTypeQ5_1
	TensorTypeQ8_0
	TensorTypeQ8_1
	TensorTypeQ2_K
	TensorTypeQ3_K
	TensorTypeQ4_K
	TensorTypeQ5_K
	TensorTypeQ6_K
	TensorTypeQ8_K

	// unexported // unquantizable by ollama
	tensorTypeIQ2_XXS
	tensorTypeIQ2_XS
	tensorTypeIQ3_XXS
	tensorTypeIQ1_S
	tensorTypeIQ4_NL
	tensorTypeIQ3_S
	tensorTypeIQ2_S
	tensorTypeIQ4_XS

	TensorTypeI8
	TensorTypeI16
	TensorTypeI32
	TensorTypeI64
	TensorTypeF64

	// unexported // unquantizable by ollama
	tensorTypeIQ1_M

	TensorTypeBF16

	// unexported // unused in gguf
	tensorTypeQ4_0_4_4
	tensorTypeQ4_0_4_8
	tensorTypeQ4_0_8_8

	// unexported // unquantizable by ollama
	tensorTypeTQ1_0
	tensorTypeTQ2_0

	// unexported // unused in gguf
	tensorTypeIQ4_NL_4_4
	tensorTypeIQ4_NL_4_8
	tensorTypeIQ4_NL_8_8
)

func (tt TensorType) NumBytes() float64 {
	return float64(tt.typeSize()) / float64(tt.blockSize())
}

func (tt TensorType) typeSize() int64 {
	switch tt {
	case TensorTypeF32:
		return 4
	case TensorTypeF16:
		return 2
	case TensorTypeQ4_0:
		return 2 + tt.blockSize()/2
	case TensorTypeQ4_1:
		return 2 + 2 + tt.blockSize()/2
	case TensorTypeQ5_0:
		return 2 + 4 + tt.blockSize()/2
	case TensorTypeQ5_1:
		return 2 + 2 + 4 + tt.blockSize()/2
	case TensorTypeQ8_0:
		return 2 + tt.blockSize()
	case TensorTypeQ8_1:
		return 2 + 2 + tt.blockSize()
	case TensorTypeQ2_K:
		return tt.blockSize()/16 + tt.blockSize()/4 + 2 + 2
	case TensorTypeQ3_K:
		return tt.blockSize()/8 + tt.blockSize()/4 + 12 + 2
	case TensorTypeQ4_K:
		return 2 + 2 + 12 + tt.blockSize()/2
	case TensorTypeQ5_K:
		return 2 + 2 + 12 + tt.blockSize()/8 + tt.blockSize()/2
	case TensorTypeQ6_K:
		return tt.blockSize()/2 + tt.blockSize()/4 + tt.blockSize()/16 + 2
	case TensorTypeQ8_K:
		return 4 + tt.blockSize() + 2*tt.blockSize()/16
	case tensorTypeIQ2_XXS:
		return 2 + 2*tt.blockSize()/8
	case tensorTypeIQ2_XS:
		return 2 + 2*tt.blockSize()/8 + tt.blockSize()/32
	case tensorTypeIQ3_XXS:
		return 2 + tt.blockSize()/4 + tt.blockSize()/8
	case tensorTypeIQ1_S:
		return 2 + tt.blockSize()/8 + tt.blockSize()/16
	case tensorTypeIQ4_NL:
		return 2 + tt.blockSize()/2
	case tensorTypeIQ3_S:
		return 2 + tt.blockSize()/4 + tt.blockSize()/8 + tt.blockSize()/32 + 4
	case tensorTypeIQ2_S:
		return 2 + tt.blockSize()/4 + tt.blockSize()/16
	case tensorTypeIQ4_XS:
		return 2 + 2 + tt.blockSize()/2 + tt.blockSize()/64
	case TensorTypeI8:
		return 1
	case TensorTypeI16:
		return 2
	case TensorTypeI32:
		return 4
	case TensorTypeI64:
		return 8
	case TensorTypeF64:
		return 8
	case tensorTypeIQ1_M:
		return tt.blockSize()/8 + tt.blockSize()/16 + tt.blockSize()/32
	case TensorTypeBF16:
		return 2
	default:
		return 0
	}
}

func (tt TensorType) blockSize() int64 {
	switch tt {
	case TensorTypeF32,
		TensorTypeF16,
		TensorTypeI8,
		TensorTypeI16,
		TensorTypeI32,
		TensorTypeI64,
		TensorTypeF64,
		TensorTypeBF16:
		return 1
	case TensorTypeQ4_0,
		TensorTypeQ4_1,
		TensorTypeQ5_0,
		TensorTypeQ5_1,
		TensorTypeQ8_0,
		TensorTypeQ8_1,
		tensorTypeIQ4_NL:
		return 32
	default:
		return 256
	}
}

func (tt TensorType) String() string {
	switch tt {
	case TensorTypeF32:
		return "f32"
	case TensorTypeF16:
		return "f16"
	case TensorTypeQ4_0:
		return "q4_0"
	case TensorTypeQ4_1:
		return "q4_1"
	case tensorTypeQ4_2:
		return "q4_2"
	case tensorTypeQ4_3:
		return "q4_3"
	case TensorTypeQ5_0:
		return "q5_0"
	case TensorTypeQ5_1:
		return "q5_1"
	case TensorTypeQ8_0:
		return "q8_0"
	case TensorTypeQ8_1:
		return "q8_1"
	case TensorTypeQ2_K:
		return "q2_k"
	case TensorTypeQ3_K:
		return "q3_k"
	case TensorTypeQ4_K:
		return "q4_k"
	case TensorTypeQ5_K:
		return "q5_k"
	case TensorTypeQ6_K:
		return "q6_k"
	case TensorTypeQ8_K:
		return "q8_k"
	case tensorTypeIQ2_XXS:
		return "iq2_xxs"
	case tensorTypeIQ2_XS:
		return "iq2_xs"
	case tensorTypeIQ3_XXS:
		return "iq3_xxs"
	case tensorTypeIQ1_S:
		return "iq1_s"
	case tensorTypeIQ4_NL:
		return "iq4_nl"
	case tensorTypeIQ3_S:
		return "iq3_s"
	case tensorTypeIQ2_S:
		return "iq2_s"
	case tensorTypeIQ4_XS:
		return "iq4_xs"
	case TensorTypeI8:
		return "i8"
	case TensorTypeI16:
		return "i16"
	case TensorTypeI32:
		return "i32"
	case TensorTypeI64:
		return "i64"
	case TensorTypeF64:
		return "f64"
	case tensorTypeIQ1_M:
		return "iq1_m"
	case TensorTypeBF16:
		return "bf16"
	case tensorTypeQ4_0_4_4:
		return "q4_0_4_4"
	case tensorTypeQ4_0_4_8:
		return "q4_0_4_8"
	case tensorTypeQ4_0_8_8:
		return "q4_0_8_8"
	case tensorTypeTQ1_0:
		return "tq1_0"
	case tensorTypeTQ2_0:
		return "tq2_0"
	case tensorTypeIQ4_NL_4_4:
		return "iq4_nl_4_4"
	case tensorTypeIQ4_NL_4_8:
		return "iq4_nl_4_8"
	case tensorTypeIQ4_NL_8_8:
		return "iq4_nl_8_8"
	default:
		return "unknown"
	}
}

func (tt TensorType) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Uint64("value", uint64(tt)),
		slog.String("name", strings.ToUpper(tt.String())),
		slog.Int64("size", tt.typeSize()),
		slog.Int64("block_size", tt.blockSize()),
		slog.Float64("num_bytes", tt.NumBytes()),
	)
}
