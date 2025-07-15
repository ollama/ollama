package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/src
// #include <stdlib.h>
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
// #include "ggml-quants.h"
import "C"

import (
	"iter"
	"slices"
	"unsafe"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// convertToF32 converts (dequantizes) the raw data to F32 so we can then quantize it
func ConvertToF32(data []byte, dtype uint32, nelements uint64) []float32 {
	f32s := make([]float32, nelements)
	elems := C.int64_t(nelements)
	switch dtype {
	case C.GGML_TYPE_F16:
		C.ggml_fp16_to_fp32_row((*C.uint16_t)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q4_0:
		C.dequantize_row_q4_0((*C.block_q4_0)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q4_1:
		C.dequantize_row_q4_1((*C.block_q4_1)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q5_0:
		C.dequantize_row_q5_0((*C.block_q5_0)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q5_1:
		C.dequantize_row_q5_1((*C.block_q5_1)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q8_0:
		C.dequantize_row_q8_0((*C.block_q8_0)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q2_K:
		C.dequantize_row_q2_K((*C.block_q2_K)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q3_K:
		C.dequantize_row_q3_K((*C.block_q3_K)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q4_K:
		C.dequantize_row_q4_K((*C.block_q4_K)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q5_K:
		C.dequantize_row_q5_K((*C.block_q5_K)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_Q6_K:
		C.dequantize_row_q6_K((*C.block_q6_K)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	case C.GGML_TYPE_BF16:
		C.ggml_bf16_to_fp32_row((*C.ggml_bf16_t)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	default:
		panic("unsupported quantization format")
	}
	return f32s
}

func Quantize(newType fsggml.TensorType, f32s []float32, shape []uint64) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		C.ggml_quantize_init(uint32(newType))
		defer C.ggml_quantize_free()

		dims := slices.Repeat([]C.int64_t{1}, 4)
		for i, s := range shape {
			dims[i] = C.int64_t(s)
		}

		bts := make([]byte, C.ggml_row_size(uint32(newType), dims[0])*C.size_t(dims[1]))
		for chunk := range dims[2] {
			offset := chunk * dims[0] * dims[1]

			n := C.ggml_quantize_chunk(
				uint32(newType),
				(*C.float)(&f32s[0]),
				unsafe.Pointer(&bts[0]),
				offset, dims[1], dims[0], nil,
			)

			if !yield(bts[:n]) {
				return
			}
		}
	}
}

func QuantizationVersion() uint32 {
	return uint32(C.GGML_QNT_VERSION)
}
