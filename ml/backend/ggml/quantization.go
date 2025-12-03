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
	"unsafe"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// ConvertToF32 converts (dequantizes) the raw data to F32 so we can then quantize it
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
	case C.GGML_TYPE_MXFP4:
		C.dequantize_row_mxfp4((*C.block_mxfp4)(unsafe.Pointer(&data[0])), (*C.float)(&f32s[0]), elems)
	default:
		panic("unsupported quantization format")
	}
	return f32s
}

func Quantize(newType fsggml.TensorType, f32s []float32, shape []uint64) []byte {
	buf := make([]byte, len(f32s)*4) // upper bound on size
	nPerRow := C.int64_t(shape[0])
	nrows := C.int64_t(1)
	if len(shape) > 1 {
		nrows = C.int64_t(shape[1])
	}
	shape2 := C.int64_t(1)
	if len(shape) > 2 {
		shape2 = C.int64_t(shape[2])
	}
	nelements_matrix := nPerRow * nrows
	newSize := C.size_t(0)
	for i03 := C.int64_t(0); i03 < shape2; i03++ {
		f32s_03 := i03 * nelements_matrix
		buf_03 := C.int64_t(C.ggml_row_size(uint32(newType), nPerRow)) * i03 * nrows
		newSize += C.ggml_quantize_chunk(
			uint32(newType),
			(*C.float)(&f32s[f32s_03]),
			unsafe.Pointer((uintptr)(unsafe.Pointer(&buf[0]))+uintptr(buf_03)),
			0,
			nrows,
			nPerRow,
			nil)
	}
	return buf[:newSize]
}

func QuantizationVersion() uint32 {
	return uint32(C.GGML_QNT_VERSION)
}
