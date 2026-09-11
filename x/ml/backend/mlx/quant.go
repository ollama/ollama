//go:build mlx

package mlx

/*
#include <stdio.h>
#include <string.h>

#include "mlx/c/array.h"
#include "mlx/c/ops.h"

// Derived from https://github.com/ml-explore/mlx/blob/main/mlx/io/gguf_quants.cpp

void unpack_32_4(uint8_t* data, int8_t* dst) {
	memset(dst, 0, 16);
	for (int j = 0; j < 16; ++j) {
		uint8_t x = (data[j + 2] & 0x0F); // j+2 to skip scale bytes.
		if (j % 2 != 0) {
		x <<= 4;
		}
		dst[j / 2] += x;
	}
	// Last 16 weights are in the higher bits
	for (int j = 0; j < 16; ++j) {
		uint8_t x = (data[j + 2] >> 4);
		if (j % 2 != 0) {
		x <<= 4;
		}
		dst[8 + j / 2] += x;
	}
}

// Extracts (weight, scales, biases) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(
		uint8_t* data,
		mlx_array* weights_arr,
		mlx_array* scales_arr,
		mlx_array* biases_arr) {
	const uint64_t bytes_per_block = 18; // 2 bytes scale, 32x0.5 byte weights
	uint8_t* weights = mlx_array_data_uint8(*weights_arr);
	float16_t* scales = mlx_array_data_float16(*scales_arr);
	float16_t* biases = mlx_array_data_float16(*biases_arr);
	for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
		scales[i] = *((float16_t*)data);
		biases[i] = -8 * scales[i];
		unpack_32_4(data, weights);
		weights += 16;
		data += bytes_per_block;
	}
}

// Extracts (weight, scales, biases) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit bias|32 x 4bit weights|.
void extract_q4_1_data(
		uint8_t* data,
		mlx_array* weights_arr,
		mlx_array* scales_arr,
		mlx_array* biases_arr) {
	const uint64_t bytes_per_block = 20; // 2 bytes scale, 2 bytes bias, 32x0.5 byte weights
	uint8_t* weights = mlx_array_data_uint8(*weights_arr);
	float16_t* scales = mlx_array_data_float16(*scales_arr);
	float16_t* biases = mlx_array_data_float16(*biases_arr);
	for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
		scales[i] = *((float16_t*)data);
		biases[i] = *((float16_t*)(data) + 1);
		unpack_32_4(data, weights);
		weights += 16;
		data += bytes_per_block;
	}
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(
		uint8_t* data,
		mlx_array* weights_arr,
		mlx_array* scales_arr,
		mlx_array* biases_arr) {
	const uint64_t weights_per_block = 32;
	const uint64_t bytes_per_block = 34; // 2 bytes scale, 32x1 byte weights
	uint8_t* weights = mlx_array_data_uint8(*weights_arr);
	float16_t* scales = mlx_array_data_float16(*scales_arr);
	float16_t* biases = mlx_array_data_float16(*biases_arr);
	for (int64_t i = 0; i < mlx_array_size(*scales_arr); i++) {
		uint8_t* block_data = data + i * bytes_per_block;
		scales[i] = *((float16_t*)block_data);
		biases[i] = -128 * scales[i];
		for (int64_t j = 0; j < weights_per_block; ++j) {
			uint8_t x = block_data[j + 2]; // j+2 to skip the scale bytes.
			// Original data is in int8_t, so we add a bias of -128 and invert the
			// first bit.
			x ^= 1 << 7;
			weights[i * weights_per_block + j] = x;
		}
	}
}

// Drived from ggml-quants.c

#define QK_K 256

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;             // super-block scale
} block_q6_K;

void dequant_row_q6_K(const void * restrict vx, void * restrict vy, int k) {
    const int64_t nb = k / QK_K;
	block_q6_K *x = (block_q6_K *)vx;
	float16_t* y = (float16_t *)vy;

    for (int i = 0; i < nb; i++) {
		float16_t d = 0.0;
		memcpy(&d, &x[i].d, sizeof(d));

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

#define K_SCALE_SIZE 12
#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            uint16_t d;    // super-block scale for quantized scales
            uint16_t dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        uint16_t dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequant_row_q4_K(const void * restrict vx, void * restrict vy, int k) {
	block_q4_K *x = (block_q4_K *)vx;
	float16_t* y = (float16_t *)vy;
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
		float16_t d = 0.0;
		memcpy(&d, &x[i].d, sizeof(d));
		float16_t min = 0.0;
		memcpy(&min, &x[i].dmin, sizeof(d));

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float16_t d1 = d * sc; const float16_t m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float16_t d2 = d * sc; const float16_t m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}



*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/x448/float16"
)

func gguf_load_quantized(data unsafe.Pointer, name string, final_shape []C.int, dtype uint32, stream C.mlx_stream) (r C.mlx_array, err error) {
	shape := append([]C.int{}, final_shape...)
	var weights_per_byte C.int
	if dtype == 2 || dtype == 3 {
		weights_per_byte = 2
	} else if dtype == 8 {
		weights_per_byte = 1
	} else {
		return r, fmt.Errorf("unsupported tensor type %d", dtype)
	}

	weights_per_block := C.int(32)
	if shape[len(shape)-1]%weights_per_block != 0 {
		return r, fmt.Errorf("[load_gguf] tensor has incompatible last dim shape: %d", shape[len(shape)-1])
	}

	weights_shape := append([]C.int{}, shape...)
	weights_shape[len(weights_shape)-1] /= (weights_per_byte * 4)
	w_nbytes := C.int(unsafe.Sizeof(uint32(0)))
	for i := range weights_shape {
		w_nbytes *= weights_shape[i]
	}
	w_data := make([]byte, w_nbytes)
	cbytes := C.CBytes(w_data)
	defer C.free(cbytes)
	weights := C.mlx_array_new_data(
		cbytes,
		&weights_shape[0],
		C.int(len(weights_shape)),
		C.MLX_UINT32,
	)

	// For scales and bias
	shape[len(shape)-1] = shape[len(shape)-1] / weights_per_block
	sb_nbytes := C.int(unsafe.Sizeof(float16.Float16(0)))
	for i := range shape {
		sb_nbytes *= shape[i]
	}

	s_data := make([]byte, sb_nbytes)
	cbytes = C.CBytes(s_data)
	defer C.free(cbytes)
	scales := C.mlx_array_new_data(
		cbytes,
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)
	b_data := make([]byte, sb_nbytes)
	cbytes = C.CBytes(b_data)
	defer C.free(cbytes)
	biases := C.mlx_array_new_data(
		cbytes,
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)
	var bits C.int
	switch dtype {
	case 2:
		C.extract_q4_0_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 4
	case 3:
		C.extract_q4_1_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 4
	case 8:
		C.extract_q8_0_data((*C.uint8_t)(data), &weights, &scales, &biases)
		bits = 8
	}
	groupSize := C.mlx_optional_int{value: 32, has_value: true}
	bitsOpt := C.mlx_optional_int{value: bits, has_value: true}
	var dtypeOpt C.mlx_optional_dtype // has_value defaults to false
	C.mlx_dequantize(
		&r,
		weights,
		scales,
		biases,
		groupSize,
		bitsOpt,
		nil, // TODO mode
		dtypeOpt,
		stream,
	)
	C.mlx_array_free(weights)
	C.mlx_array_free(scales)
	C.mlx_array_free(biases)

	return r, nil
}

func load_k_quantized(data unsafe.Pointer, name string, shape []C.int, dtype uint32, stream C.mlx_stream) (r C.mlx_array, err error) {
	size := 1
	for _, d := range shape {
		size *= int(d)
	}
	fdata := make([]float16.Float16, size)
	switch dtype {
	case 14:
		C.dequant_row_q6_K(
			data,
			unsafe.Pointer(&fdata[0]),
			C.int(size),
		)

	case 12:
		C.dequant_row_q4_K(
			data,
			unsafe.Pointer(&fdata[0]),
			C.int(size),
		)
	default:
		return r, fmt.Errorf("unsupported K quant")
	}

	r = C.mlx_array_new_data(
		unsafe.Pointer(&fdata[0]),
		&shape[0],
		C.int(len(shape)),
		C.MLX_FLOAT16,
	)
	return r, nil
}
