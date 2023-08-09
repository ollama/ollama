/**
 * llama.cpp - git 8183159cf3def112f6d1fe94815fce70e1bffa12
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "ggml.h"

#include <stdint.h>
#include <assert.h>
#include <stddef.h>

// Super-block size
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elemenets each
// Effectively 2.5625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_fp16_t d;           // super-block scale for quantized scales
    ggml_fp16_t dmin;        // super-block scale for quantized mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elemenets each
// Effectively 3.4375 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[2];
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 2, "wrong q3_K block size/padding");
#else
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[12];        // scales, quantized with 6 bits
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");
#endif

// 4-bit quantization
// 16 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d[2];          // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;             // super-block scale for quantized scales
    ggml_fp16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
#endif

// 5-bit quantization
// 16 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d;               // super-block scale
    int8_t  scales[QK_K/16];     // 8-bit block scales
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;               // super-block scale for quantized scales
    ggml_fp16_t dmin;            // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elemenets each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_fp16_t d;           // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");


// Quantization
void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int k);
void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int k);
void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int k);
void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int k);
void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k);
void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int k);

void quantize_row_q2_K(const float * restrict x, void * restrict y, int k);
void quantize_row_q3_K(const float * restrict x, void * restrict y, int k);
void quantize_row_q4_K(const float * restrict x, void * restrict y, int k);
void quantize_row_q5_K(const float * restrict x, void * restrict y, int k);
void quantize_row_q6_K(const float * restrict x, void * restrict y, int k);
void quantize_row_q8_K(const float * restrict x, void * restrict y, int k);

// Dequantization
void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int k);
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k);
void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k);
void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int k);
void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int k);
void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int k);

// Dot product
void ggml_vec_dot_q2_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);
void ggml_vec_dot_q3_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);
void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);
void ggml_vec_dot_q6_K_q8_K(int n, float * restrict s, const void * restrict vx, const void * restrict vy);

// Quantization with histogram collection
size_t ggml_quantize_q2_K(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q3_K(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q4_K(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q5_K(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist);

