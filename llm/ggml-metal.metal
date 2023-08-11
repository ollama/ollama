//go:build darwin

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

#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
    half d;          // delta
    half m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

static void dequantize_row_q4_0(device const block_q4_0 * x, device float * y, int k) {
    const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

static void dequantize_row_q4_1(device const block_q4_1 * x, device float * y, int k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;
        const half m = x[i].m;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

kernel void kernel_add(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % ne00];
}

kernel void kernel_mul(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % ne00];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

constant float GELU_COEF_A    = 0.044715f;
constant float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        threadgroup float  * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    buf[tpitg[0]] = -INFINITY;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] = MAX(buf[tpitg[0]], psrc0[i00]);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] = MAX(buf[tpitg[0]], buf[tpitg[0] + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float max = buf[0];

    // parallel sum
    buf[tpitg[0]] = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] += exp(psrc0[i00] - max);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] += buf[tpitg[0] + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sum = buf[0];

    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        pdst[i00] = exp(psrc0[i00] - max) / sum;
    }
}

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_get_rows_f16(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    for (int j = 0; j < ne00; j++) {
        dst[i*nb1 + j] = ((device half *) ((device char *) src0 + r*nb01))[j];
    }
}

kernel void kernel_get_rows_q4_0(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_0(
            (device const block_q4_0 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q4_1(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_1(
            (device const block_q4_1 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean  = sum[0];

    // recenter
    device float * y = dst + tgpig*ne00;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
    }

    // VARIANCE
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += y[i00] * y[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float variance = sum[0];

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}


kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);
    device const float * x_scalar = (device const float *) x;
    float4 sumf=0;
    float all_sum=0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (tiisg == 0) {
        sum[sgitg] = all_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast, simd group number is ntg / 32
    for (uint i = ntg / 32 / 2; i > 0; i /= 2) {
       if (tpitg < i) {
           sum[tpitg] += sum[tpitg + i];
       }
    }
    if (tpitg == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {sum[0] += x_scalar[i];}
        sum[0] /= ne00;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    device float * y_scalar = (device float *) y;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
    if (tpitg == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {y_scalar[i00] = x_scalar[i00] * scale;}
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float2 acc = 0.f;
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);
    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;
    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);
    float2 acc = 0.f;
    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4 // each SIMD group works on 4 rows
#define N_SIMDGROUP 2 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      giard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32(device const void * src0, device const float * src1, device float * dst,
                    int64_t ne00, int64_t ne10, int64_t ne0, int64_t ne01,
                    uint2 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * nsg + sgitg) * nr;
    device const block_q_type * x = (device const block_q_type *) src0 + first_row * nb;
    device const float      * y = (device const float      *) src1 + r1*ne10;
    float yl[16];       // src1 vector cache
    float sumf[nr]={0.f};

    const int ix = tiisg/2;
    const int il = 8*(tiisg%2);

    device const float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;
            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mat_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne01[[buffer(4)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne10,ne0,ne01,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mat_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne01[[buffer(4)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne10,ne0,ne01,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mat_f16_f32(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tpig[[thread_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    device const half  * x = (device const half  *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    sum[tpitg.x] = 0.0f;

    for (int i = tpitg.x; i < ne00; i += tptg.x) {
        sum[tpitg.x] += (float) x[i] * (float) y[i];
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tptg.x/2; i > 0; i /= 2) {
        if (tpitg.x < i) {
            sum[tpitg.x] += sum[tpitg.x + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tpitg.x == 0) {
        dst[im*ne1*ne0 + r1*ne0 + r0] = sum[0];
    }
}


kernel void kernel_alibi_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant      float & m0,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
    float m_k = pow(m0, i2 + 1);
    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0] + m_k * (i00 - ne00 + 1);
    }
}

kernel void kernel_rope(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant       int & n_past,
        constant       int & n_dims,
        constant       int & mode,
        constant     float & freq_base,
        constant     float & freq_scale,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i3 = tpig[2];
    const int64_t i2 = tpig[1];
    const int64_t i1 = tpig[0];

    const bool is_neox = mode & 2;
    const float theta_scale = pow(freq_base, -2.0f/n_dims);

    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);

    float theta = freq_scale * (float)p;

    if (!is_neox) {
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cos(theta);
            const float sin_theta = sin(theta);

            theta *= theta_scale;

            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        // TODO: implement
    }
}

kernel void kernel_cpy_f16_f16(
        device const half * src0,
        device       half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        device const float * src0,
        device        half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

//============================================ k-quants ======================================================

#ifndef QK_K
#define QK_K 256
#else
static_assert(QK_K == 256 || QK_K == 64, "QK_K must be 256 or 64");
#endif

#if QK_K == 256
#define K_SCALE_SIZE 12
#else
#define K_SCALE_SIZE 4
#endif

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half d;           // super-block scale for quantized scales
    half dmin;        // super-block scale for quantized mins
} block_q2_K;
// 84 bytes / block

typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#if QK_K == 64
    uint8_t scales[2];
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;

#if QK_K == 64
typedef struct {
    half    d[2];          // super-block scales/mins
    uint8_t scales[2];
    uint8_t qs[QK_K/2];    // 4-bit quants
} block_q4_K;
#else
typedef struct {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
#endif

#if QK_K == 64
typedef struct {
    half  d;                     // super-block scales/mins
    int8_t  scales[QK_K/16];     // 8-bit block scales
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
#else
typedef struct {
    half d;                      // super-block scale for quantized scales
    half dmin;                   // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
// 176 bytes / block
#endif

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
} block_q6_K;
// 210 bytes / block

static inline uchar4 get_scale_min_k4(int j, device const uint8_t * q) {
    uchar4 r;
    if (j < 4) {
        r[0] = q[j+0] & 63;
        r[2] = q[j+1] & 63;
        r[1] = q[j+4] & 63;
        r[3] = q[j+5] & 63;
    } else {
        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        r[2] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4);
        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
        r[3] = (q[j+5] >>  4) | ((q[j+1] >> 6) << 4);
    }
    return r;
}

//========================================== dequantization =============================

static void dequantize_row_q2_K(device const block_q2_K * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = x[i].d;
        const float min = x[i].dmin;

        device const uint8_t * q = x[i].qs;

#if QK_K == 256
        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
#else
        float dl1 = d * (x[i].scales[0] & 0xF), ml1 = min * (x[i].scales[0] >> 4);
        float dl2 = d * (x[i].scales[1] & 0xF), ml2 = min * (x[i].scales[1] >> 4);
        float dl3 = d * (x[i].scales[2] & 0xF), ml3 = min * (x[i].scales[2] >> 4);
        float dl4 = d * (x[i].scales[3] & 0xF), ml4 = min * (x[i].scales[3] >> 4);
        for (int l = 0; l < 16; ++l) {
            y[l+ 0] = dl1 * ((q[l] >> 0) & 3) - ml1;
            y[l+16] = dl2 * ((q[l] >> 2) & 3) - ml2;
            y[l+32] = dl3 * ((q[l] >> 4) & 3) - ml3;
            y[l+48] = dl4 * ((q[l] >> 6) & 3) - ml4;
        }
        y += QK_K;
#endif

    }
}

static void dequantize_row_q3_K(device const block_q3_K * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    uint16_t aux[8];
    thread const int8_t * scales = (thread const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = (float)(x[i].d);

        device const uint8_t * q = x[i].qs;
        device const uint8_t * h = x[i].hmask;
        uint8_t m = 1;

        device const uint16_t * a = (device const uint16_t *)x[i].scales;
        aux[0] = (a[0] & kmask2) | (((a[4] >> 0) & kmask1) << 4);
        aux[1] = (a[1] & kmask2) | (((a[5] >> 0) & kmask1) << 4);
        aux[2] = (a[2] & kmask2) | (((a[4] >> 2) & kmask1) << 4);
        aux[3] = (a[3] & kmask2) | (((a[5] >> 2) & kmask1) << 4);
        aux[4] = ((a[0] >> 4) & kmask2) | (((a[4] >> 4) & kmask1) << 4);
        aux[5] = ((a[1] >> 4) & kmask2) | (((a[5] >> 4) & kmask1) << 4);
        aux[6] = ((a[2] >> 4) & kmask2) | (((a[4] >> 6) & kmask1) << 4);
        aux[7] = ((a[3] >> 4) & kmask2) | (((a[5] >> 6) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((h[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((h[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
#else
    for (int i = 0; i < nb; i++) {

        const float d_all = (float)(x[i].d);

        device const uint8_t * q = x[i].qs;
        device const uint8_t * hm = x[i].hmask;

        const float d1 = d_all * ((x[i].scales[0] & 0xF) - 8);
        const float d2 = d_all * ((x[i].scales[0] >>  4) - 8);
        const float d3 = d_all * ((x[i].scales[1] & 0xF) - 8);
        const float d4 = d_all * ((x[i].scales[1] >>  4) - 8);

        for (int l = 0; l < 8; ++l) {
            uint8_t h = hm[l];
            y[l+ 0] = d1 * ((int8_t)((q[l+0] >> 0) & 3) - ((h & 0x01) ? 0 : 4));
            y[l+ 8] = d1 * ((int8_t)((q[l+8] >> 0) & 3) - ((h & 0x02) ? 0 : 4));
            y[l+16] = d2 * ((int8_t)((q[l+0] >> 2) & 3) - ((h & 0x04) ? 0 : 4));
            y[l+24] = d2 * ((int8_t)((q[l+8] >> 2) & 3) - ((h & 0x08) ? 0 : 4));
            y[l+32] = d3 * ((int8_t)((q[l+0] >> 4) & 3) - ((h & 0x10) ? 0 : 4));
            y[l+40] = d3 * ((int8_t)((q[l+8] >> 4) & 3) - ((h & 0x20) ? 0 : 4));
            y[l+48] = d4 * ((int8_t)((q[l+0] >> 6) & 3) - ((h & 0x40) ? 0 : 4));
            y[l+56] = d4 * ((int8_t)((q[l+8] >> 6) & 3) - ((h & 0x80) ? 0 : 4));
        }
        y += QK_K;
    }
#endif

}

static void dequantize_row_q4_K(device const block_q4_K * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        device const uint8_t * q = x[i].qs;

#if QK_K == 256
        const float d = x[i].d;
        const float min = x[i].dmin;

        device const uint8_t * scales = x[i].scales;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            const uchar4 sc = get_scale_min_k4(is, scales);
            const float d1 = d * sc[0]; const float m1 = min * sc[1];
            const float d2 = d * sc[2]; const float m2 = min * sc[3];
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
#else
        device const uint8_t * s = x[i].scales;
        device const half2 * dh = (device const half2 *)x[i].d;
        const float2 d = (float2)dh[0];
        const float d1 = d[0] * (s[0] & 0xF);
        const float d2 = d[0] * (s[1] & 0xF);
        const float m1 = d[1] * (s[0] >>  4);
        const float m2 = d[1] * (s[1] >>  4);
        for (int l = 0; l < 32; ++l) {
            y[l+ 0] = d1 * (q[l] & 0xF) - m1;
            y[l+32] = d2 * (q[l] >>  4) - m2;
        }
        y += QK_K;
#endif

    }
}

static void dequantize_row_q5_K(device const block_q5_K * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

#if QK_K == 256
   for (int i = 0; i < nb; i++) {

        const float d = (float)(x[i].d);
        const float min = (float)(x[i].dmin);

        device const uint8_t * ql = x[i].qs;
        device const uint8_t * qh = x[i].qh;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            const uchar4 sc = get_scale_min_k4(is, x[i].scales);
            const float d1 = d * sc[0]; const float m1 = min * sc[1];
            const float d2 = d * sc[2]; const float m2 = min * sc[3];
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
#else
    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;

        device const uint8_t * ql = x[i].qs;
        device const uint8_t * qh = x[i].qh;
        device const int8_t  * sc = x[i].scales;

        for (int l = 0; l < 8; ++l) {
            y[l+ 0] = d * sc[0] * ((ql[l+ 0] & 0xF) - (qh[l] & 0x01 ? 0 : 16));
            y[l+ 8] = d * sc[0] * ((ql[l+ 8] & 0xF) - (qh[l] & 0x02 ? 0 : 16));
            y[l+16] = d * sc[1] * ((ql[l+16] & 0xF) - (qh[l] & 0x04 ? 0 : 16));
            y[l+24] = d * sc[1] * ((ql[l+24] & 0xF) - (qh[l] & 0x08 ? 0 : 16));
            y[l+32] = d * sc[2] * ((ql[l+ 0] >>  4) - (qh[l] & 0x10 ? 0 : 16));
            y[l+40] = d * sc[2] * ((ql[l+ 8] >>  4) - (qh[l] & 0x20 ? 0 : 16));
            y[l+48] = d * sc[3] * ((ql[l+16] >>  4) - (qh[l] & 0x40 ? 0 : 16));
            y[l+56] = d * sc[3] * ((ql[l+24] >>  4) - (qh[l] & 0x80 ? 0 : 16));
        }
        y += QK_K;
    }
#endif

}

static void dequantize_row_q6_K(device const block_q6_K * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        device const uint8_t * ql = x[i].ql;
        device const uint8_t * qh = x[i].qh;
        device const int8_t  * sc = x[i].scales;

        const float d = x[i].d;

#if QK_K == 256
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
#else
        for (int l = 0; l < 16; ++l) {
            const int8_t q1 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l+16]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l+ 0] = d * sc[0] * q1;
            y[l+16] = d * sc[1] * q2;
            y[l+32] = d * sc[2] * q3;
            y[l+48] = d * sc[3] * q4;
        }
        y  += 64;
#endif
    }
}

kernel void kernel_get_rows_q2_K(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q2_K(
            (device const block_q2_K *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q3_K(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q3_K(
            (device const block_q3_K *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q4_K(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_K(
            (device const block_q4_K *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q5_K(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q5_K(
            (device const block_q5_K *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q6_K(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q6_K(
            (device const block_q6_K *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

//====================================== dot products =========================

kernel void kernel_mul_mat_q2_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne01[[buffer(4)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;
    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row;
    device const float      * y = (device const float      *) src1 + r1*ne10;
    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q2_K) * nb;

#if QK_K == 256
    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int im = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * im + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*im + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 4 * QK_K;
    }
#else
    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0...1

    device const float * y4 = y + ix * QK_K + 8 * it;

    for (int ib = ix; ib < nb; ib += 16) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+16]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+32]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+48]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[1] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[2] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[3] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] >> 4) + sumy[1] * (sc[1] >> 4) + sumy[2] * (sc[2] >> 4) + sumy[3] * (sc[3] >> 4));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 16 * QK_K;
    }
#endif

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + first_row + row] = all_sum;
        }
    }
}

#if QK_K == 256
kernel void kernel_mul_mat_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    float yl[16];

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/2;
    const int ix  = tiisg%2;
    const int ip  = tid/8;          // 0 or 1
    const int il  = tid/2 - 4*ip;   // 0...3
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    const uint16_t m1 = 1 << (4*ip + il);
    const uint16_t m2 = m1 << 8;

    const int shift = 2*il;
    const uint16_t qm1 = 0x0003 << shift;
    const uint16_t qm2 = 0x0300 << shift;
    const int32_t v1 = 4 << shift;
    const int32_t v2 = 1024 << shift;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + 2*(il/2);
    const int ik = 4 + (il%2);

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    const int step = sizeof(block_q3_K) * nb / 2;

    device const float * y1 = yy + ix*QK_K + y_offset;

    float sumf1[2] = {0.f}, sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 2) {

        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0];
            yl[l+8] = y1[l+16];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {

            const float d_all = (float)dh[0];
            const char2 scales = as_type<char2>((uint16_t)(((a[il] >> s_shift1) & kmask2) | (((a[ik] >> s_shift2) & kmask1) << 4)));

            float s1 = 0, s2 = 0;
            for (int l = 0; l < n; l += 2) {
                const uint16_t qs = q[l/2];
                s1 += yl[l+0] * ((int32_t)(qs & qm1) - ((h[l/2] & m1) ? 0 : v1));
                s2 += yl[l+1] * ((int32_t)(qs & qm2) - ((h[l/2] & m2) ? 0 : v2));
            }
            float d = d_all * (s1 + 1.f/256.f * s2);
            sumf1[row] += d * scales[0];
            sumf2[row] += d;

            s1 = s2 = 0;
            for (int l = 0; l < n; l += 2) {
                const uint16_t qs = q[l/2+8];
                s1 += yl[l+8] * ((int32_t)(qs & qm1) - ((h[l/2+8] & m1) ? 0 : v1));
                s2 += yl[l+9] * ((int32_t)(qs & qm2) - ((h[l/2+8] & m2) ? 0 : v2));
            }
            d = d_all * (s1 + 1.f/256.f * s2);
            sumf1[row] += d * scales[1];
            sumf2[row] += d;

            q  += step;
            h  += step;
            a  += step;
            dh += step;

        }

        y1 += 2 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] - 32.f*sumf2[row]) / (1 << shift);
        const float tot = simd_sum(sumf);
        if (tiisg == 0) {
            dst[r1*ne0 + first_row + row] = tot;
        }
    }
}
#else
kernel void kernel_mul_mat_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    const int row = 2 * r0 + sgitg;

    device const block_q3_K * x = (device const block_q3_K *) src0 + row*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;
    const int ix = tiisg/4;
    const int il = 4 * (tiisg%4);// 0, 4, 8, 12
    const int im = il/8;         // 0, 0, 1, 1
    const int in = il%8;         // 0, 4, 0, 4

    float2 sum = {0.f, 0.f};

    for (int i = ix; i < nb; i += 8) {

        const float d_all = (float)(x[i].d);

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + il);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + in);
        device const uint16_t * s = (device const uint16_t *)(x[i].scales);
        device const float    * y = yy + i * QK_K + il;

        const float d1 = d_all * ((int32_t)(s[0] & 0x000F) - 8);
        const float d2 = d_all * ((int32_t)(s[0] & 0x00F0) - 128) * 1.f/64.f;
        const float d3 = d_all * ((int32_t)(s[0] & 0x0F00) - 2048) * 1.f/4096.f;
        const float d4 = d_all * ((int32_t)(s[0] & 0xF000) - 32768) * 1.f/262144.f;

        for (int l = 0; l < 4; l += 2) {
            const uint16_t hm = h[l/2] >> im;
            sum[0] += y[l+ 0] * d1 * ((int32_t)(q[l/2] & 0x0003) - ((hm & 0x0001) ? 0 :  4))
                    + y[l+16] * d2 * ((int32_t)(q[l/2] & 0x000c) - ((hm & 0x0004) ? 0 : 16))
                    + y[l+32] * d3 * ((int32_t)(q[l/2] & 0x0030) - ((hm & 0x0010) ? 0 : 64))
                    + y[l+48] * d4 * ((int32_t)(q[l/2] & 0x00c0) - ((hm & 0x0040) ? 0 : 256));
            sum[1] += y[l+ 1] * d1 * ((int32_t)(q[l/2] & 0x0300) - ((hm & 0x0100) ? 0 : 1024))
                    + y[l+17] * d2 * ((int32_t)(q[l/2] & 0x0c00) - ((hm & 0x0400) ? 0 : 4096))
                    + y[l+33] * d3 * ((int32_t)(q[l/2] & 0x3000) - ((hm & 0x1000) ? 0 : 16384))
                    + y[l+49] * d4 * ((int32_t)(q[l/2] & 0xc000) - ((hm & 0x4000) ? 0 : 65536));
        }

    }
    const float sumf = sum[0] + sum[1] * 1.f/256.f;

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + row] = tot;
    }

}
#endif

#if QK_K == 256
kernel void kernel_mul_mat_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne01[[buffer(4)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int im = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;
    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row;
    device const float      * y = (device const float      *) src1 + r1*ne10;
    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * im + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + im;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            sc += step;
            dh += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + first_row + row] = all_sum;
        }
    }
}
#else
kernel void kernel_mul_mat_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne01[[buffer(4)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int ix = tiisg/4;  // 0...7
    const int it = tiisg%4;  // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;
    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row;
    device const float      * y = (device const float      *) src1 + r1*ne10;
    float yl[8];
    float yh[8];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 8 * it;

    uint16_t sc16[4];

    for (int ib = ix; ib < nb; ib += 8) {

        float2 sumy = {0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i] = y4[i+ 0]; sumy[0] += yl[i];
            yh[i] = y4[i+32]; sumy[1] += yh[i];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;
        device const half     * dh = x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & 0x000f;
            sc16[1] = sc[0] & 0x0f00;
            sc16[2] = sc[0] & 0x00f0;
            sc16[3] = sc[0] & 0xf000;

            float2 acc1 = {0.f, 0.f};
            float2 acc2 = {0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (qs[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (qs[i/2] & 0x0F00);
                acc2[0] += yh[i+0] * (qs[i/2] & 0x00F0);
                acc2[1] += yh[i+1] * (qs[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc16[0] +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc16[1] * 1.f/4096.f) -
                         dmin * 1.f/16.f * (sumy[0] * sc16[2] + sumy[1] * sc16[3] * 1.f/256.f);

            qs += step;
            sc += step;
            dh += step;
        }

        y4 += 8 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + first_row + row] = all_sum;
        }
    }
}
#endif

kernel void kernel_mul_mat_q5_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    float sumf[2]={0.f};

    const int step = sizeof(block_q5_K) * nb;

#if QK_K == 256
#
    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int im  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1 = 1u << (2*im);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {

        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + im;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {

            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc = {0.f, 0.f, 0.f, 0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc[0] += yl[l+0] * ((uint16_t)(q1[l] & 0x0F) + (h & hm1 ? 16 : 0));
                acc[1] += yl[l+8] * ((uint16_t)(q1[l] & 0xF0) + (h & hm2 ? 256 : 0));
                acc[2] += yh[l+0] * ((uint16_t)(q2[l] & 0x0F) + (h & hm3 ? 16 : 0));
                acc[3] += yh[l+8] * ((uint16_t)(q2[l] & 0xF0) + (h & hm4 ? 256 : 0));
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (acc[0] * sc8[0] + acc[1] * sc8[1] * 1.f/16.f + acc[2] * sc8[4] + acc[3] * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            qh += step;
            dh += step/2;
            a  += step/2;

        }

        y1 += 4 * QK_K;

    }
#else
    float yl[8], yh[8];

    const int il = 4 * (tiisg/8);  // 0, 4, 8, 12
    const int ix = tiisg%8;
    const int im = il/8;         // 0, 0, 1, 1
    const int in = il%8;         // 0, 4, 0, 4

    device const float * y = yy + ix*QK_K + il;

    for (int i = ix; i < nb; i += 8) {

        for (int l = 0; l < 4; ++l) {
            yl[l+0] = y[l+ 0];
            yl[l+4] = y[l+16];
            yh[l+0] = y[l+32];
            yh[l+4] = y[l+48];
        }

        device const half * dh = &x[i].d;
        device const uint8_t * q = x[i].qs + il;
        device const uint8_t * h = x[i].qh + in;
        device const int8_t  * s = x[i].scales;

        for (int row = 0; row < 2; ++row) {

            const float d = dh[0];

            float2 acc = {0.f, 0.f};
            for (int l = 0; l < 4; ++l) {
                const uint8_t hl = h[l] >> im;
                acc[0] += yl[l+0] * s[0] * ((int16_t)(q[l+ 0] & 0x0F) - (hl & 0x01 ? 0 : 16))
                        + yl[l+4] * s[1] * ((int16_t)(q[l+16] & 0x0F) - (hl & 0x04 ? 0 : 16));
                acc[1] += yh[l+0] * s[2] * ((int16_t)(q[l+ 0] & 0xF0) - (hl & 0x10 ? 0 : 256))
                        + yh[l+4] * s[3] * ((int16_t)(q[l+16] & 0xF0) - (hl & 0x40 ? 0 : 256));
            }
            sumf[row] += d * (acc[0] + 1.f/16.f * acc[1]);

            q += step;
            h += step;
            s += step;
            dh += step/2;

        }

        y += 8 * QK_K;
    }
#endif

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + first_row + row] = tot;
        }
    }

}

kernel void kernel_mul_mat_q6_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    const int row = 2 * r0 + sgitg;

    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb; //r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    float sumf = 0;

#if QK_K == 256
    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {

        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

#else
    const int ix  = tiisg/4;
    const int il  = 4*(tiisg%4);

    for (int i = ix; i < nb; i += 8) {
        device const float * y = yy + i * QK_K + il;
        device const uint8_t * ql = x[i].ql + il;
        device const uint8_t * qh = x[i].qh + il;
        device const int8_t  * s  = x[i].scales;

        const float d = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 4; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((ql[l+ 0] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+16] * ((int8_t)((ql[l+16] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+32] * ((int8_t)((ql[l+ 0] >>  4) | ((qh[l] & kmask3) >> 0)) - 32);
            sums[3] += y[l+48] * ((int8_t)((ql[l+16] >>  4) | ((qh[l] & kmask4) >> 2)) - 32);
        }
        sumf += d * (sums[0] * s[0] + sums[1] * s[1] + sums[2] * s[2] + sums[3] * s[3]);
    }

#endif

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + row] = tot;
    }
}
