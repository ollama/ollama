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

#include "k_quants.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

//
// 2-6 bit quantization in super-blocks
//


//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    int weight_type = rmse_type%2;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = weight_type == 1 ? x[i] * x[i] : 1;
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = sumlx/suml2;
    float best = scale * sumlx;
    for (int itry = 0; itry < 3; ++itry) {
        iscale = 1/scale;
        float slx = 0;
        float sl2 = 0;
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            if (l + nmax != L[i]) { changed = true; }
            float w = weight_type == 1 ? x[i] * x[i] : 1.f;
            slx += w*x[i]*l;
            sl2 += w*l*l;
        }
        if (!changed || sl2 == 0 || slx*slx <= best*sl2) { break; }
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        sumlx = slx; suml2 = sl2;
        scale = sumlx/suml2;
        best = scale * sumlx;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = weight_type == 1 ? x[i]*x[i] : 1;
            int l = L[i] - nmax;
            float slx = sumlx - w*x[i]*l;
            if (slx > 0) {
                float sl2 = suml2 - w*l*l;
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MAX(-nmax, MIN(nmax-1, new_l));
                if (new_l != l) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = nmax + new_l; sumlx = slx; suml2 = sl2;
                        scale = sumlx / suml2; best = scale * sumlx;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) { break; }
    }
    if (rmse_type < 3) {
        return scale;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = weight_type == 1 ? x[i] * x[i] : 1;
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

static float make_qkx1_quants(int n, int nmax, const float * restrict x, uint8_t * restrict L, float * restrict the_min, int ntry) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

//========================- 2-bit (de)-quantization

void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {

        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_qkx1_quants(16, 3, x + 16*j, L + 16*j, &mins[j], 5);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = ggml_fp32_to_fp16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = ggml_fp32_to_fp16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = ggml_fp32_to_fp16(max_min/q4scale);
        } else {
            y[i].dmin = ggml_fp32_to_fp16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = ggml_fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = ggml_fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
#else
        for (int l = 0; l < 16; ++l) {
            y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * q = x[i].qs;

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
            y[l+ 0] = dl1 * ((int8_t)((q[l] >> 0) & 3)) - ml1;
            y[l+16] = dl2 * ((int8_t)((q[l] >> 2) & 3)) - ml2;
            y[l+32] = dl3 * ((int8_t)((q[l] >> 4) & 3)) - ml3;
            y[l+48] = dl4 * ((int8_t)((q[l] >> 6) & 3)) - ml4;
        }
        y += QK_K;
#endif
    }
}

void quantize_row_q2_K(const float * restrict x, void * restrict vy, int k) {
    quantize_row_q2_K_reference(x, vy, k);
}

size_t ggml_quantize_q2_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    const int nb = k / QK_K;

    // TODO - collect histograms - although, at a second thought, I don't really care about them
    (void)hist;

    for (int j = 0; j < nb; j += k) {
        block_q2_K * restrict y = (block_q2_K *)dst + j/QK_K;
        quantize_row_q2_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q2_K));
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

#if QK_K == 256
        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = ggml_fp32_to_fp16(1/iscale);
        } else {
            y[i].d = ggml_fp32_to_fp16(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = ggml_fp16_to_fp32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }
#else
        if (max_scale) {
            float iscale = -8.f/max_scale;
            for (int j = 0; j < QK_K/16; j+=2) {
                int l1 = nearest_int(iscale*scales[j]);
                l1 = 8 + MAX(-8, MIN(7, l1));
                int l2 = nearest_int(iscale*scales[j+1]);
                l2 = 8 + MAX(-8, MIN(7, l2));
                y[i].scales[j/2] = l1 | (l2 << 4);
            }
            y[i].d = ggml_fp32_to_fp16(1/iscale);
        } else {
            for (int j = 0; j < QK_K/16; j+=2) {
                y[i].scales[j/2] = 0;
            }
            y[i].d = ggml_fp32_to_fp16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int s = j%2 == 0 ? y[i].scales[j/2] & 0xF : y[i].scales[j/2] >> 4;
            float d = ggml_fp16_to_fp32(y[i].d) * (s - 8);
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }
#endif

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }
#else
        for (int l = 0; l < 16; ++l) {
            y[i].qs[l] = L[l] | (L[l + 16] << 2) | (L[l + 32] << 4) | (L[l + 48] << 6);
        }
#endif

        x += QK_K;
    }
}

#if QK_K == 256
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}
#else
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    assert(QK_K == 64);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d_all = ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;

        const float d1 = d_all * ((x[i].scales[0] & 0xF) - 8);
        const float d2 = d_all * ((x[i].scales[0] >>  4) - 8);
        const float d3 = d_all * ((x[i].scales[1] & 0xF) - 8);
        const float d4 = d_all * ((x[i].scales[1] >>  4) - 8);

        for (int l=0; l<8; ++l) {
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
}
#endif

void quantize_row_q3_K(const float * restrict x, void * restrict vy, int k) {
    quantize_row_q3_K_reference(x, vy, k);
}

size_t ggml_quantize_q3_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    const int nb = k / QK_K;

    // TODO - collect histograms - although, at a second thought, I don't really care about them
    (void)hist;

    for (int j = 0; j < nb; j += k) {
        block_q3_K * restrict y = (block_q3_K *)dst + j/QK_K;
        quantize_row_q3_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q3_K));
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 5);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

#if QK_K == 256
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = ggml_fp32_to_fp16(max_scale/63.f);
        y[i].dmin = ggml_fp32_to_fp16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = ggml_fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = ggml_fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }
#else
        const float s_factor = 15.f;
        float inv_scale = max_scale > 0 ? s_factor/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? s_factor/max_min   : 0.f;
        int d1 = nearest_int(inv_scale*scales[0]);
        int m1 = nearest_int(inv_min*mins[0]);
        int d2 = nearest_int(inv_scale*scales[1]);
        int m2 = nearest_int(inv_min*mins[1]);
        y[i].scales[0] = d1 | (m1 << 4);
        y[i].scales[1] = d2 | (m2 << 4);
        y[i].d[0] = ggml_fp32_to_fp16(max_scale/s_factor);
        y[i].d[1] = ggml_fp32_to_fp16(max_min/s_factor);

        float sumlx = 0;
        int   suml2 = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            const uint8_t sd = y[i].scales[j] & 0xF;
            const uint8_t sm = y[i].scales[j] >>  4;
            const float d = ggml_fp16_to_fp32(y[i].d[0]) * sd;
            if (!d) continue;
            const float m = ggml_fp16_to_fp32(y[i].d[1]) * sm;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + m)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
                sumlx += (x[32*j + ii] + m)*l*sd;
                suml2 += l*l*sd*sd;
            }
        }
        if (suml2) {
            y[i].d[0] = ggml_fp32_to_fp16(sumlx/suml2);
        }
#endif
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;

    }
}

void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * q = x[i].qs;

#if QK_K == 256

        const float d   = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
#else
        const float dall = ggml_fp16_to_fp32(x[i].d[0]);
        const float mall = ggml_fp16_to_fp32(x[i].d[1]);
        const float d1 = dall * (x[i].scales[0] & 0xF), m1 = mall * (x[i].scales[0] >> 4);
        const float d2 = dall * (x[i].scales[1] & 0xF), m2 = mall * (x[i].scales[1] >> 4);
        for (int l = 0; l < 32; ++l) {
            y[l+ 0] = d1 * (q[l] & 0xF) - m1;
            y[l+32] = d2 * (q[l] >>  4) - m2;
        }
        y += QK_K;
#endif

    }
}

void quantize_row_q4_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q4_K * restrict y = vy;
    quantize_row_q4_K_reference(x, y, k);
}

size_t ggml_quantize_q4_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    (void)hist; // TODO: collect histograms
    for (int j = 0; j < nb; j += k) {
        block_q4_K * restrict y = (block_q4_K *)dst + j/QK_K;
        quantize_row_q4_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q4_K));
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

#if QK_K == 256
    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
#else
    int8_t L[QK_K];
    float scales[QK_K/16];
#endif

    for (int i = 0; i < nb; i++) {

#if QK_K == 256

        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 5);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = ggml_fp32_to_fp16(max_scale/63.f);
        y[i].dmin = ggml_fp32_to_fp16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = ggml_fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = ggml_fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * restrict qh = y[i].qh;
        uint8_t * restrict ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }
#else
        float max_scale = 0, amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_qx_quants(16, 16, x + 16*j, L + 16*j, 1);
            float abs_scale = fabsf(scales[j]);
            if (abs_scale > amax) {
                amax = abs_scale;
                max_scale = scales[j];
            }
        }

        float iscale = -128.f/max_scale;
        for (int j = 0; j < QK_K/16; ++j) {
            int l = nearest_int(iscale*scales[j]);
            y[i].scales[j] = MAX(-128, MIN(127, l));
        }
        y[i].d = ggml_fp32_to_fp16(1/iscale);

        for (int j = 0; j < QK_K/16; ++j) {
            const float d = ggml_fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-16, MIN(15, l));
                L[16*j + ii] = l + 16;
            }
        }

        uint8_t * restrict qh = y[i].qh;
        uint8_t * restrict ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        for (int j = 0; j < 32; ++j) {
            int jm = j%8;
            int is = j/8;
            int l1 = L[j];
            if (l1 > 15) {
                l1 -= 16; qh[jm] |= (1 << is);
            }
            int l2 = L[j + 32];
            if (l2 > 15) {
                l2 -= 16; qh[jm] |= (1 << (4 + is));
            }
            ql[j] = l1 | (l2 << 4);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

#if QK_K == 256

        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
#else
        float d = ggml_fp16_to_fp32(x[i].d);
        const int8_t * restrict s = x[i].scales;
        for (int l = 0; l < 8; ++l) {
            y[l+ 0] = d * s[0] * ((ql[l+ 0] & 0xF) - (qh[l] & 0x01 ? 0 : 16));
            y[l+ 8] = d * s[0] * ((ql[l+ 8] & 0xF) - (qh[l] & 0x02 ? 0 : 16));
            y[l+16] = d * s[1] * ((ql[l+16] & 0xF) - (qh[l] & 0x04 ? 0 : 16));
            y[l+24] = d * s[1] * ((ql[l+24] & 0xF) - (qh[l] & 0x08 ? 0 : 16));
            y[l+32] = d * s[2] * ((ql[l+ 0] >>  4) - (qh[l] & 0x10 ? 0 : 16));
            y[l+40] = d * s[2] * ((ql[l+ 8] >>  4) - (qh[l] & 0x20 ? 0 : 16));
            y[l+48] = d * s[3] * ((ql[l+16] >>  4) - (qh[l] & 0x40 ? 0 : 16));
            y[l+56] = d * s[3] * ((ql[l+24] >>  4) - (qh[l] & 0x80 ? 0 : 16));
        }
        y += QK_K;
#endif
    }
}

void quantize_row_q5_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q5_K * restrict y = vy;
    quantize_row_q5_K_reference(x, y, k);
}

size_t ggml_quantize_q5_K(const float * restrict src, void * restrict dst, int n, int k, int64_t * restrict hist) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    (void)hist;
    for (int j = 0; j < nb; j += k) {
        block_q5_K * restrict y = (block_q5_K *)dst + j/QK_K;
        quantize_row_q5_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q5_K));
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        float iscale = -128.f/max_scale;
        y[i].d = ggml_fp32_to_fp16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = ggml_fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * restrict ql = y[i].ql;
        uint8_t * restrict qh = y[i].qh;
#if QK_K == 256
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }
#else
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = L[l +  0] & 0xF;
            const uint8_t q2 = L[l + 32] & 0xF;
            ql[l] = q1 | (q2 << 4);
        }
        for (int l = 0; l < 16; ++l) {
            qh[l] = (L[l] >> 4) | ((L[l + 16] >> 4) << 2) | ((L[l + 32] >> 4) << 4) | ((L[l + 48] >> 4) << 6);
        }
#endif

        x += QK_K;

    }
}

void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

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

void quantize_row_q6_K(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK_K == 0);
    block_q6_K * restrict y = vy;
    quantize_row_q6_K_reference(x, y, k);
}

size_t ggml_quantize_q6_K(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    (void)hist; // TODO

    for (int j = 0; j < nb; j += k) {
        block_q6_K * restrict y = (block_q6_K *)dst + j/QK_K;
        quantize_row_q6_K_reference(src + j, y, k);
    }
    return (n/QK_K*sizeof(block_q6_K));
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        const float iscale = -128.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void quantize_row_q8_K(const float * restrict x, void * restrict y, int k) {
    quantize_row_q8_K_reference(x, y, k);
}

//===================================== Dot ptoducts =================================

//
// Helper functions
//
#if __AVX__ || __AVX2__ || __AVX512F__

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

#if QK_K == 256
void ggml_vec_dot_q2_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);

    int8x16x2_t q2bytes;
    uint8_t aux[16];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        const uint8_t * restrict sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8x2_t mins16 = {vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))};
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[0]), vget_low_s16 (q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[1]), vget_low_s16 (q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#if defined(__ARM_FEATURE_DOTPROD)
#define MULTIPLY_ACCUM_WITH_SCALE(index)\
        isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is+(index)];\
        isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is+1+(index)];
#else
#define MULTIPLY_ACCUM_WITH_SCALE(index)\
        {\
    const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[0]), vget_low_s8 (q8bytes.val[0])),\
                                   vmull_s8(vget_high_s8(q2bytes.val[0]), vget_high_s8(q8bytes.val[0])));\
    const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[1]), vget_low_s8 (q8bytes.val[1])),\
                                   vmull_s8(vget_high_s8(q2bytes.val[1]), vget_high_s8(q8bytes.val[1])));\
    isum += vaddvq_s16(p1) * aux[is+(index)] + vaddvq_s16(p2) * aux[is+1+(index)];\
        }
#endif

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)\
        q8bytes = vld1q_s8_x2(q8); q8 += 32;\
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));\
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));\
        MULTIPLY_ACCUM_WITH_SCALE((index));


        for (int j = 0; j < QK_K/128; ++j) {

            const uint8x16x2_t q2bits = vld1q_u8_x2(q2); q2 += 32;

            int8x16x2_t q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));
            MULTIPLY_ACCUM_WITH_SCALE(0);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

            is += 8;
        }
        sum += d * isum;

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float dall = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
        const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

        const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        const __m128i scales[2] = { scales_0, scales_1 };

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {

            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = _mm_add_epi32(p0, p1);
            p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5);
            p6 = _mm_add_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

#else

void ggml_vec_dot_q2_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const int32x4_t  vzero = vdupq_n_s32(0);

    int8x16x4_t q2bytes;

    uint32_t aux32[2];
    const uint8_t * scales = (const uint8_t *)aux32;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const float dmin = -y[i].d * (float)x[i].dmin;

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;

        aux32[0] = sc[0] & 0x0f0f0f0f;
        aux32[1] = (sc[0] >> 4) & 0x0f0f0f0f;

        sum += dmin * (scales[4] * y[i].bsums[0] + scales[5] * y[i].bsums[1] + scales[6] * y[i].bsums[2] + scales[7] * y[i].bsums[3]);

        int isum1 = 0, isum2 = 0;

        const uint8x16_t q2bits = vld1q_u8(q2);

        const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits, m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 2), m3));
        q2bytes.val[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 4), m3));
        q2bytes.val[3] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits, 6), m3));

#if defined(__ARM_FEATURE_DOTPROD)
        isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * scales[0];
        isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * scales[1];
        isum1 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[2], q8bytes.val[2])) * scales[2];
        isum2 += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[3], q8bytes.val[3])) * scales[3];
#else
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q2bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q2bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        isum1 += vaddvq_s16(p1) * scales[0];
        isum2 += vaddvq_s16(p2) * scales[1];

        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q2bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p4 = vaddq_s16(vmull_s8(vget_low_s8 (q2bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q2bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum1 += vaddvq_s16(p3) * scales[2];
        isum2 += vaddvq_s16(p4) * scales[3];
#endif
        sum += d * (isum1 + isum2);

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);

    __m256 acc = _mm256_setzero_ps();

    uint32_t ud, um;
    const uint8_t * restrict db = (const uint8_t *)&ud;
    const uint8_t * restrict mb = (const uint8_t *)&um;

    float summs = 0;

    // TODO: optimize this

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;
        ud = (sc[0] >> 0) & 0x0f0f0f0f;
        um = (sc[0] >> 4) & 0x0f0f0f0f;

        int32_t smin = mb[0] * y[i].bsums[0] + mb[1] * y[i].bsums[1] + mb[2] * y[i].bsums[2] + mb[3] * y[i].bsums[3];
        summs += dmin * smin;

        const __m128i q2bits = _mm_loadu_si128((const __m128i*)q2);
        const __m256i q2_0 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q2bits, 2), q2bits), m3);
        const __m256i q2_1 = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q2bits, 6), _mm_srli_epi16(q2bits, 4)), m3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
        const __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);

        const __m256i p_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p0, 0));
        const __m256i p_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p0, 1));
        const __m256i p_2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p1, 0));
        const __m256i p_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p1, 1));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[0]), _mm256_cvtepi32_ps(p_0), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[1]), _mm256_cvtepi32_ps(p_1), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[2]), _mm256_cvtepi32_ps(p_2), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * db[3]), _mm256_cvtepi32_ps(p_3), acc);
    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);

    __m256 acc = _mm256_setzero_ps();

    uint32_t ud, um;
    const uint8_t * restrict db = (const uint8_t *)&ud;
    const uint8_t * restrict mb = (const uint8_t *)&um;

    float summs = 0;

    // TODO: optimize this

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint32_t * restrict sc = (const uint32_t *)x[i].scales;
        ud = (sc[0] >> 0) & 0x0f0f0f0f;
        um = (sc[0] >> 4) & 0x0f0f0f0f;

        int32_t smin = mb[0] * y[i].bsums[0] + mb[1] * y[i].bsums[1] + mb[2] * y[i].bsums[2] + mb[3] * y[i].bsums[3];
        summs += dmin * smin;

        const __m128i q2bits = _mm_loadu_si128((const __m128i*)q2);
        const __m128i q2_0 = _mm_and_si128(q2bits, m3);
        const __m128i q2_1 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
        const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
        const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p0 = _mm_maddubs_epi16(q2_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i p1 = _mm_maddubs_epi16(q2_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i p2 = _mm_maddubs_epi16(q2_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i p3 = _mm_maddubs_epi16(q2_3, _mm256_extractf128_si256(q8_1, 1));

        const __m256i p_0 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p0, p0)), _mm_cvtepi16_epi32(p0));
        const __m256i p_1 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p1, p1)), _mm_cvtepi16_epi32(p1));
        const __m256i p_2 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p2, p2)), _mm_cvtepi16_epi32(p2));
        const __m256i p_3 = MM256_SET_M128I(_mm_cvtepi16_epi32(_mm_unpackhi_epi64(p3, p3)), _mm_cvtepi16_epi32(p3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[0]), _mm256_cvtepi32_ps(p_0)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[1]), _mm256_cvtepi32_ps(p_1)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[2]), _mm256_cvtepi32_ps(p_2)), acc);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d * db[3]), _mm256_cvtepi32_ps(p_3)), acc);
    }

    *s = hsum_float_8(acc) + summs;

#else

    float sumf = 0;

    int isum[4];

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        isum[0] = isum[1] = isum[2] = isum[3] = 0;
        for (int l =  0; l < 16; ++l) {
            isum[0] += q8[l+ 0] * ((q2[l] >> 0) & 3);
            isum[1] += q8[l+16] * ((q2[l] >> 2) & 3);
            isum[2] += q8[l+32] * ((q2[l] >> 4) & 3);
            isum[3] += q8[l+48] * ((q2[l] >> 6) & 3);
        }
        for (int l = 0; l < 4; ++l) {
            isum[l] *= (sc[l] & 0xF);
        }
        sumf += dall * (isum[0] + isum[1] + isum[2] + isum[3]) - dmin * summs;
    }
    *s = sumf;
#endif
}
#endif

#if QK_K == 256
void ggml_vec_dot_q3_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    uint32_t aux[3];
    uint32_t utmp[4];

    const uint8x16_t m3b = vdupq_n_u8(0x3);
#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t  vzero = vdupq_n_s32(0);
#endif

    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0, 1);
    const uint8x16_t m2 = vshlq_n_u8(m0, 2);
    const uint8x16_t m3 = vshlq_n_u8(m0, 3);
    const int8_t m32 = 32;

    int8x16x4_t q3bytes;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const int8_t  * restrict q8 = y[i].qs;

        uint8x16x2_t qhbits = vld1q_u8_x2(qh);

        uint8x16x4_t q3h;

        int32_t isum = 0;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        for (int j = 0; j < QK_K/128; ++j) {

            const uint8x16x2_t q3bits = vld1q_u8_x2(q3); q3 += 32;
            const int8x16x4_t q8bytes_1 = vld1q_s8_x4(q8); q8 += 64;
            const int8x16x4_t q8bytes_2 = vld1q_s8_x4(q8); q8 += 64;

            q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
            q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
            q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
            q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];
#else
            int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[0]), vget_low_s8 (q8bytes_1.val[0])),
                                     vmull_s8(vget_high_s8(q3bytes.val[0]), vget_high_s8(q8bytes_1.val[0])));
            int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[1]), vget_low_s8 (q8bytes_1.val[1])),
                                     vmull_s8(vget_high_s8(q3bytes.val[1]), vget_high_s8(q8bytes_1.val[1])));
            int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[2]), vget_low_s8 (q8bytes_1.val[2])),
                                     vmull_s8(vget_high_s8(q3bytes.val[2]), vget_high_s8(q8bytes_1.val[2])));
            int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[3]), vget_low_s8 (q8bytes_1.val[3])),
                                     vmull_s8(vget_high_s8(q3bytes.val[3]), vget_high_s8(q8bytes_1.val[3])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1] + vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif
            scale += 4;

            q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
            q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
            q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
            q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
            isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];
#else
            p0 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[0]), vget_low_s8 (q8bytes_2.val[0])),
                           vmull_s8(vget_high_s8(q3bytes.val[0]), vget_high_s8(q8bytes_2.val[0])));
            p1 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[1]), vget_low_s8 (q8bytes_2.val[1])),
                           vmull_s8(vget_high_s8(q3bytes.val[1]), vget_high_s8(q8bytes_2.val[1])));
            p2 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[2]), vget_low_s8 (q8bytes_2.val[2])),
                           vmull_s8(vget_high_s8(q3bytes.val[2]), vget_high_s8(q8bytes_2.val[2])));
            p3 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[3]), vget_low_s8 (q8bytes_2.val[3])),
                           vmull_s8(vget_high_s8(q3bytes.val[3]), vget_high_s8(q8bytes_2.val[3])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1] + vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif
            scale += 4;

            if (j == 0) {
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
            }

        }
        sum += d * isum;

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        // high bit
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        // integer accumulator
        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is  = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));

        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i mone = _mm_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    uint32_t *aux;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // Set up scales
        aux = (uint32_t *)x[i].scales;
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        const __m128i scales[2] = { scales_0, scales_1 };

        // high bit *128*2 from block_q3_K.hmask[QK_K/8]
        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].hmask[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].hmask[16]);

        // integer accumulator
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
            const __m128i q3bits_0 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;
            const __m128i q3bits_1 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;

            // prepare low and high bits
            const int bit = j << 2;

            const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);

            const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            const __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+1)), bit+1), 2);
            const __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+1)), bit+1), 2);

            const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            const __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+2)), bit+2), 2);
            const __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+2)), bit+2), 2);

            const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            const __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+3)), bit+3), 2);
            const __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+3)), bit+3), 2);

            // load Q8 quants from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            // multiply with scales
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            // accumulate
            p16_0 = _mm_add_epi32(p16_0, p16_1);
            p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5);
            p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));

        }

        // multiply with block scale and accumulate
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc);

#else
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}

#else

void ggml_vec_dot_q3_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t  vzero = vdupq_n_s32(0);
#endif

    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const uint8x16_t mh  = vdupq_n_u8(4);

    int8x16x4_t q3bytes;

    uint16_t aux16[2];
    int8_t * scales = (int8_t *)aux16;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        uint8x16x4_t q3h;

        const uint8x8_t  hbits    = vld1_u8(x[i].hmask);
        const uint8x16_t q3bits   = vld1q_u8(x[i].qs);
        const int8x16x4_t q8bytes = vld1q_s8_x4(y[i].qs);

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        for (int j = 0; j < 4; ++j) scales[j] -= 8;

        int32_t isum = -4*(scales[0] * y[i].bsums[0] + scales[2] * y[i].bsums[1] + scales[1] * y[i].bsums[2] + scales[3] * y[i].bsums[3]);

        const float d = y[i].d * (float)x[i].d;

        const uint8x16_t htmp = vcombine_u8(hbits, vshr_n_u8(hbits, 1));
        q3h.val[0] = vandq_u8(mh, vshlq_n_u8(htmp, 2));
        q3h.val[1] = vandq_u8(mh, htmp);
        q3h.val[2] = vandq_u8(mh, vshrq_n_u8(htmp, 2));
        q3h.val[3] = vandq_u8(mh, vshrq_n_u8(htmp, 4));

        q3bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q3bits, m3b),                q3h.val[0]));
        q3bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 2), m3b), q3h.val[1]));
        q3bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(vshrq_n_u8(q3bits, 4), m3b), q3h.val[2]));
        q3bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q3bits, 6),                q3h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[0], q8bytes.val[0])) * scales[0];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[1], q8bytes.val[1])) * scales[2];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[2], q8bytes.val[2])) * scales[1];
        isum += vaddvq_s32(vdotq_s32(vzero, q3bytes.val[3], q8bytes.val[3])) * scales[3];
#else
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q3bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q3bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q3bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q3bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q3bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum += vaddvq_s16(p0) * scales[0] + vaddvq_s16(p1) * scales[2] + vaddvq_s16(p2) * scales[1] + vaddvq_s16(p3) * scales[3];
#endif

        sum += d * isum;

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m1 = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    uint64_t aux64;

    uint16_t aux16[2];
    const int8_t * aux8 = (const int8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        const __m256i scale_0 = MM256_SET_M128I(_mm_set1_epi16(aux8[2] - 8), _mm_set1_epi16(aux8[0] - 8));
        const __m256i scale_1 = MM256_SET_M128I(_mm_set1_epi16(aux8[3] - 8), _mm_set1_epi16(aux8[1] - 8));

        memcpy(&aux64, x[i].hmask, 8);

        const __m128i haux = _mm_set_epi64x(aux64 >> 1, aux64 >> 0);
        __m256i q3h_0 = MM256_SET_M128I(_mm_srli_epi16(haux, 2), haux);
        __m256i q3h_1 = _mm256_srli_epi16(q3h_0, 4);
        q3h_0 = _mm256_slli_epi16(_mm256_andnot_si256(q3h_0, m1), 2);
        q3h_1 = _mm256_slli_epi16(_mm256_andnot_si256(q3h_1, m1), 2);

        // load low 2 bits
        const __m128i q3bits = _mm_loadu_si128((const __m128i*)q3);

        // prepare low and high bits
        const __m256i q3aux  = MM256_SET_M128I(_mm_srli_epi16(q3bits, 2), q3bits);
        const __m256i q3l_0 = _mm256_and_si256(q3aux, m3);
        const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3aux, 4), m3);

        // load Q8 quants
        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
        // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
        // and 2 if the high bit was set)
        const __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
        const __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);

        __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);

        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

        // multiply with scales
        p16_0 = _mm256_madd_epi16(scale_0, p16_0);
        p16_1 = _mm256_madd_epi16(scale_1, p16_1);

        p16_0 = _mm256_add_epi32(p16_0, p16_1);

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(p16_0), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m1 = _mm_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    uint64_t aux64;

    uint16_t aux16[2];
    const int8_t * aux8 = (const int8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t a = *(const uint16_t *)x[i].scales;
        aux16[0] = a & 0x0f0f;
        aux16[1] = (a >> 4) & 0x0f0f;

        const __m128i scale_0 = _mm_set1_epi16(aux8[0] - 8);
        const __m128i scale_1 = _mm_set1_epi16(aux8[2] - 8);
        const __m128i scale_2 = _mm_set1_epi16(aux8[1] - 8);
        const __m128i scale_3 = _mm_set1_epi16(aux8[3] - 8);

        memcpy(&aux64, x[i].hmask, 8);

        __m128i q3h_0 = _mm_set_epi64x(aux64 >> 1, aux64 >> 0);
        __m128i q3h_1 = _mm_srli_epi16(q3h_0, 2);
        __m128i q3h_2 = _mm_srli_epi16(q3h_0, 4);
        __m128i q3h_3 = _mm_srli_epi16(q3h_0, 6);
        q3h_0 = _mm_slli_epi16(_mm_andnot_si128(q3h_0, m1), 2);
        q3h_1 = _mm_slli_epi16(_mm_andnot_si128(q3h_1, m1), 2);
        q3h_2 = _mm_slli_epi16(_mm_andnot_si128(q3h_2, m1), 2);
        q3h_3 = _mm_slli_epi16(_mm_andnot_si128(q3h_3, m1), 2);

        // load low 2 bits
        const __m128i q3bits = _mm_loadu_si128((const __m128i*)q3);

        // prepare low and high bits
        const __m128i q3l_0 = _mm_and_si128(q3bits, m3);
        const __m128i q3l_1 = _mm_and_si128(_mm_srli_epi16(q3bits, 2), m3);
        const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits, 4), m3);
        const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits, 6), m3);

        // load Q8 quants
        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm_maddubs_epi16,
        // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
        // and 2 if the high bit was set)
        const __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, _mm256_extractf128_si256(q8_1, 1));

        __m128i p16_0 = _mm_maddubs_epi16(q3l_0, _mm256_extractf128_si256(q8_0, 0));
        __m128i p16_1 = _mm_maddubs_epi16(q3l_1, _mm256_extractf128_si256(q8_0, 1));
        __m128i p16_2 = _mm_maddubs_epi16(q3l_2, _mm256_extractf128_si256(q8_1, 0));
        __m128i p16_3 = _mm_maddubs_epi16(q3l_3, _mm256_extractf128_si256(q8_1, 1));

        p16_0 = _mm_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm_sub_epi16(p16_3, q8s_3);

        // multiply with scales
        p16_0 = _mm_madd_epi16(scale_0, p16_0);
        p16_1 = _mm_madd_epi16(scale_1, p16_1);
        p16_2 = _mm_madd_epi16(scale_2, p16_2);
        p16_3 = _mm_madd_epi16(scale_3, p16_3);

        p16_0 = _mm_add_epi32(p16_0, p16_2);
        p16_1 = _mm_add_epi32(p16_1, p16_3);
        __m256i p16 = MM256_SET_M128I(p16_1, p16_0);

        // multiply with block scale and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(p16)), acc);

    }

    *s = hsum_float_8(acc);

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    int32_t scales[4];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        int8_t * restrict a = aux8;
        for (int l = 0; l < 8; ++l) {
            a[l+ 0] = (int8_t)((q3[l+0] >> 0) & 3) - (hm[l] & 0x01 ? 0 : 4);
            a[l+ 8] = (int8_t)((q3[l+8] >> 0) & 3) - (hm[l] & 0x02 ? 0 : 4);
            a[l+16] = (int8_t)((q3[l+0] >> 2) & 3) - (hm[l] & 0x04 ? 0 : 4);
            a[l+24] = (int8_t)((q3[l+8] >> 2) & 3) - (hm[l] & 0x08 ? 0 : 4);
            a[l+32] = (int8_t)((q3[l+0] >> 4) & 3) - (hm[l] & 0x10 ? 0 : 4);
            a[l+40] = (int8_t)((q3[l+8] >> 4) & 3) - (hm[l] & 0x20 ? 0 : 4);
            a[l+48] = (int8_t)((q3[l+0] >> 6) & 3) - (hm[l] & 0x40 ? 0 : 4);
            a[l+56] = (int8_t)((q3[l+8] >> 6) & 3) - (hm[l] & 0x80 ? 0 : 4);
        }

        scales[0] = (x[i].scales[0] & 0xF) - 8;
        scales[1] = (x[i].scales[0] >>  4) - 8;
        scales[2] = (x[i].scales[1] & 0xF) - 8;
        scales[3] = (x[i].scales[1] >>  4) - 8;

        memset(aux32, 0, 8*sizeof(int32_t));
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] += q8[l] * a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux32[l] += scales[j] * aux16[l];
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}
#endif

#if QK_K == 256
void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);
#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    int8x16x2_t q4bytes;
    int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        const uint32x2_t mins8 = {utmp[1] & kmask1, ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)};
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        //int32x4_t isum = mzero;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const uint8x16x2_t q4bits = vld1q_u8_x2(q4); q4 += 32;

#ifdef __ARM_FEATURE_DOTPROD
            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
#else
            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
            const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi1 += vaddvq_s16(vaddq_s16(p0, p1)) * scales[2*j+0];

            q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
            const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi2 += vaddvq_s16(vaddq_s16(p2, p3)) * scales[2*j+1];

#endif
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);
            sumi = _mm256_add_epi32(sumi, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            sumi = _mm256_add_epi32(sumi, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else


    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = ggml_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#else
void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);

#ifdef __ARM_FEATURE_DOTPROD
    const int32x4_t mzero = vdupq_n_s32(0);
#endif

    float sumf = 0;

    int8x16x2_t q4bytes;
    int8x16x4_t q8bytes;

    float sum_mins = 0.f;

    uint16_t aux16[2];
    const uint8_t * restrict scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const uint16_t * restrict a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        const int32_t summi = scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]);
        sum_mins += y[i].d * (float)x[i].d[1] * summi;

        const float d = y[i].d * (float)x[i].d[0];

        const uint8x16x2_t q4bits = vld1q_u8_x2(q4);

#ifdef __ARM_FEATURE_DOTPROD
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

        const int32x4_t p1 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
        const int32_t sumi1 = vaddvq_s32(p1) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

        const int32x4_t p2 = vdotq_s32(vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[2]), q4bytes.val[1], q8bytes.val[3]);
        const int32_t sumi2 = vaddvq_s32(p2) * scales[1];

#else
        q8bytes = vld1q_s8_x4(q8);
        q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
        q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));
        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi1 = vaddvq_s16(vaddq_s16(p0, p1)) * scales[0];

        q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
        q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));
        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[3])));
        int32_t sumi2 = vaddvq_s16(vaddq_s16(p2, p3)) * scales[1];

#endif
        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf - sum_mins;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    uint16_t aux16[2];
    const uint8_t * scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = ggml_fp16_to_fp32(x[i].d[0]) * y[i].d;
        const float m = ggml_fp16_to_fp32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
        const __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);

        const __m256i p32l = _mm256_madd_epi16(_mm256_set1_epi16(scales[0]), p16l);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32l), acc);

        const __m256i p32h = _mm256_madd_epi16(_mm256_set1_epi16(scales[1]), p16h);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(p32h), acc);

    }

    *s = hsum_float_8(acc) - summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    uint16_t aux16[2];
    const uint8_t * scales = (const uint8_t *)aux16;

    for (int i = 0; i < nb; ++i) {

        const float d = ggml_fp16_to_fp32(x[i].d[0]) * y[i].d;
        const float m = ggml_fp16_to_fp32(x[i].d[1]) * y[i].d;
        const __m256 vd = _mm256_set1_ps(d);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;

        summs += m * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bits_0 = _mm256_extractf128_si256(q4bits, 0);
        const __m128i q4bits_1 = _mm256_extractf128_si256(q4bits, 1);
        const __m128i q4_0 = _mm_and_si128(q4bits_0, m4);
        const __m128i q4_1 = _mm_and_si128(q4bits_1, m4);
        const __m128i q4_2 = _mm_and_si128(_mm_srli_epi16(q4bits_0, 4), m4);
        const __m128i q4_3 = _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
        const __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
        const __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
        const __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

        const __m128i p32_0 = _mm_madd_epi16(_mm_set1_epi16(scales[0]), p16_0);
        const __m128i p32_1 = _mm_madd_epi16(_mm_set1_epi16(scales[0]), p16_1);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(MM256_SET_M128I(p32_1, p32_0))), acc);

        const __m128i p32_2 = _mm_madd_epi16(_mm_set1_epi16(scales[1]), p16_2);
        const __m128i p32_3 = _mm_madd_epi16(_mm_set1_epi16(scales[1]), p16_3);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(MM256_SET_M128I(p32_3, p32_2))), acc);

    }

    *s = hsum_float_8(acc) - summs;

#else

    uint8_t aux8[QK_K];
    int16_t aux16[16];
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    uint16_t s16[2];
    const uint8_t * restrict scales = (const uint8_t *)s16;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        uint8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) a[l+ 0] = q4[l] & 0xF;
        for (int l = 0; l < 32; ++l) a[l+32] = q4[l]  >> 4;

        const uint16_t * restrict b = (const uint16_t *)x[i].scales;
        s16[0] = b[0] & 0x0f0f;
        s16[1] = (b[0] >> 4) & 0x0f0f;

        sumf -= y[i].d * ggml_fp16_to_fp32(x[i].d[1]) * (scales[2] * (y[i].bsums[0] + y[i].bsums[1]) + scales[3] * (y[i].bsums[2] + y[i].bsums[3]));

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d[0]);

        for (int j = 0; j < QK_K/32; ++j) {
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            q8 += 16; a += 16;
            for (int l = 0; l < 16; ++l) aux16[l] += q8[l] * a[l];
            q8 += 16; a += 16;
            const float dl = d * scales[j];
            for (int l = 0; l < 8; ++l) sums[l] += dl * (aux16[l] + aux16[l+8]);
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif

#if QK_K == 256
void ggml_vec_dot_q5_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];


#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);

    int8x16x4_t q5bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        int32_t sumi_mins = vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        uint8x16x2_t qhbits = vld1q_u8_x2(qh);

        uint8x16x4_t q5h;

        int32_t sumi = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const uint8x16x2_t q5bits = vld1q_u8_x2(q5); q5 += 32;
            const int8x16x4_t q8bytes = vld1q_s8_x4(q8); q8 += 64;

            q5h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q5h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            q5h.val[2] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[0]), 3);
            q5h.val[3] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[1]), 3);
            qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 2);
            qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 2);

            q5bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0], m4b), q5h.val[0]));
            q5bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1], m4b), q5h.val[1]));
            q5bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0], 4), q5h.val[2]));
            q5bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1], 4), q5h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

            sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]), q5bytes.val[1], q8bytes.val[1])) * *scales++;
            sumi += vaddvq_s32(vdotq_s32(vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]), q5bytes.val[3], q8bytes.val[3])) * *scales++;
#else

            const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                           vmull_s8(vget_high_s8(q5bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                           vmull_s8(vget_high_s8(q5bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            sumi += vaddvq_s16(vaddq_s16(p0, p1)) * *scales++;

            const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                           vmull_s8(vget_high_s8(q5bytes.val[2]), vget_high_s8(q8bytes.val[2])));
            const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                           vmull_s8(vget_high_s8(q5bytes.val[3]), vget_high_s8(q8bytes.val[3])));
            sumi += vaddvq_s16(vaddq_s16(p2, p3)) * *scales++;
#endif
        }

        sumf += d * sumi - dmin * sumi_mins;

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

   for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

#if QK_K == 256
        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
#else
        // TODO
        const float d = 0, dmin = 0;
#endif

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m128i mone  = _mm_set1_epi8(1);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int bit = 0;

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            __m128i q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            __m128i q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc) + summs;

#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = ggml_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#else

void ggml_vec_dot_q5_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);
    const uint8x16_t mh = vdupq_n_u8(16);

    int8x16x4_t q5bytes;
    uint8x16x4_t q5h;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * (float)x[i].d;
        const int8_t * sc = x[i].scales;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const uint8x8_t qhbits = vld1_u8(qh);

        const uint8x16x2_t q5bits = vld1q_u8_x2(q5);
        const int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        const uint8x16_t htmp = vcombine_u8(qhbits, vshr_n_u8(qhbits, 1));
        q5h.val[0] = vbicq_u8(mh, vshlq_n_u8(htmp, 4));
        q5h.val[1] = vbicq_u8(mh, vshlq_n_u8(htmp, 2));
        q5h.val[2] = vbicq_u8(mh, htmp);
        q5h.val[3] = vbicq_u8(mh, vshrq_n_u8(htmp, 2));

        q5bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[0], m4b)), vreinterpretq_s8_u8(q5h.val[0]));
        q5bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q5bits.val[1], m4b)), vreinterpretq_s8_u8(q5h.val[1]));
        q5bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[0], 4)), vreinterpretq_s8_u8(q5h.val[2]));
        q5bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q5bits.val[1], 4)), vreinterpretq_s8_u8(q5h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

        int32_t sumi1 = sc[0] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]));
        int32_t sumi2 = sc[1] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[1], q8bytes.val[1]));
        int32_t sumi3 = sc[2] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]));
        int32_t sumi4 = sc[3] * vaddvq_s32(vdotq_s32(mzero, q5bytes.val[3], q8bytes.val[3]));

        sumf += d * (sumi1 + sumi2 + sumi3 + sumi4);

#else

        const int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                       vmull_s8(vget_high_s8(q5bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        const int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                       vmull_s8(vget_high_s8(q5bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        int32_t sumi = sc[0] * vaddvq_s16(p0) + sc[1] * vaddvq_s16(p1);

        const int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                       vmull_s8(vget_high_s8(q5bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        const int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q5bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                       vmull_s8(vget_high_s8(q5bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        sumi += sc[2] * vaddvq_s16(p2) + sc[3] * vaddvq_s16(p3);

        sumf += d*sumi;
#endif

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);

        const __m256i scale_l = MM256_SET_M128I(_mm_set1_epi16(x[i].scales[1]), _mm_set1_epi16(x[i].scales[0]));
        const __m256i scale_h = MM256_SET_M128I(_mm_set1_epi16(x[i].scales[3]), _mm_set1_epi16(x[i].scales[2]));

        int64_t aux64;
        memcpy(&aux64, x[i].qh, 8);
        const __m128i haux128 = _mm_set_epi64x(aux64 >> 1, aux64);
        const __m256i haux256 = MM256_SET_M128I(_mm_srli_epi16(haux128, 2), haux128);

        const __m256i q5h_0 = _mm256_slli_epi16(_mm256_andnot_si256(haux256, mone), 4);
        const __m256i q5h_1 = _mm256_slli_epi16(_mm256_andnot_si256(_mm256_srli_epi16(haux256, 4), mone), 4);

        const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
        const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m256i p16_0 = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5l_0, q8_0));
        const __m256i p16_1 = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5l_1, q8_1));
        const __m256i s16_0 = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q5h_0, q8_0));
        const __m256i s16_1 = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q5h_1, q8_1));

        const __m256i dot = _mm256_sub_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(s16_0, s16_1));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mone  = _mm_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);

        const __m128i scale_0 = _mm_set1_epi16(x[i].scales[0]);
        const __m128i scale_1 = _mm_set1_epi16(x[i].scales[1]);
        const __m128i scale_2 = _mm_set1_epi16(x[i].scales[2]);
        const __m128i scale_3 = _mm_set1_epi16(x[i].scales[3]);

        int64_t aux64;
        memcpy(&aux64, x[i].qh, 8);
        const __m128i haux128_0 = _mm_set_epi64x(aux64 >> 1, aux64);
        const __m128i haux128_1 = _mm_srli_epi16(haux128_0, 2);

        const __m128i q5h_0 = _mm_slli_epi16(_mm_andnot_si128(haux128_0, mone), 4);
        const __m128i q5h_1 = _mm_slli_epi16(_mm_andnot_si128(haux128_1, mone), 4);
        const __m128i q5h_2 = _mm_slli_epi16(_mm_andnot_si128(_mm_srli_epi16(haux128_0, 4), mone), 4);
        const __m128i q5h_3 = _mm_slli_epi16(_mm_andnot_si128(_mm_srli_epi16(haux128_1, 4), mone), 4);

        const __m128i q5l_0 = _mm_and_si128(_mm256_extractf128_si256(q5bits, 0), m4);
        const __m128i q5l_1 = _mm_and_si128(_mm256_extractf128_si256(q5bits, 1), m4);
        const __m128i q5l_2 = _mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q5bits, 0), 4), m4);
        const __m128i q5l_3 = _mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q5bits, 1), 4), m4);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        const __m128i p16_0 = _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5l_0, _mm256_extractf128_si256(q8_0, 0)));
        const __m128i p16_1 = _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5l_1, _mm256_extractf128_si256(q8_0, 1)));
        const __m128i p16_2 = _mm_madd_epi16(scale_2, _mm_maddubs_epi16(q5l_2, _mm256_extractf128_si256(q8_1, 0)));
        const __m128i p16_3 = _mm_madd_epi16(scale_3, _mm_maddubs_epi16(q5l_3, _mm256_extractf128_si256(q8_1, 1)));
        const __m128i s16_0 = _mm_madd_epi16(scale_0, _mm_maddubs_epi16(q5h_0, _mm256_extractf128_si256(q8_0, 0)));
        const __m128i s16_1 = _mm_madd_epi16(scale_1, _mm_maddubs_epi16(q5h_1, _mm256_extractf128_si256(q8_0, 1)));
        const __m128i s16_2 = _mm_madd_epi16(scale_2, _mm_maddubs_epi16(q5h_2, _mm256_extractf128_si256(q8_1, 0)));
        const __m128i s16_3 = _mm_madd_epi16(scale_3, _mm_maddubs_epi16(q5h_3, _mm256_extractf128_si256(q8_1, 1)));

        const __m128i dot_0 = _mm_sub_epi32(_mm_add_epi32(p16_0, p16_2), _mm_add_epi32(s16_0, s16_2));
        const __m128i dot_1 = _mm_sub_epi32(_mm_add_epi32(p16_1, p16_3), _mm_add_epi32(s16_1, s16_3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(dot_1, dot_0))), acc);

    }

    *s = hsum_float_8(acc);

#else

    int8_t aux8[QK_K];
    int16_t aux16[16];
    float   sums [8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        int8_t * restrict a = aux8;
        for (int l = 0; l < 32; ++l) {
            a[l+ 0] = q4[l] & 0xF;
            a[l+32] = q4[l]  >> 4;
        }
        for (int is = 0; is < 8; ++is) {
            uint8_t m = 1 << is;
            for (int l = 0; l < 8; ++l) a[8*is + l] -= (hm[l] & m ? 0 : 16);
        }

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const int8_t * restrict sc = x[i].scales;

        for (int j = 0; j < QK_K/16; ++j) {
            const float dl = d * sc[j];
            for (int l = 0; l < 16; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l <  8; ++l) sums[l] += dl * (aux16[l] + aux16[8+l]);
            q8 += 16; a += 16;
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
#endif


#if QK_K == 256
void ggml_vec_dot_q6_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    //const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    int8x16x4_t q6bytes;
    uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int8x16_t scales = vld1q_s8(scale);
        const int16x8x2_t q6scales = {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))};

        const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[0]), vget_low_s16 (q6scales.val[0])),
                                                   vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                         vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[1]), vget_low_s16 (q6scales.val[1])),
                                                   vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        int32_t isum_mins = vaddvq_s32(prod);

        int32_t isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            uint8x16x2_t qhbits = vld1q_u8_x2(qh); qh += 32;
            uint8x16x4_t q6bits = vld1q_u8_x4(q6); q6 += 64;
            int8x16x4_t q8bytes = vld1q_s8_x4(q8); q8 += 64;

            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

            isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;

#else

            int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                     vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                     vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
            scale += 2;

            int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                     vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
            int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                     vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
            isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
            scale += 2;
#endif

            q8bytes = vld1q_s8_x4(q8); q8 += 64;

            shifted = vshrq_n_u8(qhbits.val[0], 4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[0], 6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

#if defined(__ARM_FEATURE_DOTPROD)

            isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;

            //for (int l = 0; l < 4; ++l) {
            //    const int32x4_t p = vdotq_s32(vzero, q6bytes.val[l], q8bytes.val[l]);
            //    isum += vaddvq_s32(p) * *scale++;
            //}
#else
            p0 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                    vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
            p1 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                    vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
            isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];
            scale += 2;

            p2 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                    vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
            p3 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                    vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
            isum += vaddvq_s16(p2) * scale[0] + vaddvq_s16(p3) * scale[1];
            scale += 2;
#endif

        }
        //sum += isum * d_all * y[i].d;
        sum += d_all * y[i].d * (isum - 32 * isum_mins);

    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m32s = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 2), m3), 4);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 2), m3), 4);
            const __m128i q4h_4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 4), m3), 4);
            const __m128i q4h_5 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 4), m3), 4);
            const __m128i q4h_6 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_0, 6), m3), 4);
            const __m128i q4h_7 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH_1, 6), m3), 4);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m4), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m4), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m4), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m4), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m4), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m4), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m4), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m4), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i q8s_0 = _mm_maddubs_epi16(m32s, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(m32s, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(m32s, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(m32s, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(m32s, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(m32s, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(m32s, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(m32s, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_2 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);
            const __m128i scale_3 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi8(shuffle, m2);

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_2, scale_2)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_3, scale_3)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));

        }

        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#else

void ggml_vec_dot_q6_K_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    int8x16x4_t q6bytes;
    uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = (float)x[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int32_t isum = 0;

        uint8x16_t   qhbits = vld1q_u8(qh);
        uint8x16x2_t q6bits = vld1q_u8_x2(q6);
        int8x16x4_t q8bytes = vld1q_s8_x4(q8);

        q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits), 4);
        uint8x16_t shifted = vshrq_n_u8(qhbits, 2);
        q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 4);
        q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
        shifted = vshrq_n_u8(qhbits, 6);
        q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

        q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
        q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
        q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[2])), m32s);
        q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[3])), m32s);

#if defined(__ARM_FEATURE_DOTPROD)

        isum += vaddvq_s32(vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                vaddvq_s32(vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
#else

        int16x8_t p0 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[0]), vget_low_s8 (q8bytes.val[0])),
                                 vmull_s8(vget_high_s8(q6bytes.val[0]), vget_high_s8(q8bytes.val[0])));
        int16x8_t p1 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[1]), vget_low_s8 (q8bytes.val[1])),
                                 vmull_s8(vget_high_s8(q6bytes.val[1]), vget_high_s8(q8bytes.val[1])));
        isum += vaddvq_s16(p0) * scale[0] + vaddvq_s16(p1) * scale[1];

        int16x8_t p2 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[2]), vget_low_s8 (q8bytes.val[2])),
                                 vmull_s8(vget_high_s8(q6bytes.val[2]), vget_high_s8(q8bytes.val[2])));
        int16x8_t p3 = vaddq_s16(vmull_s8(vget_low_s8 (q6bytes.val[3]), vget_low_s8 (q8bytes.val[3])),
                                 vmull_s8(vget_high_s8(q6bytes.val[3]), vget_high_s8(q8bytes.val[3])));
        isum += vaddvq_s16(p2) * scale[2] + vaddvq_s16(p3) * scale[3];
#endif

        sum += isum * d_all * y[i].d;

    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m256i sumi = _mm256_setzero_si256();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 2), q4bitsH), m2), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(q4bitsH, 6), _mm_srli_epi16(q4bitsH, 4)), m2), 4);

        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_1);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
        __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);

        __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);

        p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm256_sub_epi16(p16_1, q8s_1);

        p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(3);
    const __m128i m32s = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m64 scales_1 = _mm_set1_pi8(x[i].scales[0]);
        const __m64 scales_2 = _mm_set1_pi8(x[i].scales[1]);
        const __m64 scales_3 = _mm_set1_pi8(x[i].scales[2]);
        const __m64 scales_4 = _mm_set1_pi8(x[i].scales[3]);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        const __m128i scale_0 = _mm_set_epi64(scales_2, scales_1);
        const __m128i scale_1 = _mm_set_epi64(scales_4, scales_3);

        const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
        const __m128i q4bitsH = _mm_loadu_si128((const __m128i*)qh);

        const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH, m2), 4);
        const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 2), m2), 4);
        const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 4), m2), 4);
        const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(q4bitsH, 6), m2), 4);

        const __m128i q4_0 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 0), m4), q4h_0);
        const __m128i q4_1 = _mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(q4bits1, 1), m4), q4h_1);
        const __m128i q4_2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 0), 4), m4), q4h_2);
        const __m128i q4_3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(_mm256_extractf128_si256(q4bits1, 1), 4), m4), q4h_3);

        const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8+ 0));
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8+32));

        __m128i q8s_0 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 0));
        __m128i q8s_1 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_0, 1));
        __m128i q8s_2 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 0));
        __m128i q8s_3 = _mm_maddubs_epi16(m32s, _mm256_extractf128_si256(q8_1, 1));

        __m128i p16_0 = _mm_maddubs_epi16(q4_0, _mm256_extractf128_si256(q8_0, 0));
        __m128i p16_1 = _mm_maddubs_epi16(q4_1, _mm256_extractf128_si256(q8_0, 1));
        __m128i p16_2 = _mm_maddubs_epi16(q4_2, _mm256_extractf128_si256(q8_1, 0));
        __m128i p16_3 = _mm_maddubs_epi16(q4_3, _mm256_extractf128_si256(q8_1, 1));

        p16_0 = _mm_sub_epi16(p16_0, q8s_0);
        p16_1 = _mm_sub_epi16(p16_1, q8s_1);
        p16_2 = _mm_sub_epi16(p16_2, q8s_2);
        p16_3 = _mm_sub_epi16(p16_3, q8s_3);

        p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
        p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_0, scale_0)), p16_1);
        p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
        p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_unpackhi_epi64(scale_1, scale_1)), p16_3);

        sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
        sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi_1, sumi_0))), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int l = 0; l < 16; ++l) {
            a[l+ 0] = (int8_t)((q4[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            a[l+16] = (int8_t)((q4[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            a[l+32] = (int8_t)((q4[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            a[l+48] = (int8_t)((q4[l+16] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ggml_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#endif
