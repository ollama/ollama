#include "cpy.hpp"

#include <float.h>

#include "dequantize.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"


static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    float *       dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<float, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f16_f32(const char * cxi, char * cdsti) {
    const sycl::half * xi   = (const sycl::half *) cxi;
    float *            dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t * xi   = (const int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t * xi   = (const int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                        const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                        const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
static void cpy_blck_q_q(const char * cxi, char * cdsti) {
    const T * xi = (const T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


static void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}



template <dequantize_kernel_t dequant, int qk> static void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *) (cdsti);

    for (int j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, int qk>
static void cpy_q_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }


    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_q_f32(const char * cx, char * cdst, const int ne, const int ne00, const int ne01, const int ne02,
                      const int nb00, const int nb01, const int nb02, const int nb03, const int ne10, const int ne11,
                      const int ne12, const int nb10, const int nb11, const int nb12, const int nb13,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    if (i >= ne) {
        return;
    }

    const int i03      = i / (ne00 * ne01 * ne02);
    const int i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    const int i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    const int x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    const int i13        = i / (ne10 * ne11 * ne12);
    const int i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    const int i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    const int dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static void ggml_cpy_f16_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_f32_q8_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q8_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_f32_q4_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q4_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q4_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_0_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_0_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_q5_1_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

static void ggml_cpy_q5_1_f32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ne;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

static void ggml_cpy_f32_iq4_nl_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                     const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                     const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                     const int nb12, const int nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)), [=](sycl::nd_item<3> item_ct1) {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

static void ggml_cpy_f16_f16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i16_i16_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_i32_i32_sycl(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                  const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                  const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                  const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

static void ggml_cpy_q8_0_q8_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_0_q5_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q5_1_q5_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_0_q4_0(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {
    const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


static void ggml_cpy_q4_1_q4_1(const char * cx, char * cdst, const int ne, const int ne00, const int ne01,
                                   const int ne02, const int nb00, const int nb01, const int nb02, const int nb03,
                                   const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                   const int nb12, const int nb13, queue_ptr stream) {

   const int num_blocks = ceil_div(ne, SYCL_CPY_BLOCK_SIZE);
   stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)), [=](sycl::nd_item<3> item_ct1) {
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) try {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_TENSOR_BINARY_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;
    if ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
