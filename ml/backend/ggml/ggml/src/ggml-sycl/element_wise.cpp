#include "common.hpp"
#include "ggml.h"
#include "element_wise.hpp"

static void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset, const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

template<typename T>
static void gelu(const T * x, T * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const T GELU_COEF_A    = static_cast<T>(0.044715f);
    const T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = static_cast<T>(0.5f) * xi *
             (static_cast<T>(1.0f) +
              sycl::tanh(SQRT_2_OVER_PI * xi * (static_cast<T>(1.0f) + GELU_COEF_A * xi * xi)));
}

template<typename T>
static void silu(const T * x, T * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (static_cast<T>(1.0f) + sycl::native::exp(-x[i]));
}

template<typename T>
static void gelu_quick(const T *x, T *dst, int k,
                           const sycl::nd_item<3> &item_ct1) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(GELU_QUICK_COEF * x[i])));
}

template<typename T>
static void tanh(const T *x, T *dst, int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::tanh((x[i]));
}

template<typename T>
static void relu(const T * x, T * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((x[i]), static_cast<T>(0));
}

template<typename T>
static void sigmoid(const T * x, T * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (static_cast<T>(1.0f) + sycl::native::exp(-x[i]));
}

template<typename T>
static void sqrt(const T * x, T * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::sqrt(x[i]);
}

template<typename T>
static void sin(const T * x, T * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::sin(x[i]);
}

template<typename T>
static void cos(const T * x, T * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::cos(x[i]);
}

template<typename T>
static void hardsigmoid(const T * x, T * dst, const int k,
                            const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x[i] + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
}

template<typename T>
static void hardswish(const T * x, T * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x[i] + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
}

template<typename T>
static void exp(const T * x, T * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::exp(x[i]);
}

template<typename T>
static void log(const T * x, T * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    T xi = x[i];
    if (xi <= 0) {
        dst[i] = neg_infinity<T>();
    } else {
        dst[i] = sycl::log(xi);
    }
}

template<typename T>
static void neg(const T * x, T * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = -x[i];
}

template<typename T>
static void step(const T * x, T * dst, const int k,
                          const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] > static_cast<T>(0.0f);
}

template<typename T>
static void leaky_relu(const T *x, T *dst, const int k, const float negative_slope,
                           const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((x[i]), static_cast<T>(0)) +
             sycl::fmin((x[i]), static_cast<T>(0.0f)) * negative_slope;
}

template<typename T>
static void sqr(const T * x, T * dst, const int k,
                    const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

template<typename  T>
static void upscale(const T  *x, T *dst, const int nb00, const int nb01,
                        const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int ne13, const float sf0, const float sf1,
                        const float sf2, const float sf3, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_local_id(0) +
               item_ct1.get_group(0) * item_ct1.get_local_range(0);
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }
    // operation
    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *(const T *)((const char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

template <typename T>
static void pad(const T  *x, T *dst, const int ne0, const int ne00, const int ne01, const int ne02,
                    const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00 && item_ct1.get_group(1) < (size_t) ne01 && item_ct1.get_group(0) < (size_t) ne02) {
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                         item_ct1.get_group(0) * ne00 * ne01;
            dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = static_cast<T>(0.0f);
    }
}


template<typename T>
static void clamp(const T * x, T * dst, const float min, const float max, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
}

static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, queue_ptr stream) {
    int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

template<typename T>
static void gelu_sycl(const T *x, T *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu(x, dst, k, item_ct1);
        });
}

template<typename T>
static void silu_sycl(const T *x, T *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_SILU_BLOCK_SIZE - 1) / SYCL_SILU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            silu(x, dst, k, item_ct1);
        });
}

template<typename T>
static void gelu_quick_sycl(const T *x, T *dst, const int k,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_quick(x, dst, k, item_ct1);
        });
}

template<typename T>
static void tanh_sycl(const T *x, T *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_TANH_BLOCK_SIZE - 1) / SYCL_TANH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            tanh(x, dst, k, item_ct1);
        });
}

template<typename T>
static void relu_sycl(const T *x, T *dst, const int k,
                          queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            relu(x, dst, k, item_ct1);
        });
}

template<typename T>
static void hardsigmoid_sycl(const T *x, T *dst, const int k,
                                 queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSIGMOID_BLOCK_SIZE - 1) / SYCL_HARDSIGMOID_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSIGMOID_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardsigmoid(x, dst, k, item_ct1);
        });
}

template<typename T>
static void hardswish_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_HARDSWISH_BLOCK_SIZE - 1) / SYCL_HARDSWISH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_HARDSWISH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            hardswish(x, dst, k, item_ct1);
        });
}

template<typename T>
static void exp_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_EXP_BLOCK_SIZE - 1) / SYCL_EXP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            exp(x, dst, k, item_ct1);
        });
}

template<typename T>
static void log_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_EXP_BLOCK_SIZE - 1) / SYCL_EXP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_EXP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            log(x, dst, k, item_ct1);
        });
}

template<typename T>
static void neg_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            neg(x, dst, k, item_ct1);
        });
}

template<typename T>
static void step_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_NEG_BLOCK_SIZE - 1) / SYCL_NEG_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_NEG_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            step(x, dst, k, item_ct1);
        });
}

template<typename T>
static void sigmoid_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIGMOID_BLOCK_SIZE - 1) / SYCL_SIGMOID_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIGMOID_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIGMOID_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sigmoid(x, dst, k, item_ct1);
        });
}

template<typename T>
static void sqrt_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQRT_BLOCK_SIZE - 1) / SYCL_SQRT_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQRT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQRT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqrt(x, dst, k, item_ct1);
        });
}

template<typename T>
static void sin_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIN_BLOCK_SIZE - 1) / SYCL_SIN_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sin(x, dst, k, item_ct1);
        });
}

template<typename T>
static void cos_sycl(const T *x, T *dst, const int k,
                               queue_ptr stream) {
    const int num_blocks = (k + SYCL_SIN_BLOCK_SIZE - 1) / SYCL_SIN_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SIN_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            cos(x, dst, k, item_ct1);
        });
}

template<typename T>
static void leaky_relu_sycl(const T *x, T *dst, const int k,
                                const float negative_slope,
                                queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            leaky_relu(x, dst, k, negative_slope, item_ct1);
        });
}

template<typename T>
static void sqr_sycl(const T *x, T *dst, const int k,
                         queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQR_BLOCK_SIZE - 1) / SYCL_SQR_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqr(x, dst, k, item_ct1);
        });
}

template<typename T>
static void upscale_sycl(const T *x, T *dst, const int nb00, const int nb01,
                             const int nb02, const int nb03, const int ne10, const int ne11,
                             const int ne12, const int ne13, const float sf0, const float sf1,
                             const float sf2, const float sf3, queue_ptr stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;
    sycl::range<1> gridDim(num_blocks * SYCL_UPSCALE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(gridDim, sycl::range<1>(SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            upscale(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, item_ct1);
        });
}

template<typename T>
static void pad_sycl(const T *x, T *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad(x, dst, ne0, ne00, ne01, ne02, item_ct1);
        });
}

template<typename T>
static void clamp_sycl(const T *x, T *dst, const float min,
                           const float max, const int k,
                           queue_ptr stream) {
    const int num_blocks = (k + SYCL_CLAMP_BLOCK_SIZE - 1) / SYCL_CLAMP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            clamp(x, dst, min, max, k, item_ct1);
        });
}

inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                silu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                silu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                gelu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                gelu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                gelu_quick_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                gelu_quick_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                tanh_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                tanh_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                relu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                relu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                hardsigmoid_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                hardsigmoid_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                hardswish_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                hardswish_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                exp_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                exp_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                log_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                log_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                sigmoid_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                sigmoid_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                sqrt_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                sqrt_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                sin_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                sin_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                cos_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                cos_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                step_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                step_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                neg_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                neg_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif

    GGML_ASSERT(dst->src[0]->type == dst->type);
    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                leaky_relu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), negative_slope, main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                leaky_relu_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), negative_slope, main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
 #if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                sqr_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                sqr_sycl(data_pts.src, data_pts.dst, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float sf0 = (float) dst->ne[0] / dst->src[0]->ne[0];
    const float sf1 = (float) dst->ne[1] / dst->src[0]->ne[1];
    const float sf2 = (float) dst->ne[2] / dst->src[0]->ne[2];
    const float sf3 = (float) dst->ne[3] / dst->src[0]->ne[3];
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                upscale_sycl(data_pts.src, data_pts.dst, dst->src[0]->nb[0], dst->src[0]->nb[1], dst->src[0]->nb[2],
                        dst->src[0]->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3,
                        main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                upscale_sycl(data_pts.src, data_pts.dst, dst->src[0]->nb[0], dst->src[0]->nb[1], dst->src[0]->nb[2],
                        dst->src[0]->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3,
                        main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    GGML_ASSERT(dst->src[0]->ne[3] == 1 && dst->ne[3] == 1);  // just 3D tensors
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                pad_sycl(data_pts.src, data_pts.dst, dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->ne[0],
                        dst->ne[1], dst->ne[2], main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                pad_sycl(data_pts.src, data_pts.dst, dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->ne[0],
                        dst->ne[1], dst->ne[2], main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if defined(GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    switch (dst->type) {
#if defined(GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                clamp_sycl(data_pts.src, data_pts.dst, min, max, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                clamp_sycl(data_pts.src, data_pts.dst, min, max, ggml_nelements(dst->src[0]), main_stream);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
            break;
    }
}

inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    const float * src1_dd = static_cast<const float*>(dst->src[1]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_sycl(src0_dd, src1_dd, dst_dd, ggml_nelements(dst), dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], nb1, nb2, offset, main_stream);
}


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_sqrt(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_sin(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_cos(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_acc(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_gelu(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_silu(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_gelu_quick(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_tanh(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_relu(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_sigmoid(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_hardsigmoid(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_hardswish(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}


void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_exp(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_log(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_neg(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_step(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_leaky_relu(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_sqr(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_upscale(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_pad(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s: DST Tensor type: %s\n", __func__, ggml_type_name(dst->type));
    ggml_sycl_op_clamp(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

