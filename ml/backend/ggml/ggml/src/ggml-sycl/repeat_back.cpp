#include "repeat_back.hpp"

#include "common.hpp"

#define GGML_ASSERT_TENSOR_FITS_INT(t) \
    GGML_ASSERT((t)->ne[0] < INT_MAX && (t)->ne[1] < INT_MAX && (t)->ne[2] < INT_MAX && (t)->ne[3] < INT_MAX)

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src0_dd = (const float *) dst->src[0]->data;
    float *       dst_dd  = (float *) dst->data;

    GGML_ASSERT_TENSOR_FITS_INT(dst);
    GGML_ASSERT_TENSOR_FITS_INT(dst->src[0]);

    const int ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int ne00 = dst->src[0]->ne[0], ne01 = dst->src[0]->ne[1], ne02 = dst->src[0]->ne[2],
              ne03 = dst->src[0]->ne[3];

    const int nr0 = ne00 / ne0;
    const int nr1 = ne01 / ne1;
    const int nr2 = ne02 / ne2;
    const int nr3 = ne03 / ne3;

    const int nb0 = dst->src[0]->nb[0];
    const int nb1 = dst->src[0]->nb[1];
    const int nb2 = dst->src[0]->nb[2];
    const int nb3 = dst->src[0]->nb[3];

    const char * base = (const char *) src0_dd;

    const size_t  total      = (size_t) ne0 * ne1 * ne2 * ne3;
    constexpr int BLOCK_SIZE = 256;
    const int     num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float inv_ne0      = 1.0f / ne0;
    const float inv_ne_01    = 1.0f / (ne0 * ne1);
    const float inv_ne_012   = 1.0f / (ne0 * ne1 * ne2);
    const int   repeat_count = nr0 * nr1 * nr2 * nr3;

    queue_ptr stream = ctx.stream();

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            const size_t i = item_ct1.get_global_linear_id();
            if (i >= total) {
                return;
            }

            const int i3 = (int) (i * inv_ne_012);
            const int i2 = (int) (i * inv_ne_01) - i3 * ne2;
            const int i1 = (int) (i * inv_ne0) - (int) (i * inv_ne_01) * ne1;
            const int i0 = i - (int) (i * inv_ne0) * ne0;

            int   j0 = 0, j1 = 0, j2 = 0, j3 = 0;
            float acc = 0.0f;

            for (int j = 0; j < repeat_count; ++j) {
                const float * ptr = (const float *) (base + (i0 + j0 * ne0) * nb0 + (i1 + j1 * ne1) * nb1 +
                    (i2 + j2 * ne2) * nb2 + (i3 + j3 * ne3) * nb3);
                acc += *ptr;

                int carry = (++j0 >= nr0);
                j0 -= carry * nr0;
                carry = (carry && (++j1 >= nr1));
                j1 -= carry * nr1;
                carry = (carry && (++j2 >= nr2));
                j2 -= carry * nr2;
                j3 += carry;
            }
            dst_dd[i] = acc;
        });
}
