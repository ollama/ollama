#include "ssm_conv.hpp"
#include "common.hpp"

#include <cstdio>

using namespace sycl;

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(global_range, local_range),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int channel = static_cast<int>(idx % d_inner);
                const int token   = static_cast<int>((idx / d_inner) % n_t);
                const int seq     = static_cast<int>(idx / (static_cast<size_t>(d_inner) * static_cast<size_t>(n_t)));

                const float *s = src_data
                    + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
                    + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
                    + static_cast<size_t>(token);

                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

                float sumf = 0.0f;
                for (int i0 = 0; i0 < d_conv; ++i0) {
                    sumf += s[i0] * c[i0];
                }

                const size_t dst_idx =
                    static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
                    static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
                    static_cast<size_t>(channel);

                dst_data[dst_idx] = sumf;
            }
        );
    });
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        float *dst_data       = static_cast<float *>(dst->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq
        );

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}
