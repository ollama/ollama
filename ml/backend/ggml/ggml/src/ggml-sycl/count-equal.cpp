#include "count-equal.hpp"

#include <cstdint>

template <typename T>
static void count_equal(const T *__restrict__ x, const T *__restrict__ y,
                        int64_t *__restrict__ dst, const int64_t dk,
                        const int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    const int64_t i1 = sycl::min(i0 + dk, k);

    int nequal = 0;

    for (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        const T xi = x[i];
        const T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum(nequal);

    if (item_ct1.get_local_id(2) != 0) {
        return;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (int *)dst, nequal);
}

void ggml_sycl_count_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    const int id       = get_current_device_id();
    const int nsm = ggml_sycl_info().devices[id].nsm;

    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    const int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_COUNT_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    const dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    const dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_COUNT_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_COUNT_EQUAL_CHUNK_SIZE),
        1, 1);

    switch (src0->type) {
    case GGML_TYPE_I32: {
        const int *src0_d = (const int *)src0->data;
        const int *src1_d = (const int *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                count_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } break;
    default:
        GGML_ASSERT(false);
        break;
    }
}
