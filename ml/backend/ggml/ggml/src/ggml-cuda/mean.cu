#include "mean.cuh"
#include "reduce_rows.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

template <typename T> __global__ void divide_by_count(T * result, size_t count) {
    *result /= static_cast<T>(count);
}

void ggml_cuda_op_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

// Special case for reducing vectors
#ifdef GGML_CUDA_USE_CUB
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus iscapturing;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &iscapturing));
#endif // USE_CUDA_GRAPH
    if ((nrows == 1) &&
#ifdef USE_CUDA_GRAPH
            // CUDA_GRAPHS_DISABLED
            ((ncols > 65536) &&
             ((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
              ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
              ctx.cuda_graph->disable_due_to_failed_graph_capture)) ||
        // CUDA_GRAPHS ENABLED
        ((ncols > 32768) &&
         !((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
           ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
           ctx.cuda_graph->disable_due_to_failed_graph_capture))) {
#else
        (ncols > 65536)) {
#endif // USE_CUDA_GRAPH
        // Single row - use device-wide reduction
        size_t           tmp_size = 0;
        ggml_cuda_pool & pool     = ctx.pool();

        DeviceReduce::Sum(nullptr, tmp_size, src0_d, dst_d, ncols, stream);

        ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
        DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, src0_d, dst_d, ncols, stream);

        // Divide by ncols
        divide_by_count<float><<<1, 1, 0, stream>>>(dst_d, ncols);
        return;
    }
#endif // GGML_CUDA_USE_CUB

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f32</*norm=*/true><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f32</*norm=*/true><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    }
}
