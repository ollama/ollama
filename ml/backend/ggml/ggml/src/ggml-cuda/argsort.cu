#include "argsort.cuh"

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}


template<ggml_sort_order order>
static __global__ void k_argsort_i32_i32(const int32_t * x, int * dst, const int ncols, const int ncols_pad) {
    extern __shared__ int shared_mem[];
    int * indices = shared_mem;

    const int tid = threadIdx.x;
    const int row = blockIdx.y;

    // Initialize all indices, handling the case where threads < ncols_pad
    for (int i = tid; i < ncols_pad; i += blockDim.x) {
        indices[i] = i < ncols ? i : 0; // Use 0 for padding indices
    }
    __syncthreads();

    // Bitonic sort
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for (int i = tid; i < ncols_pad; i += blockDim.x) {
                const int ij = i ^ j;
                if (ij > i) {
                    // Only compare values within the actual data range
                    if (i < ncols && ij < ncols) {
                        if ((i & k) == 0) {
                            if (order == GGML_SORT_ORDER_ASC) {
                                if (x[row * ncols + indices[i]] > x[row * ncols + indices[ij]]) {
                                    int tmp = indices[i];
                                    indices[i] = indices[ij];
                                    indices[ij] = tmp;
                                }
                            } else {
                                if (x[row * ncols + indices[i]] < x[row * ncols + indices[ij]]) {
                                    int tmp = indices[i];
                                    indices[i] = indices[ij];
                                    indices[ij] = tmp;
                                }
                            }
                        } else {
                            if (order == GGML_SORT_ORDER_ASC) {
                                if (x[row * ncols + indices[i]] < x[row * ncols + indices[ij]]) {
                                    int tmp = indices[i];
                                    indices[i] = indices[ij];
                                    indices[ij] = tmp;
                                }
                            } else {
                                if (x[row * ncols + indices[i]] > x[row * ncols + indices[ij]]) {
                                    int tmp = indices[i];
                                    indices[i] = indices[ij];
                                    indices[ij] = tmp;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write sorted indices to output, only threads handling valid data
    for (int i = tid; i < ncols; i += blockDim.x) {
        dst[row * ncols + i] = indices[i];
    }
}

static void argsort_i32_i32_cuda(const int32_t * x, int * dst, const int ncols, const int nrows, ggml_sort_order order, cudaStream_t stream) {
    // Bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    // Ensure thread count doesn't exceed maximum (typically 1024)
    const int max_threads = 1024;  // This is the typical max for most GPUs
    const int threads_per_block = ncols_pad > max_threads ? max_threads : ncols_pad;

    const dim3 block_dims(threads_per_block, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // Check if shared memory size is within limits
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    // Instead of logging an error, use GGML_ASSERT with a descriptive message
    GGML_ASSERT(shared_mem <= max_shared_mem && "argsort: required shared memory exceeds device limit");

    // Launch kernels with the updated thread configuration
    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_i32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_i32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}


void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    if (src0->type == GGML_TYPE_I32) {
        argsort_i32_i32_cuda((const int32_t *)src0_d, (int *)dst_d, ncols, nrows, order, stream);
    } else {
        argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
    }
}
