#include "quantize.cuh"
#include <cstdint>

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    const float xi = ix0 < kx ? x[ix1*kx + ix0] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <bool need_sum>
static __global__ void quantize_mmq_q8_1(
    const float * __restrict__ x, void * __restrict__ vy, const int64_t kx0, const int64_t kx1, const int64_t kx0_padded) {

    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = kx1*blockIdx.z + blockIdx.y;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*(gridDim.y*gridDim.x*blockDim.x/(4*QK8_1)); // first block of channel
    const int64_t ib  = ib0 + (ix0 / (4*QK8_1))*kx1 + blockIdx.y;              // block index in channel
    const int64_t iqs = ix0 % (4*QK8_1);                                       // quant index in block

    const float xi = ix0 < kx0 ? x[ix1*kx0 + ix0] : 0.0f;
    float amax = fabsf(xi);

    amax = warp_reduce_max(amax);

    float sum;
    if (need_sum) {
        sum = warp_reduce_sum(xi);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs % QK8_1 != 0) {
        return;
    }

    if (need_sum) {
        y[ib].ds[iqs/QK8_1] = make_half2(d, sum);
    } else {
        ((float *) y[ib].ds)[iqs/QK8_1] = d;
    }
}

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1*channels, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx0_padded);

    GGML_UNUSED(type_x);
}

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    GGML_ASSERT(kx0_padded % (4*QK8_1) == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1, channels);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    if (mmq_need_sum(type_x)) {
        quantize_mmq_q8_1<true><<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
    } else {
        quantize_mmq_q8_1<false><<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx1, kx0_padded);
    }
}
