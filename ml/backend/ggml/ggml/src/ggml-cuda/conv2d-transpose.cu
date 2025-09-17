#include <algorithm>

#include "conv2d-transpose.cuh"
#include "ggml.h"

__global__ void conv2d_transpose_kernel(const float * __restrict__ input, const half * __restrict__ kernel,
                                        float * __restrict__ output, const int in_w, const int in_h, const int out_w,
                                        const int out_h, const int kernel_w, const int kernel_h, const int stride,
                                        const int c_in, const int c_out, const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_elements = out_w * out_h * c_out * batches;

    if (global_idx >= total_elements) {
        return;
    }

    const int out_x_idx = global_idx % out_w;
    const int out_y_idx = (global_idx / out_w) % out_h;
    const int c_idx     = (global_idx / (out_w * out_h)) % c_out;
    const int n_idx     = global_idx / (out_w * out_h * c_out);

    float accumulator = 0;
    // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

    for (int c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int in_y = out_y_idx - kh;
            if (in_y < 0 || in_y % stride) continue;
            in_y /= stride;
            if (in_y >= in_h) continue;

            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_x = out_x_idx - kw;
                if (in_x < 0 || in_x % stride) continue;
                in_x /= stride;
                if (in_x >= in_w) continue;

                const int input_idx = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + (in_w) *in_y + in_x;
                const int kernel_idx =
                    (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx + (kernel_w) *kh + kw;

                float input_val = input[input_idx];
                half  kern_val  = kernel[kernel_idx];

                accumulator += input_val * (float) kern_val;
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + (out_w) *out_y_idx + out_x_idx] = accumulator;
}

//input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * input_data  = (const float *) input->data;
    float *       output_data = (float *) dst->data;
    const half * kernel_data = (const half *) kernel->data;

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    cudaStream_t st = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int total  = (output_w * output_h * channels_out * batches);
    const int blocks = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    conv2d_transpose_kernel<<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
        input_data, kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w, kernel_h, stride,
        channels_in, channels_out, batches);
}
