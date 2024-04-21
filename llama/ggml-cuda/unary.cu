#include "unary.cuh"

static __global__ void gelu_f32(const float * x, float * dst, const int k) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f*xi*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void gelu_quick_f32(const float * x, float * dst, int k) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x[i])));
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void tanh_f32(const float * x, float * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static __global__ void sigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + expf(-x[i]));
}

static __global__ void hardsigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void hardswish_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void leaky_relu_f32(const float * x, float * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) + fminf(x[i], 0.0f) * negative_slope;
}

static __global__ void sqr_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static void gelu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void gelu_quick_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_quick_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void tanh_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    tanh_f32<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    sigmoid_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardsigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    hardsigmoid_f32<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardswish_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    hardswish_f32<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void leaky_relu_f32_cuda(const float * x, float * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

static void sqr_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    sqr_f32<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    tanh_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), negative_slope, stream);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}
