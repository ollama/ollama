#include "im2col.cuh"

#define MAX_GRIDDIM_Z 65535

template <typename T>
static  __global__ void im2col_kernel(
        const float * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        int s0, int s1, int p0, int p1, int d0, int d1) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) {
        return;
    }

    const int64_t iic = i / (KH_KW);
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    const int64_t  iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz+=MAX_GRIDDIM_Z) {
        const int64_t  in = iz / OH;
        const int64_t  ioh = iz - in * OH;

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;

        const int64_t offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = 0.0f;
        } else {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = x[offset_src + iih * IW + iiw];
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void im2col_cuda(const float * x, T* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {
    const int64_t IC_KH_KW = IC * KH * KW;
    const int64_t num_blocks = (IC_KH_KW + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
    const int64_t N_OH = N * OH;
    const int64_t KH_KW = KW*KH;
    dim3 block_nums(num_blocks, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    im2col_kernel<<<block_nums, MIN(IC_KH_KW, CUDA_IM2COL_BLOCK_SIZE) , 0, stream>>>(x, dst, IC, IW, IH, OH, OW, KW, KH,
                                                                                     IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                                                                     s0, s1, p0, p1, d0, d1);
}

static void im2col_cuda_f16(const float * x, half * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

static void im2col_cuda_f32(const float * x, float * dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N, int64_t IC_IH_IW, int64_t IH_IW,
    int s0,int s1,int p0,int p1,int d0,int d1, cudaStream_t stream) {

    im2col_cuda<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const int64_t N        = src1->ne[is_2D ? 3 : 2];
    const int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    if(dst->type == GGML_TYPE_F16) {
        im2col_cuda_f16(src1_d, (half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_cuda_f32(src1_d, (float *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}
