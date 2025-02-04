#include "common.cuh"

struct mma_int_A_I16K4 {
    static constexpr int I  = 16;
    static constexpr int K  = 4;
    static constexpr int ne = 2;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load(const int * __restrict__ xs0, const int & stride) {
#if defined(INT8_MMA_AVAILABLE)
        const int * xs = xs0 + (threadIdx.x%I)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(x[0]), "+r"(x[1])
            : "l"(xs));
#else
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
#endif // defined(INT8_MMA_AVAILABLE)
    }
};

struct mma_int_A_I16K8 {
    static constexpr int I  = 16;
    static constexpr int K  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l%2) * (I/2) + threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = (l/2) * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load(const int * __restrict__ xs0, const int & stride) {
#if defined(INT8_MMA_AVAILABLE)
        const int * xs = xs0 + (threadIdx.x%I)*stride + (threadIdx.x/I)*(K/2);
        asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "l"(xs));
#else
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_i(l)*stride + get_k(l)];
        }
#endif // defined(INT8_MMA_AVAILABLE)
    }

    __device__ __forceinline__ void load_low(const int * __restrict__ xs0, const int & stride) {
        ((mma_int_A_I16K4 *) x)[0].load(xs0, stride);
    }
};

struct mma_int_B_J8K4 {
    static constexpr int J  = 8;
    static constexpr int K  = 4;
    static constexpr int ne = 1;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int /* l */) {
        const int ret = threadIdx.x % K;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load(const int * __restrict__ xs0, const int & stride) {
#if defined(INT8_MMA_AVAILABLE) && false // Loading as 4 byte values is faster
        const int * xs = xs0 + (threadIdx.x%J)*stride;
        asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];"
            : "+r"(x[0])
            : "l"(xs));
#else
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
#endif // defined(INT8_MMA_AVAILABLE)
    }
};

struct mma_int_B_J8K8 {
    static constexpr int J  = 8;
    static constexpr int K  = 8;
    static constexpr int ne = 2;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_j(const int /* l */) {
        const int ret = threadIdx.x / (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    static __device__ __forceinline__ int get_k(const int l) {
        const int ret = l * (K/2) + threadIdx.x % (K/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  K);
        return ret;
    }

    __device__ __forceinline__ void load(const int * __restrict__ xs0, const int & stride) {
#if defined(INT8_MMA_AVAILABLE) && false // Loading as 4 byte values is faster
        const int * xs = xs0 + (threadIdx.x%J)*stride + ((threadIdx.x/J)*(K/2)) % K;
        asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "+r"(x[0]), "+r"(x[1])
            : "l"(xs));
#else
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            x[l] = xs0[get_j(l)*stride + get_k(l)];
        }
#endif // defined(INT8_MMA_AVAILABLE)
    }
};

struct mma_int_C_I16J8 {
    static constexpr int I  = 16;
    static constexpr int J  = 8;
    static constexpr int ne = 4;

    int x[ne] = {0};

    static __device__ __forceinline__ int get_i(const int l) {
        const int ret = (l/2) * (I/2) + threadIdx.x / (J/2);
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  I);
        return ret;
    }

    static __device__ __forceinline__ int get_j(const int l) {
        const int ret = 2 * (threadIdx.x % (J/2)) + l%2;
        GGML_CUDA_ASSUME(ret >= 0);
        GGML_CUDA_ASSUME(ret <  J);
        return ret;
    }

    __device__ __forceinline__ void mma_K4(const mma_int_A_I16K4 & mma_A, const mma_int_B_J8K4 & mma_B) {
#ifdef INT8_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
    }

    __device__ __forceinline__ void mma_K8(const mma_int_A_I16K8 & mma_A, const mma_int_B_J8K8 & mma_B) {
#ifdef INT8_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[0]), "r"(mma_A.x[1]), "r"(mma_A.x[2]), "r"(mma_A.x[3]), "r"(mma_B.x[0]), "r"(mma_B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[0]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[1]), "r"(mma_B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[0]), "+r"(x[1])
            : "r"(mma_A.x[2]), "r"(mma_B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(x[2]), "+r"(x[3])
            : "r"(mma_A.x[3]), "r"(mma_B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED(mma_A);
        GGML_UNUSED(mma_B);
        NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
    }
};
