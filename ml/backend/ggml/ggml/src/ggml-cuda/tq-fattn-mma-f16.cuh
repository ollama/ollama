// Phase D — TQ-aware copy of stock fattn-mma-f16.cuh.
//
// Stage 1: verbatim copy with minimal symbol renames so it links alongside
// stock without colliding. No functional change vs stock; not routed yet.
// Later stages patch only the K/V load sites to decode TQ-compressed bytes
// into the same mma fragment shape stock FA expects.
//
// Renames vs stock fattn-mma-f16.cuh:
//   - ggml_cuda_flash_attn_ext_mma_f16_case -> tq_cuda_flash_attn_ext_mma_f16_case
//   - DECL_FATTN_MMA_F16_CASE              -> DECL_TQ_FATTN_MMA_F16_CASE
//   - DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2   -> DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2
//
// Internal `static`/inline helpers (flash_attn_ext_f16, *_load_tile,
// *_process_tile, *_iter, ggml_cuda_fattn_mma_get_*) are kept as-is; they
// have internal linkage per TU and never conflict because tq template
// instance .cu files include only this header, never stock's.
//
// ollama: phase-D copy-and-patch — do NOT sync back upstream

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// mma_zero_c: first-mma variant that guarantees C=0.
//
// The stock mma() uses "+r" constraints on (int*)D.x. On Blackwell sm_120,
// ptxas may allocate a DIFFERENT physical register for the inline asm "=r"
// output than the float register D.x[0] holds in the outer context. The write
// goes to a temp int register; the outer float register (which the probe and
// subsequent mma() read as C) remains NaN from decode register pressure.
//
// Fix: use "=f"(D.x[N]) — write directly to the FLOAT register that the outer
// context holds for D.x[N]. No type-pun, no register aliasing. The mma PTX
// .f32 output is naturally a float register, so "=f" is semantically correct.
// Only needed for the FIRST k-tile mma of each accumulator slot; subsequent
// calls use regular mma() which accumulates via "+r" into the same registers.

// Narrow: tile<16,8,float> D — used for KQ_C (cols_per_warp==8) and VKQ_C.
static __device__ __forceinline__ void mma_zero_c(
        tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(D.x[0]), "=f"(D.x[1]), "=f"(D.x[2]), "=f"(D.x[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[1]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#else // Turing: two m16n8k8
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(D.x[0]), "=f"(D.x[1]), "=f"(D.x[2]), "=f"(D.x[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// Wide: tile<16,16,float> D — used for KQ_C (cols_per_warp!=8, column-major).
static __device__ __forceinline__ void mma_zero_c(
        tile<16, 16, float> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
    const int * Axi = (const int *) A.x;
    const int * Bxi = (const int *) B.x;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(D.x[0]), "=f"(D.x[1]), "=f"(D.x[2]), "=f"(D.x[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[2]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(D.x[4]), "=f"(D.x[5]), "=f"(D.x[6]), "=f"(D.x[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[1]), "r"(Bxi[3]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#else // Turing: four m16n8k8
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(D.x[0]), "=f"(D.x[1]), "=f"(D.x[2]), "=f"(D.x[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(D.x[0]), "+f"(D.x[1]), "+f"(D.x[2]), "+f"(D.x[3])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(D.x[4]), "=f"(D.x[5]), "=f"(D.x[6]), "=f"(D.x[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(D.x[4]), "+f"(D.x[5]), "+f"(D.x[6]), "+f"(D.x[7])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#elif defined(AMD_WMMA_AVAILABLE)
#if defined(RDNA4)
    using halfx8_t  = __attribute__((ext_vector_type(8)))  _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8)))  float;
    floatx8_t &        acc_frag = reinterpret_cast<floatx8_t &>(D.x[0]);
    const halfx8_t &   a_frag   = reinterpret_cast<const halfx8_t &>(A.x[0]);
    const halfx8_t &   b_frag   = reinterpret_cast<const halfx8_t &>(B.x[0]);
    floatx8_t zero = {0,0,0,0,0,0,0,0};
    acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, zero);
#elif defined(RDNA3)
    using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8)))  float;
    floatx8_t &        acc_frag = reinterpret_cast<floatx8_t &>(D.x[0]);
    const halfx16_t &  a_frag   = reinterpret_cast<const halfx16_t &>(A.x[0]);
    const halfx16_t &  b_frag   = reinterpret_cast<const halfx16_t &>(B.x[0]);
    floatx8_t zero = {0,0,0,0,0,0,0,0};
    acc_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, zero);
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif // RDNA4
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// TQ kernel argument surface (Stage 2).
//
// Bundles all TurboQuant decode metadata into a single struct so the kernel
// chain (kernel -> process_tile -> iter -> load_tile patches) takes one
// extra param per level instead of ~15 scalars. Stage 2 plumbs the struct
// through; Stage 3-4 wire it into the K/V load sites.
//
// Convention: K_packed == nullptr means "not a TQ launch" — tq_launch_fattn
// and the kernel both gate the TQ-decode branches on that nullptr so Stage 2
// is bit-exact with stock (the existing f16 dequant + stock FA path runs
// when K_packed is null).
struct TqKernelArgs {
    // K compressed buffer pointers (all device memory)
    const uint8_t * K_packed;
    const int16_t * K_outlier_indices;
    const uint8_t * K_outlier_packed;
    const float   * K_scales;
    const float   * K_outlier_scales;
    const float   * K_zeros;
    const float   * K_outlier_zeros;

    // V compressed buffer (mirror layout)
    const uint8_t * V_packed;
    const int16_t * V_outlier_indices;
    const uint8_t * V_outlier_packed;
    const float   * V_scales;
    const float   * V_outlier_scales;
    const float   * V_zeros;
    const float   * V_outlier_zeros;

    // Codebook shared between K and V
    const float   * codebook;

    // K decode parameters
    int K_bits;
    int K_outlier_bits;
    int K_outlier_count;
    int K_packedBytes;          // regular K packed bytes per (cell, head)
    int K_outlier_packedBytes;  // outlier packed bytes per (cell, head)

    // V decode parameters
    int V_bits;
    int V_outlier_bits;
    int V_outlier_count;
    int V_packedBytes;
    int V_outlier_packedBytes;

    // Encoded-K end bound: cells in [0, K_valid_cells) hold real encoded
    // data; cells beyond that are uninitialized cache slots. The kernel
    // iterates over the full K->ne[1] capacity (= ne11) regardless, so the
    // decode helper must explicitly zero tile_K for cell_addr >= K_valid_cells
    // — otherwise NaN scales/zeros from unencoded slots propagate into
    // tile_K, then through Q·K^T into softmax, producing NaN logits.
    // Stock f16 FA doesn't hit this because raw K is fully-populated and
    // the causal mask handles invalid positions; Phase D's compressed K
    // has a partial-fill problem that the mask can't recover from.
    int K_valid_cells;

    // Indexed-addressing support: when non-NULL, locs[logical_cell] maps
    // to physical cache slot. NULL = contiguous (cell = firstCell + c).
    const int32_t * locs;
};

// Type-erased kernel pointer carrying the TQ arg surface. Distinct from
// stock's fattn_kernel_t (different signature) so tq_launch_fattn enforces
// that only TQ-capable kernels are passed.
typedef void (* tq_fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const TqKernelArgs tq_args);

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 4,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  4,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  4,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  4,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#ifdef CP_ASYNC_AVAILABLE
    return ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup) {
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity with cp.async is 16 bytes, with synchronous data loading it's 4 bytes.
    if constexpr (use_cp_async) {
        static_assert(!oob_check, "OOB check not compatible with cp_async");
        constexpr int preload = 64;
        constexpr int h2_per_chunk = 16/sizeof(half2);
        const int chunks_per_row = D2 / h2_per_chunk;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        // TODO use ggml_cuda_memcpy_1
        auto load = [&] __device__ (const int n) {
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
            const int k0_stop  =                             D2 - D2 % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i*stride_tile + k] = !oob_check || i < i_sup ? KV[i*stride_KV + k] : make_half2(0.0f, 0.0f);
                }
            }
        };
        // 1: max 32* 4=128 bytes,  64 half
        // 2: max 16* 4= 64 bytes,  32 half
        // 3: max  8* 4= 32 bytes,  16 half
        // 4: max  4* 4= 16 bytes,   8 half
        ggml_cuda_unroll<4>{}(load);
    }
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int i_sup, const int j0, const uint3 ne01) {
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*WARP_SIZE && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + j_vram*stride_mask + i);
        }
    } else if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[j_vram*stride_mask + i] : half(0.0f);
            }
        }
    } else if constexpr (nbatch_fa < 2*WARP_SIZE) {
        constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (WARP_SIZE/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + j_vram*stride_mask + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*WARP_SIZE) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + j_vram*stride_mask + i);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Local copy of tq-fattn-vec.cuh's per-element decoder. Inlined here so
// tq-fattn-mma-f16.cuh stays self-contained without pulling vec's template
// machinery. Mirrors `tq_decode_elem` at tq-fattn-vec.cuh:10.
static __device__ __forceinline__ float tq_mma_decode_elem(
    const uint8_t * packed_row, const float * codebook, float rms_scale, int elem, int bits) {
    const int bit_pos  = elem * bits;
    const int byte_idx = bit_pos >> 3;
    const int shift    = bit_pos & 7;
    const int mask_val = (1 << bits) - 1;
    int idx = ((int)(packed_row[byte_idx] >> shift)) & mask_val;
    if (shift + bits > 8) {
        idx |= ((int)(packed_row[byte_idx + 1] << (8 - shift))) & mask_val;
    }
    return codebook[idx] * rms_scale;
}

// ------------------------------------------------------------------------------------------------------------------
// tq_decode_K_to_tile — Phase D Stage 3.
//
// Produces the same smem layout that flash_attn_ext_f16_load_tile would have
// produced from a f16-dequant'd K_h2 source: tile_K[i*stride_tile_K + k] holds
// a half2 containing two adjacent head-dim f16 values for cell i (cell-major,
// dim-minor). load_ldmatrix downstream reads this exact layout.
//
// Decode model (mirrors the K-decode in tq-fattn-vec.cuh):
//   For each (cell, d in 0..D-1):
//     1. Linear-scan outlier_indices[cell, head_kv][0..outlier_count) for `d`.
//        - Hit at slot `s` -> decode tq_args.K_outlier_packed[cell, head_kv]
//          at slot `s` with K_outlier_bits, scale by K_outlier_scales[cell, head_kv],
//          add K_outlier_zeros[cell, head_kv] if asymmetric.
//        - Miss -> compute rank-below-d `r = d - count(outlier_pos < d)`, decode
//          tq_args.K_packed[cell, head_kv] at index `r` with K_bits, scale by
//          K_scales[cell, head_kv], add K_zeros[cell, head_kv] if asymmetric.
// Output two adjacent dims (d=2*k_h2, d=2*k_h2+1) packed as half2.
//
// Work distribution: simple flat split — total work is nbatch_fa cells ×
// D2 half2 entries, threads = nwarps*WARP_SIZE. No cp.async pipelining yet
// (Stage 5 candidate optimization).
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq_decode_K_to_tile(
        half2 * const __restrict__ tile_K,
        const int      k_VKQ_0,     // global cell index of this tile's first row
        const int      k0_start_h2, // starting half2 dim index (within full D2)
        const int      D2,          // half2 dim count to decode (k0_diff)
        const int      head_kv,     // KV head index for this block
        const int      nKVHeads,
        const int      i_sup,
        const TqKernelArgs & tq_args) {
    const int bits          = tq_args.K_bits;
    const int packedBytes   = tq_args.K_packedBytes;
    const int olcount       = tq_args.K_outlier_count;
    const int olbits        = tq_args.K_outlier_bits;
    const bool has_outliers = (tq_args.K_outlier_indices != nullptr);
    const bool asymmetric   = (tq_args.K_zeros != nullptr);

    // Outlier packed-row size as written by the encoder (4-byte-aligned).
    const int olpb = tq_args.K_outlier_packedBytes;

    // Codebook layout: codebook[0 .. (1<<bits)) is regular; codebook
    // [(1<<bits) .. (1<<bits)+(1<<olbits)) is the outlier codebook (matches
    // tq-fattn-vec.cuh:1576 `outlier_codebook_ptr = codebook + (1 << bits)`).
    const float * codebook        = tq_args.codebook;
    const float * outlier_cb      = codebook + (1 << bits);

    const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int total_threads = nwarps * WARP_SIZE;
    const int total_work    = nbatch_fa * D2;

    for (int w = tid; w < total_work; w += total_threads) {
        const int i    = w / D2;            // cell-relative index in [0, nbatch_fa)
        const int k_h2 = w % D2;            // half2-dim index in [0, D2)
        const int k_h2_abs = k0_start_h2 + k_h2;  // absolute half2 index in full D

        const int64_t cell_rel = (int64_t)k_VKQ_0 + i;

        // Physical cache slot for this cell — locs[cell_rel] when indexed,
        // cell_rel otherwise. Used for K/V/scales/zeros/outlier addressing
        // off the firstCell-adjusted bases.
        const int64_t cell_addr = (tq_args.locs != nullptr && cell_rel < tq_args.K_valid_cells)
            ? (int64_t)tq_args.locs[cell_rel]
            : cell_rel;

        // Two-tier OOB check:
        //   1. Stock's per-tile k_VKQ_sup bound (oob_check fires on last
        //      iter when the tile partially overlaps ne11).
        //   2. K_valid_cells: TQ-specific cap on actual encoded K. The
        //      kernel iterates through all ne11 cells of the K cache
        //      regardless of how many were encoded; cells past
        //      K_valid_cells contain uninitialized scales/zeros that
        //      would decode to NaN. Zero tile_K instead.
        if ((oob_check && i >= i_sup) || cell_rel >= tq_args.K_valid_cells) {
            tile_K[i*stride_tile + k_h2] = make_half2(0.0f, 0.0f);
            continue;
        }
        const int64_t scale_off = cell_addr * nKVHeads + head_kv;
        const uint8_t * K_row = tq_args.K_packed
            + cell_addr * nKVHeads * packedBytes + head_kv * packedBytes;
        const uint8_t * O_row = has_outliers
            ? tq_args.K_outlier_packed + cell_addr * nKVHeads * olpb + head_kv * olpb
            : nullptr;
        const int16_t * O_idx = has_outliers
            ? tq_args.K_outlier_indices + cell_addr * nKVHeads * olcount + head_kv * olcount
            : nullptr;

        const float rms   = tq_args.K_scales[scale_off];
        const float o_rms = has_outliers ? tq_args.K_outlier_scales[scale_off] : 0.0f;
        const float zero  = asymmetric ? tq_args.K_zeros[scale_off] : 0.0f;
        const float o_zero = (asymmetric && has_outliers && tq_args.K_outlier_zeros)
            ? tq_args.K_outlier_zeros[scale_off] : 0.0f;

        float out[2];
        #pragma unroll
        for (int s_pair = 0; s_pair < 2; ++s_pair) {
            const int d = 2*k_h2_abs + s_pair;
            int outl_slot = -1;
            int outl_below = 0;

            if (has_outliers) {
                for (int s = 0; s < olcount; ++s) {
                    const int pos = O_idx[s];
                    if (pos == d) { outl_slot = s; }
                    outl_below += (pos >= 0) & (pos < d);
                }
            }

            float v;
            if (outl_slot >= 0) {
                v = tq_mma_decode_elem(O_row, outlier_cb, o_rms, outl_slot, olbits);
                if (asymmetric) v += o_zero;
            } else {
                const int r = d - outl_below;
                v = tq_mma_decode_elem(K_row, codebook, rms, r, bits);
                if (asymmetric) v += zero;
            }
            out[s_pair] = v;
        }

        tile_K[i*stride_tile + k_h2] = __floats2half2_rn(out[0], out[1]);
    }
}

// V-decode analogue of tq_decode_K_to_tile. Structurally identical; uses
// V_ fields from TqKernelArgs. K_valid_cells bounds V as well: V and K are
// always co-encoded in K+V presets, so they share the same valid-cell count.
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq_decode_V_to_tile(
        half2 * const __restrict__ tile_V,
        const int      k_VKQ_0,     // global cell index of this tile's first row
        const int      i0_start_h2, // starting half2 dim index within DV
        const int      D2,          // half2 dim count to decode (i0_diff/2)
        const int      head_kv,
        const int      nKVHeads,
        const int      i_sup,
        const TqKernelArgs & tq_args) {
    const int bits          = tq_args.V_bits;
    const int packedBytes   = tq_args.V_packedBytes;
    const int olcount       = tq_args.V_outlier_count;
    const int olbits        = tq_args.V_outlier_bits;
    const bool has_outliers = (tq_args.V_outlier_indices != nullptr);
    const bool asymmetric   = (tq_args.V_zeros != nullptr);
    const int olpb          = tq_args.V_outlier_packedBytes;

    const float * codebook   = tq_args.codebook;
    const float * outlier_cb = codebook + (1 << bits);

    const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int total_threads = nwarps * WARP_SIZE;
    const int total_work    = nbatch_fa * D2;

    for (int w = tid; w < total_work; w += total_threads) {
        const int i    = w / D2;
        const int v_h2 = w % D2;
        const int v_h2_abs = i0_start_h2 + v_h2;

        const int64_t cell_rel = (int64_t)k_VKQ_0 + i;

        // Physical cache slot for this cell — locs[cell_rel] when indexed,
        // cell_rel otherwise. Used for V/scales/zeros/outlier addressing.
        const int64_t cell_addr = (tq_args.locs != nullptr && cell_rel < tq_args.K_valid_cells)
            ? (int64_t)tq_args.locs[cell_rel]
            : cell_rel;

        if ((oob_check && i >= i_sup) || cell_rel >= tq_args.K_valid_cells) {
            tile_V[i*stride_tile + v_h2] = make_half2(0.0f, 0.0f);
            continue;
        }
        const int64_t scale_off = cell_addr * nKVHeads + head_kv;
        const uint8_t * V_row = tq_args.V_packed
            + cell_addr * nKVHeads * packedBytes + head_kv * packedBytes;
        const uint8_t * O_row = has_outliers
            ? tq_args.V_outlier_packed + cell_addr * nKVHeads * olpb + head_kv * olpb
            : nullptr;
        const int16_t * O_idx = has_outliers
            ? tq_args.V_outlier_indices + cell_addr * nKVHeads * olcount + head_kv * olcount
            : nullptr;

        const float rms    = tq_args.V_scales[scale_off];
        const float o_rms  = has_outliers ? tq_args.V_outlier_scales[scale_off] : 0.0f;
        const float zero   = asymmetric ? tq_args.V_zeros[scale_off] : 0.0f;
        const float o_zero = (asymmetric && has_outliers && tq_args.V_outlier_zeros)
            ? tq_args.V_outlier_zeros[scale_off] : 0.0f;

        float out[2];
        #pragma unroll
        for (int s_pair = 0; s_pair < 2; ++s_pair) {
            const int d = 2*v_h2_abs + s_pair;
            int outl_slot  = -1;
            int outl_below = 0;

            if (has_outliers) {
                for (int s = 0; s < olcount; ++s) {
                    const int pos = O_idx[s];
                    if (pos == d) { outl_slot = s; }
                    outl_below += (pos >= 0) & (pos < d);
                }
            }

            float v;
            if (outl_slot >= 0) {
                v = tq_mma_decode_elem(O_row, outlier_cb, o_rms, outl_slot, olbits);
                if (asymmetric) v += o_zero;
            } else {
                const int r = d - outl_below;
                v = tq_mma_decode_elem(V_row, codebook, rms, r, bits);
                if (asymmetric) v += zero;
            }
            out[s_pair] = v;
        }

        tile_V[i*stride_tile + v_h2] = __floats2half2_rn(out[0], out[1]);
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const int head_kv,
        const int nKVHeads,
        const TqKernelArgs & tq_args) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2);

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    constexpr int stride_tile_V = mla ? stride_tile_K : nbatch_V2 + 4;

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    // KQ_C zero-init is handled by mma_zero_c on the first k-tile mma call below.

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!mla, "multi-stage loading not implemented for MLA");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        // Phase D Stage 4: K+V presets need V TQ-decoded here. The multi-stage
        // pipeline previously only f16-loaded V_h2 (correct for K-only, where V
        // is real f16), but for K+V V_h2 is TQ-packed bytes — f16-loading them
        // yields garbage -> NaN. Decode synchronously into tile_V (no cp.async,
        // mirroring the synchronous K decode below); tile_V is made visible by
        // the __syncthreads in the next-iter preload block before the VKQ mma
        // consumes it. K-only (V_packed == nullptr) keeps the f16 load.
        if (tq_args.V_packed != nullptr) {
            tq_decode_V_to_tile<stride_tile_V, nwarps, nbatch_fa, oob_check>
                (tile_V, k_VKQ_0, 0, nbatch_V2, head_kv, nKVHeads, k_VKQ_sup, tq_args);
        } else {
            flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
        }
    } else {
        constexpr bool use_cp_async = nstages == 1;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + k_VKQ_0, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }
    }

#pragma unroll
    for (int k0_start = 0; k0_start < DKQ/2; k0_start += nbatch_K2) {
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;
        const int k0_diff = k0_stop - k0_start;

        if constexpr (nstages <= 1) {
            constexpr bool use_cp_async = nstages == 1;
            // Phase D Stage 3: when tq_args.K_packed is set, decode TQ-compressed
            // K into tile_K instead of f16-loading. The smem layout matches what
            // load_ldmatrix downstream expects (tile_K[i*stride + k_h2]).
            if (tq_args.K_packed != nullptr) {
                tq_decode_K_to_tile<stride_tile_K, nwarps, nbatch_fa, oob_check>
                    (tile_K, k_VKQ_0, k0_start, k0_diff, head_kv, nKVHeads, k_VKQ_sup, tq_args);
            } else {
                flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, k0_diff, stride_K, k_VKQ_sup);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
            }
            __syncthreads();
        }

        // Calculate tile of KQ.
        // First k-tile uses mma_zero_c to guarantee C=0: mma() "+r" reads the
        // accumulator register as C, but decode compute may have recycled those
        // registers, leaving NaN on high-register-pressure arches (sm_120).
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0  = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
                const int slot    = i_KQ_00 / (np * T_A_KQ::I);
                T_A_KQ K_A;
                // First k-tile: zero accumulator explicitly.
                load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + 0, stride_tile_K);
                if constexpr (cols_per_warp == 8) {
                    mma_zero_c(KQ_C[slot], K_A, Q_B[k0_start/T_A_KQ::J]);
                } else {
                    mma_zero_c(KQ_C[slot], Q_B[k0_start/T_A_KQ::J], K_A);
                }
                // Remaining k-tiles: regular accumulation (D now holds valid result).
#pragma unroll
                for (int k_KQ_0 = k0_start + T_A_KQ::J; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[slot], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        mma(KQ_C[slot], Q_B[k_KQ_0/T_A_KQ::J], K_A);
                    }
                }
            }
        } else {
            // First k-tile: load Q, then mma_zero_c for each row.
            {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k0_start, stride_tile_Q);
#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + 0, stride_tile_K);
                    if constexpr (cols_per_warp == 8) {
                        mma_zero_c(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
#if defined(AMD_WMMA_AVAILABLE)
                        mma_zero_c(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        mma_zero_c(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif
                    }
                }
            }
            // Remaining k-tiles: regular accumulation.
#pragma unroll
            for (int k_KQ_0 = k0_start + T_A_KQ::J; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);
#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
                    T_A_KQ K_A;
                    load_ldmatrix(K_A, tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
#if defined(AMD_WMMA_AVAILABLE)
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[l % 2]);
                    KQ_rowsum_add[l % 2] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    // Turing + Volta:
                    KQ_max_new[(l/2) % 2] = fmaxf(KQ_max_new[(l/2) % 2], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#else
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                // Turing + Volta:
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[(l/2) % 2]);
                    KQ_rowsum_add[(l/2) % 2] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (mask_h + k_VKQ_0 + nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            }
            // Phase D Stage 3: in TQ-mode, decode next iter's tile_K synchronously.
            // The cp.async pipeline assumption (next-iter K already in smem on
            // entry to next iter) is preserved — we just fill the buffer with
            // decoded f16 instead of f16-loaded bytes.
            if (tq_args.K_packed != nullptr) {
                tq_decode_K_to_tile<stride_tile_K, nwarps, nbatch_fa, oob_check>
                    (tile_K, k_VKQ_0 + nbatch_fa, 0, nbatch_K2, head_kv, nKVHeads, k_VKQ_sup, tq_args);
            } else {
                flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
            }
        }
    }


    // For MLA K and V have the same data.
    // Therefore, iterate over V in reverse and re-use the data if possible.
    static_assert(!mla || nstages <= 1, "combination of MLA and multi-stage loading not implemented");
    constexpr int reusable_cutoff = mla ? (DKQ - 1) - (DKQ - 1) % (2*nbatch_K2) - (DKQ - DV) : DV;

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_stop = DV; i0_stop > 0; i0_stop -= 2*nbatch_V2) {
        const int i0_start = i0_stop - 2*nbatch_V2 > 0 ? i0_stop - 2*nbatch_V2 : 0;
        const int i0_diff  = i0_stop - i0_start;

        if constexpr (nstages <= 1) {
            if (i0_start < reusable_cutoff) {
                constexpr bool use_cp_async = nstages == 1;
                if (tq_args.V_packed != nullptr) {
                    tq_decode_V_to_tile<stride_tile_V, nwarps, nbatch_fa, oob_check>
                        (tile_V, k_VKQ_0, i0_start/2, i0_diff/2, head_kv, nKVHeads, k_VKQ_sup, tq_args);
                } else {
                    flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                        (V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                }
                __syncthreads();
            }
        }
        const half2 * tile_V_i = i0_start < reusable_cutoff ? tile_V : tile_V + (i0_start - reusable_cutoff)/2;

#if defined(TURING_MMA_AVAILABLE)
        constexpr int i0_stride = cols_per_warp == 8 ? T_C_VKQ::I : 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major => swap A and B.
                    mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::J)], A);
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix(A, tile_V_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            __syncthreads(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0,
        head_kv, nKVHeads,
        tq_args);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<> struct mma_tile_sizes<8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#else // Volta
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop,
        const int head_kv,
        const int nKVHeads,
        const TqKernelArgs & tq_args) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    constexpr int stride_tile_K = nbatch_K2 + 4;

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    constexpr int stride_tile_V = mla ? stride_tile_K : nbatch_V2 + 4;
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
    constexpr int nvkqc = cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J);
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
    constexpr int nvkqc =                              DV/(2*T_C_VKQ::J);
#endif // defined(TURING_MMA_AVAILABLE)
    // Zero VKQ_C before the kb0 loop — registers are undefined at kernel start.
    // Use = {} (C++ value-init) so this compiles for both float and half2 element
    // types without triggering MSVC's no-implicit-float-to-half2-conversion error.
#pragma unroll
    for (int iv = 0; iv < nvkqc; ++iv) {
#pragma unroll
        for (int lv = 0; lv < T_C_VKQ::ne; ++lv) {
            VKQ_C[iv].x[lv] = {};
        }
    }

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if (jt*ncols1 + j < int(ne01.z)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }

    }

    __syncthreads();

    int kb0 = kb0_start;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + kb0*nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }
        // Phase D Stage 3: same dance as the iter's cp.async-preload site —
        // first-iter K tile filled by TQ decode in TQ-mode, f16 load otherwise.
        if (tq_args.K_packed != nullptr) {
            tq_decode_K_to_tile<stride_tile_K, nwarps, nbatch_fa, oob_check>
                (tile_K, kb0*nbatch_fa, 0, nbatch_K2, head_kv, nKVHeads, k_VKQ_sup, tq_args);
        } else {
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
        }
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = true;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, head_kv, nKVHeads, tq_args);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, head_kv, nKVHeads, tq_args);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop-1; ++kb0) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, head_kv, nKVHeads, tq_args);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, head_kv, nKVHeads, tq_args);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchonization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col);
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && jt*ncols1 + j_dst >= int(ne01.z)) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop, head_kv, nKVHeads, tq_args);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool mla>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const TqKernelArgs tq_args) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE))

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

    static_assert(!mla || DKQ >= DV, "MLA needs DKQ >= DV");

    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = mla ? stride_K : nb21 / sizeof(half2);

    const int iter_k = (ne11   + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j = (ne01.z + (ncols1    - 1)) / ncols1;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
        const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;
        // Phase D: KV-head index for this block (head0 maps to head_kv via GQA ratio).
        const int head_kv  = head0 / gqa_ratio;
        const int nKVHeads = ne12;

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + head0) * (DV/2);

        const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
        const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, head_kv, nKVHeads, tq_args);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, head_kv, nKVHeads, tq_args);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
    const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
    const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

    const int head0 = zt * ncols2;
    const int head_kv  = head0 / gqa_ratio;
    const int nKVHeads = ne12;

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + head0) * (DV/2);

    const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
    const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, mla, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, head_kv, nKVHeads, tq_args);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
              tq_args);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE))
}

// tq_launch_fattn: TQ-aware copy of stock launch_fattn (fattn-common.cuh:773).
//
// Forks of stock are minimal:
//   - takes a `const TqKernelArgs & tq_args` and forwards it to the kernel
//   - gates the K/V f16 conversion on `tq_args.K_packed/V_packed == nullptr`
//     so a TQ-mode launch (Stage 3+) skips the dequant materialization while
//     a dummy-args launch (Stage 2) still produces a stock-f16 K_data pointer.
//
// Kept verbatim where unchanged so `diff` against stock shows exactly the
// Phase D delta.
template <int DV, int ncols1, int ncols2>
void tq_launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, tq_fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k,
    const TqKernelArgs & tq_args, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const bool is_mla = DV == 512; // TODO better parameterization

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(V || is_mla);

    const ggml_tensor * mask  = dst->src[3];
    // TQ op: src[4] = K_scales (not attention sinks). TQ ops never use sink
    // attention, so pass nullptr to the kernel to suppress the sinks path.
    const ggml_tensor * sinks = nullptr;

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(      Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(      K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(!V || V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = V ? (const char *) V->data : nullptr;
    size_t nb21 = V ? V->nb[1] : nb11;
    size_t nb22 = V ? V->nb[2] : nb12;
    size_t nb23 = V ? V->nb[3] : nb13;

    // Phase D gate: when TQ-mode (K_packed non-null), skip f16 dequant for K.
    // Stage 2 passes nullptr so behavior matches stock launch_fattn exactly.
    if (need_f16_K && K->type != GGML_TYPE_F16 && tq_args.K_packed == nullptr) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    // Phase D gate: same as K, applied to V.
    if (V && need_f16_V && V->type != GGML_TYPE_F16 && tq_args.V_packed == nullptr) {
        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        V_f16.alloc(ggml_nelements(V));
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
            V_data = (char *) V_f16.ptr;

            nb21 = nb21*bs*sizeof(half)/ts;
            nb22 = nb22*bs*sizeof(half)/ts;
            nb23 = nb23*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(V->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            const int64_t s01 = nb21 / ts;
            const int64_t s02 = nb22 / ts;
            const int64_t s03 = nb23 / ts;
            to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

            nb21 = V->ne[0] * sizeof(half);
            nb22 = V->ne[1] * nb21;
            nb23 = V->ne[2] * nb22;
        }
        V_data = (char *) V_f16.ptr;
    }

    // Phase D Bug 2 fix: cap the FA iter loop at K_valid_cells. Path 2c's
    // DequantK output K has ne[1] sized to actual encoded cells; stock FA
    // iterates only that range. Phase D's K_packed has ne[1]=cache_capacity,
    // so without this cap the kernel iterates past valid K (clamped to 0 by
    // tq_decode_K_to_tile's K_valid_cells check) AND past the mask's valid
    // range — mask reads past its allocation get uninitialized memory which
    // can be NaN, contaminating softmax → P·V → output. The mask is sized
    // to match K's encoded length (118 for our 117-token test), not the
    // 512-cell cache capacity. Substituting K_valid_cells for kernel ne11
    // bounds the iteration to match the mask, mirroring Path 2c's behavior.
    const int32_t kernel_ne11 = (tq_args.K_packed != nullptr && V != nullptr)
        ? (int32_t) tq_args.K_valid_cells
        : (int32_t) K->ne[1];

    const int ntiles_x = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    if (mask && kernel_ne11 % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = kernel_ne11 / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    dim3 blocks_num;
    if (stream_k) {
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks*tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        dst_tmp_meta.alloc(blocks_num.x*ncols * (2*2 + DV) * sizeof(float));
    } else {
        const int ntiles_KQ = (kernel_ne11 + nbatch_fa - 1) / nbatch_fa;

        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);

        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_total * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = (Q->ne[2]/ncols2)*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    // TQ op_params[2] = K_bits (int32), not logit_softcap. TQ attention does not
    // support logit_softcap; reading op_params[2] as float would corrupt scale.
    const float logit_softcap = 0.0f;

    memcpy(&scale,    (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    // Phase D Stage 3 critical fix: on the TQ FA op (Path 4), the K tensor is
    // TQ-compressed and K->ne[2] is KV cache capacity, NOT the real nKVHeads.
    // The kernel computes gqa_ratio = ne02 / ne12 and needs the real value.
    //   K-only presets (V_packed==nullptr): V is f16, ggml_permute(0,2,1,3) applied →
    //     ne[2]=nKVHeads, ne[1]=nCells.
    //   K+V presets   (V_packed!=nullptr):  V is TQ-compressed, use V_scales->ne[0]
    //     instead (V_scales shape is [nKVHeads, nCells]).
    const int32_t kernel_ne12 = (tq_args.K_packed != nullptr && V != nullptr)
        ? (int32_t) (tq_args.V_packed != nullptr
            ? dst->src[6]->ne[0]  // K+V: V_scales->ne[0] = nKVHeads
            : V->ne[2])           // K-only: permuted f16 V->ne[2] = nKVHeads (ne[1] is nCells)
        : (int32_t) K->ne[2];
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], kernel_ne11, kernel_ne12, K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        tq_args
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (ntiles_total % blocks_num.x != 0) {
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], Q->ne[3], kernel_ne11, nbatch_fa);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int DKQ, int DV, int ncols1, int ncols2>
void tq_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, turing_mma_available(cc) ? 16 : 32);
    const int nwarps        = nthreads / WARP_SIZE;

    constexpr bool mla = DKQ == 576;

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    // TQ op_params[2] = K_bits (int32_t), not logit_softcap. Reading it as float
    // gives ≈ 4.2e-45 (nonzero) for K_bits=3, which would select the
    // use_logit_softcap=true template and then compute KQ = 0.0f*tanh(KQ) = 0.0f,
    // zeroing all attention scores (PPL=37302). TQ attention never uses logit
    // softcap; always select the false variant.
    constexpr bool use_logit_softcap = false;
    tq_fattn_kernel_t fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, mla>;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    {
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(fattn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    // Phase D Stage 3: extract TQ-compressed K buffer pointers and metadata
    // from dst (op TQ_FLASH_ATTN_EXT source slots + op_params). Layout mirrors
    // tq-fattn-konly-outlier.cu:
    //   src[1] = K_packed       src[4]  = K_scales      src[5]  = codebook
    //   src[8] = K_zeros        src[12] = K_outlier_packed
    //   src[13] = K_outlier_scales  src[14] = K_outlier_indices
    //   src[15] = K_outlier_zeros
    //   src[11] = locs (NULL = contiguous; [nCells] i32 = indexed slots)
    //   op_params[2] = K_bits   op_params[7] = K_outlier_bits
    //   op_params[8] = K_outlier_count  op_params[9] = K_outlier_packed_bytes
    // Stage 4 wires V decode for K+V presets (src[2]=V_packed, src[6]=V_scales,
    // src[7]=V_codebook, op_params[4]=V_bits). K-only presets leave V_ fields
    // zero-initialized; the V load site null-checks V_packed to select path.
    TqKernelArgs tq_args = {};
    {
        const ggml_tensor * K_t   = dst->src[1];
        const ggml_tensor * scl_t = dst->src[4];
        const ggml_tensor * cb_t  = dst->src[5];
        const ggml_tensor * z_t   = dst->src[8];
        const ggml_tensor * op_t  = dst->src[12];
        const ggml_tensor * os_t  = dst->src[13];
        const ggml_tensor * oi_t  = dst->src[14];
        const ggml_tensor * oz_t  = dst->src[15];
        const ggml_tensor * locs_t = dst->src[11];

        tq_args.K_packed          = K_t   ? (const uint8_t *) K_t->data   : nullptr;
        tq_args.K_scales          = scl_t ? (const float   *) scl_t->data : nullptr;
        tq_args.codebook          = cb_t  ? (const float   *) cb_t->data  : nullptr;
        tq_args.K_zeros           = z_t   ? (const float   *) z_t->data   : nullptr;
        tq_args.K_outlier_packed  = op_t  ? (const uint8_t *) op_t->data  : nullptr;
        tq_args.K_outlier_scales  = os_t  ? (const float   *) os_t->data  : nullptr;
        tq_args.K_outlier_indices = oi_t  ? (const int16_t *) oi_t->data  : nullptr;
        tq_args.K_outlier_zeros   = oz_t  ? (const float   *) oz_t->data  : nullptr;
        tq_args.locs              = locs_t ? (const int32_t *) locs_t->data : nullptr;

        memcpy(&tq_args.K_bits,                 (const int32_t *) dst->op_params + 2, sizeof(int32_t));
        memcpy(&tq_args.K_outlier_bits,         (const int32_t *) dst->op_params + 7, sizeof(int32_t));
        memcpy(&tq_args.K_outlier_count,        (const int32_t *) dst->op_params + 8, sizeof(int32_t));
        memcpy(&tq_args.K_outlier_packedBytes,  (const int32_t *) dst->op_params + 9, sizeof(int32_t));

        // Regular K packedBytes derives from (D - outlier_count) * bits, rounded
        // up to 4 bytes — matches tq-fattn-konly-outlier.cu:47-48.
        const int D_full        = (int) dst->src[0]->ne[0]; // Q->ne[0] == DKQ
        const int regular_count = D_full - tq_args.K_outlier_count;
        const int raw           = (regular_count * tq_args.K_bits + 7) / 8;
        tq_args.K_packedBytes   = (raw + 3) & ~3;

        // K_valid_cells = firstCell + Q->ne[1] (last cell index just encoded).
        // op_params[3] is firstCell (offset into K cache for this batch);
        // Q->ne[1] is the number of new tokens this batch is appending. Their
        // sum is the encoded-end bound the decode helper uses to clamp reads.
        int firstCell = 0;
        memcpy(&firstCell, (const int32_t *) dst->op_params + 3, sizeof(int32_t));
        tq_args.K_valid_cells = firstCell + (int) dst->src[0]->ne[1];

        // V decode fields for K+V presets. V_outlier_indices stays nullptr
        // (outlier split is K-only in Preset struct; V has no outlier split).
        // V_zeros stays nullptr: TQEncodeKV always uses symmetric V encode
        // (tq-encode.cu passes zeros=nullptr). Codebook reused from K because
        // K_bits==V_bits for all ship presets; same ExportCodebook args → same
        // entries. V_packed nullptr → V load site falls back to stock f16 path.
        const ggml_tensor * v_scl_t = dst->src[6];
        if (v_scl_t != nullptr) {
            const ggml_tensor * vp_t = dst->src[2];
            tq_args.V_packed = vp_t ? (const uint8_t *) vp_t->data : nullptr;
            tq_args.V_scales = (const float *) v_scl_t->data;
            memcpy(&tq_args.V_bits, (const int32_t *) dst->op_params + 4, sizeof(int32_t));
            tq_args.V_packedBytes = (D_full * tq_args.V_bits + 7) / 8;
        }
    }

    tq_launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, tq_args);
}


#define DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void tq_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_TQ_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_TQ_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  2,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  4,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  8,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512, 16,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  1,  8);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  2,  8);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  4,  8);
extern DECL_TQ_FATTN_MMA_F16_CASE(512, 512,  8,  8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// For GLM 4.7 Flash
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_TQ_FATTN_MMA_F16_CASE(576, 512, 16,  4);
