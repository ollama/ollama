#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

// K-only fused flash-attention with outlier-split for D=256
// (V_PACKED=false, HAS_OUTLIERS=true).  Split from tq-fattn-konly-outlier.cu
// so each TU stays under the gas single-object size limit (~2 GiB).
// D=512 is handled by tq-fattn-d512.cu.
void ggml_cuda_tq_flash_attn_ext_konly_outlier_wide(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * zeros_t          = dst->src[8];
    const ggml_tensor * qjl_packed_t     = dst->src[9];
    const ggml_tensor * qjl_norm_t       = dst->src[10];
    const ggml_tensor * qjl_projection_t = dst->src[11];
    const ggml_tensor * outlier_packed_t  = dst->src[12];
    const ggml_tensor * outlier_scales_t  = dst->src[13];
    const ggml_tensor * outlier_indices_t = dst->src[14];
    const ggml_tensor * outlier_zeros_t   = dst->src[15];

    float   scale;
    float   logit_softcap;
    int32_t bits;
    int32_t firstCell;
    int32_t qjl_rows;
    int32_t asymmetric;
    int32_t outlier_bits;
    int32_t outlier_count;
    int32_t outlier_packed_bytes;
    memcpy(&scale,                (const float   *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap,        (const float   *)dst->op_params + 1, sizeof(float));
    memcpy(&bits,                 (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&firstCell,            (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    // op_params[4] = v_bits (unused here; V is f16)
    memcpy(&qjl_rows,             (const int32_t *)dst->op_params + 5, sizeof(int32_t));
    memcpy(&asymmetric,           (const int32_t *)dst->op_params + 6, sizeof(int32_t));
    memcpy(&outlier_bits,         (const int32_t *)dst->op_params + 7, sizeof(int32_t));
    memcpy(&outlier_count,        (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    memcpy(&outlier_packed_bytes, (const int32_t *)dst->op_params + 9, sizeof(int32_t));

    GGML_ASSERT(outlier_count > 0 && "konly-outlier-wide path requires outlier_count > 0");

    const int D        = (int)Q->ne[0];
    const int nTokensQ = (int)Q->ne[1];
    const int nCells   = (int)V->ne[1]; // V is f16 [D, nCells, nKVHeads]
    const int nKVHeads = (int)V->ne[2];

    const int regular_count = D - outlier_count;
    const int raw = (regular_count * bits + 7) / 8;
    const int packedBytes = (raw + 3) & ~3;

    const float   * zeros_ptr          = zeros_t          ? (const float   *)zeros_t->data          : nullptr;
    const uint8_t * qjl_packed_ptr     = qjl_packed_t     ? (const uint8_t *)qjl_packed_t->data     : nullptr;
    const float   * qjl_norm_ptr       = qjl_norm_t       ? (const float   *)qjl_norm_t->data       : nullptr;
    const float   * qjl_projection_ptr = qjl_projection_t ? (const float   *)qjl_projection_t->data : nullptr;
    const int qjl_packed_raw  = (qjl_rows + 7) / 8;
    const int qjl_packedBytes = (qjl_packed_raw + 3) & ~3;

    const uint8_t * outlier_packed_ptr  = outlier_packed_t  ? (const uint8_t *)outlier_packed_t->data  : nullptr;
    const float   * outlier_scales_ptr  = outlier_scales_t  ? (const float   *)outlier_scales_t->data  : nullptr;
    const int16_t * outlier_indices_ptr = outlier_indices_t ? (const int16_t *)outlier_indices_t->data : nullptr;
    const float   * outlier_zeros_ptr   = outlier_zeros_t   ? (const float   *)outlier_zeros_t->data   : nullptr;

    if (logit_softcap != 0.0f) { scale /= logit_softcap; }

    int ncols;
    if      (nTokensQ == 1) ncols = 1;
    else if (nTokensQ < 8)  ncols = 2;
    else                    ncols = 8;

    #define DISPATCH_KO(DIM, NCOLS, SOFTCAP) \
        tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, false, true>(ctx, dst, scale, logit_softcap, \
            bits, firstCell, nCells, nKVHeads, packedBytes, \
            0, 0, nullptr, nullptr, \
            zeros_ptr, asymmetric, \
            qjl_packed_ptr, qjl_norm_ptr, qjl_projection_ptr, qjl_rows, qjl_packedBytes, \
            outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr, \
            outlier_bits, outlier_count, outlier_packed_bytes)

    #define DISPATCH_KO_NCOLS(DIM, SOFTCAP) \
        if      (ncols == 1) { DISPATCH_KO(DIM, 1, SOFTCAP); } \
        else if (ncols == 2) { DISPATCH_KO(DIM, 2, SOFTCAP); } \
        else                 { DISPATCH_KO(DIM, 8, SOFTCAP); }

    // D=256 only; D=512 handled by tq-fattn-d512.cu.
    if (logit_softcap == 0.0f) { DISPATCH_KO_NCOLS(256, false); }
    else                       { DISPATCH_KO_NCOLS(256, true);  }

    #undef DISPATCH_KO_NCOLS
    #undef DISPATCH_KO

    CUDA_CHECK(cudaGetLastError());
}
