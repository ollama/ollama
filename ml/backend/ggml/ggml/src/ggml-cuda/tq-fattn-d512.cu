#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

// All D=512 flash-attention dispatch (K-only and K+V fused, with/without outliers).
// Split from tq-fattn.cu to stay under the gas single-object size limit for
// 10-arch fatbinaries. ggml_cuda_tq_flash_attn_ext routes here when D==512.
// src[9..11] and op_params[5] are reserved; outlier sources stay at src[12..15].
void ggml_cuda_tq_flash_attn_ext_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * v_scales_t   = dst->src[6];
    const ggml_tensor * v_codebook_t = dst->src[7];
    const ggml_tensor * zeros_t          = dst->src[8];
    const ggml_tensor * locs_t           = dst->src[11];
    const ggml_tensor * outlier_packed_t  = dst->src[12];
    const ggml_tensor * outlier_scales_t  = dst->src[13];
    const ggml_tensor * outlier_indices_t = dst->src[14];
    const ggml_tensor * outlier_zeros_t   = dst->src[15];

    float   scale;
    float   logit_softcap;
    int32_t bits;
    int32_t firstCell;
    int32_t v_bits;
    int32_t asymmetric;
    int32_t outlier_bits;
    int32_t outlier_count;
    int32_t outlier_packed_bytes;
    memcpy(&scale,                (const float   *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap,        (const float   *)dst->op_params + 1, sizeof(float));
    memcpy(&bits,                 (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&firstCell,            (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    memcpy(&v_bits,               (const int32_t *)dst->op_params + 4, sizeof(int32_t));
    // op_params[5] is reserved.
    memcpy(&asymmetric,           (const int32_t *)dst->op_params + 6, sizeof(int32_t));
    memcpy(&outlier_bits,         (const int32_t *)dst->op_params + 7, sizeof(int32_t));
    memcpy(&outlier_count,        (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    memcpy(&outlier_packed_bytes, (const int32_t *)dst->op_params + 9, sizeof(int32_t));

    const bool v_packed = (v_scales_t != nullptr);

    int nCells;
    if (v_packed) {
        GGML_ASSERT(dst->src[3] != nullptr && "TQ K+V fused requires mask to determine nCells");
        nCells = (int)dst->src[3]->ne[0];
    } else {
        nCells = (int)V->ne[1];
    }

    const bool has_outliers = (outlier_count > 0);
    int packedBytes;
    if (has_outliers) {
        const int regular_count = 512 - outlier_count;
        const int raw = (regular_count * bits + 7) / 8;
        packedBytes = (raw + 3) & ~3;
    } else {
        packedBytes = (512 * bits + 7) / 8;
    }
    const int v_packedBytes = v_packed ? ((512 * v_bits + 7) / 8) : 0;
    const int nKVHeads = v_packed ? (int)V->ne[0] / v_packedBytes : (int)V->ne[2];

    const float * v_scales_ptr   = v_scales_t   ? (const float *)v_scales_t->data   : nullptr;
    const float * v_codebook_ptr = v_codebook_t ? (const float *)v_codebook_t->data : nullptr;
    const float   * zeros_ptr          = zeros_t          ? (const float   *)zeros_t->data          : nullptr;

    const uint8_t * outlier_packed_ptr  = outlier_packed_t  ? (const uint8_t *)outlier_packed_t->data  : nullptr;
    const float   * outlier_scales_ptr  = outlier_scales_t  ? (const float   *)outlier_scales_t->data  : nullptr;
    const int16_t * outlier_indices_ptr = outlier_indices_t ? (const int16_t *)outlier_indices_t->data : nullptr;
    const float   * outlier_zeros_ptr   = outlier_zeros_t   ? (const float   *)outlier_zeros_t->data   : nullptr;
    const int32_t * locs_ptr            = locs_t            ? (const int32_t *)locs_t->data            : nullptr;

    if (logit_softcap != 0.0f) { scale /= logit_softcap; }

    const int nTokensQ = (int)Q->ne[1];
    // D=512 prefill: ncols=8 smem=57KB > 48KB Pascal limit → cap at ncols=2.
    const int ncols = (nTokensQ == 1) ? 1 : 2;

    // All four V_PACKED × HAS_OUTLIERS combinations for D=512.
    #define DISPATCH512(NCOLS, SOFTCAP) \
        if (v_packed && has_outliers) { \
            tq_fattn_vec_launch<512, NCOLS, SOFTCAP, true, true>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                v_bits, v_packedBytes, v_scales_ptr, v_codebook_ptr, \
                zeros_ptr, asymmetric, \
                outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr, \
                outlier_bits, outlier_count, outlier_packed_bytes, locs_ptr); \
        } else if (v_packed) { \
            tq_fattn_vec_launch<512, NCOLS, SOFTCAP, true, false>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                v_bits, v_packedBytes, v_scales_ptr, v_codebook_ptr, \
                zeros_ptr, asymmetric, \
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0, locs_ptr); \
        } else if (has_outliers) { \
            tq_fattn_vec_launch<512, NCOLS, SOFTCAP, false, true>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                0, 0, nullptr, nullptr, \
                zeros_ptr, asymmetric, \
                outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr, \
                outlier_bits, outlier_count, outlier_packed_bytes, locs_ptr); \
        } else { \
            tq_fattn_vec_launch<512, NCOLS, SOFTCAP, false, false>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                0, 0, nullptr, nullptr, \
                zeros_ptr, asymmetric, \
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0, locs_ptr); \
        }

    if (ncols == 1) {
        if (logit_softcap == 0.0f) { DISPATCH512(1, false); }
        else                       { DISPATCH512(1, true);  }
    } else {
        if (logit_softcap == 0.0f) { DISPATCH512(2, false); }
        else                       { DISPATCH512(2, true);  }
    }
    #undef DISPATCH512

    CUDA_CHECK(cudaGetLastError());
}
