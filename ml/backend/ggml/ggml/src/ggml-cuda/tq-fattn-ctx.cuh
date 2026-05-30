#pragma once
// Shared parameter extraction for TQ flash-attention per-dim dispatch TUs.
// Each dim-specific .cu file includes tq-fattn-vec.cuh and then this header,
// which unpacks op_params and src[] into named locals.  The TU then calls
// tq_fattn_vec_launch for the single fixed DIM it handles.

#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

struct TQFattnCtx {
    float   scale;
    float   logit_softcap;
    int32_t bits;
    int32_t firstCell;
    int32_t v_bits;
    int32_t asymmetric;
    int32_t outlier_bits;
    int32_t outlier_count;
    int32_t outlier_packed_bytes;

    bool v_packed;
    bool has_outliers;
    int  nCells;
    int  nKVHeads;
    int  packedBytes;
    int  v_packedBytes;
    int  nTokensQ;

    const float   * v_scales_ptr;
    const float   * v_codebook_ptr;
    const float   * zeros_ptr;
    const uint8_t * outlier_packed_ptr;
    const float   * outlier_scales_ptr;
    const int16_t * outlier_indices_ptr;
    const float   * outlier_zeros_ptr;
    const int32_t * locs_ptr;
};

// Extract TQFattnCtx from a TQ_FLASH_ATTN_EXT dst tensor.
// D is the fixed head-dim for the calling TU (used to compute packedBytes).
// src[9..10] reserved; src[11] is the indexed-addressing locs tensor.
static __host__ inline TQFattnCtx tq_fattn_extract(const ggml_tensor * dst, int D) {
    const ggml_tensor * Q             = dst->src[0];
    const ggml_tensor * V             = dst->src[2];
    const ggml_tensor * v_scales_t    = dst->src[6];
    const ggml_tensor * v_codebook_t  = dst->src[7];
    const ggml_tensor * zeros_t       = dst->src[8];
    const ggml_tensor * locs_t        = dst->src[11];
    const ggml_tensor * ol_packed_t   = dst->src[12];
    const ggml_tensor * ol_scales_t   = dst->src[13];
    const ggml_tensor * ol_indices_t  = dst->src[14];
    const ggml_tensor * ol_zeros_t    = dst->src[15];

    TQFattnCtx c;
    memcpy(&c.scale,                (const float   *)dst->op_params + 0, sizeof(float));
    memcpy(&c.logit_softcap,        (const float   *)dst->op_params + 1, sizeof(float));
    memcpy(&c.bits,                 (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&c.firstCell,            (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    memcpy(&c.v_bits,               (const int32_t *)dst->op_params + 4, sizeof(int32_t));
    // op_params[5] is reserved.
    memcpy(&c.asymmetric,           (const int32_t *)dst->op_params + 6, sizeof(int32_t));
    memcpy(&c.outlier_bits,         (const int32_t *)dst->op_params + 7, sizeof(int32_t));
    memcpy(&c.outlier_count,        (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    memcpy(&c.outlier_packed_bytes, (const int32_t *)dst->op_params + 9, sizeof(int32_t));

    c.v_packed    = (v_scales_t != nullptr);
    c.has_outliers = (c.outlier_count > 0);
    c.nTokensQ    = (int)Q->ne[1];

    if (c.v_packed) {
        GGML_ASSERT(dst->src[3] != nullptr && "TQ K+V fused requires mask to determine nCells");
        c.nCells = (int)dst->src[3]->ne[0];
    } else {
        c.nCells = (int)V->ne[1];
    }

    if (c.has_outliers) {
        const int regular_count = D - c.outlier_count;
        const int raw = (regular_count * c.bits + 7) / 8;
        c.packedBytes = (raw + 3) & ~3;
    } else {
        c.packedBytes = (D * c.bits + 7) / 8;
    }
    c.v_packedBytes = c.v_packed ? ((D * c.v_bits + 7) / 8) : 0;
    c.nKVHeads = c.v_packed ? (int)V->ne[0] / c.v_packedBytes : (int)V->ne[2];

    c.v_scales_ptr   = v_scales_t   ? (const float *)v_scales_t->data   : nullptr;
    c.v_codebook_ptr = v_codebook_t ? (const float *)v_codebook_t->data : nullptr;
    c.zeros_ptr      = zeros_t      ? (const float *)zeros_t->data      : nullptr;

    c.outlier_packed_ptr  = ol_packed_t  ? (const uint8_t *)ol_packed_t->data  : nullptr;
    c.outlier_scales_ptr  = ol_scales_t  ? (const float   *)ol_scales_t->data  : nullptr;
    c.outlier_indices_ptr = ol_indices_t ? (const int16_t *)ol_indices_t->data : nullptr;
    c.outlier_zeros_ptr   = ol_zeros_t   ? (const float   *)ol_zeros_t->data   : nullptr;
    c.locs_ptr            = locs_t       ? (const int32_t *)locs_t->data       : nullptr;

    if (c.logit_softcap != 0.0f) { c.scale /= c.logit_softcap; }
    return c;
}

// Dispatch macro for a fixed DIM.  Expands to three tq_fattn_vec_launch
// instantiations (V_PACKED × HAS_OUTLIERS), each body small enough per TU.
#define TQ_DISPATCH_DIM(CTX, DIM, NCOLS, SOFTCAP) \
    if ((CTX).v_packed && (CTX).has_outliers) { \
        tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, true, true>(ctx, dst, (CTX).scale, (CTX).logit_softcap, \
            (CTX).bits, (CTX).firstCell, (CTX).nCells, (CTX).nKVHeads, (CTX).packedBytes, \
            (CTX).v_bits, (CTX).v_packedBytes, (CTX).v_scales_ptr, (CTX).v_codebook_ptr, \
            (CTX).zeros_ptr, (CTX).asymmetric, \
            (CTX).outlier_packed_ptr, (CTX).outlier_scales_ptr, (CTX).outlier_indices_ptr, (CTX).outlier_zeros_ptr, \
            (CTX).outlier_bits, (CTX).outlier_count, (CTX).outlier_packed_bytes, \
            (CTX).locs_ptr); \
    } else if ((CTX).v_packed) { \
        tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, true, false>(ctx, dst, (CTX).scale, (CTX).logit_softcap, \
            (CTX).bits, (CTX).firstCell, (CTX).nCells, (CTX).nKVHeads, (CTX).packedBytes, \
            (CTX).v_bits, (CTX).v_packedBytes, (CTX).v_scales_ptr, (CTX).v_codebook_ptr, \
            (CTX).zeros_ptr, (CTX).asymmetric, \
            nullptr, nullptr, nullptr, nullptr, 0, 0, 0, \
            (CTX).locs_ptr); \
    } else { \
        tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, false, false>(ctx, dst, (CTX).scale, (CTX).logit_softcap, \
            (CTX).bits, (CTX).firstCell, (CTX).nCells, (CTX).nKVHeads, (CTX).packedBytes, \
            0, 0, nullptr, nullptr, \
            (CTX).zeros_ptr, (CTX).asymmetric, \
            nullptr, nullptr, nullptr, nullptr, 0, 0, 0, \
            (CTX).locs_ptr); \
    }
