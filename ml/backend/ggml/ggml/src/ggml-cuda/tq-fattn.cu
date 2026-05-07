#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant fused flash attention requires compute capability 6.0+");
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * v_scales_t   = dst->src[6];  // NULL for K-only fused
    const ggml_tensor * v_codebook_t = dst->src[7];
    const ggml_tensor * zeros_t          = dst->src[8];   // NULL if symmetric
    const ggml_tensor * qjl_packed_t     = dst->src[9];   // NULL if no QJL
    const ggml_tensor * qjl_norm_t       = dst->src[10];
    const ggml_tensor * qjl_projection_t = dst->src[11];
    const ggml_tensor * outlier_packed_t  = dst->src[12]; // NULL if no outliers
    const ggml_tensor * outlier_scales_t  = dst->src[13];
    const ggml_tensor * outlier_indices_t = dst->src[14];
    const ggml_tensor * outlier_zeros_t   = dst->src[15]; // NULL if no outliers or !asymmetric

    float   scale;
    float   logit_softcap;
    int32_t bits;
    int32_t firstCell;
    int32_t v_bits;
    int32_t qjl_rows;
    int32_t asymmetric;
    int32_t outlier_bits;
    int32_t outlier_count;
    int32_t outlier_packed_bytes;
    memcpy(&scale,                (const float   *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap,        (const float   *)dst->op_params + 1, sizeof(float));
    memcpy(&bits,                 (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&firstCell,            (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    memcpy(&v_bits,               (const int32_t *)dst->op_params + 4, sizeof(int32_t));
    memcpy(&qjl_rows,             (const int32_t *)dst->op_params + 5, sizeof(int32_t));
    memcpy(&asymmetric,           (const int32_t *)dst->op_params + 6, sizeof(int32_t));
    memcpy(&outlier_bits,         (const int32_t *)dst->op_params + 7, sizeof(int32_t));
    memcpy(&outlier_count,        (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    memcpy(&outlier_packed_bytes, (const int32_t *)dst->op_params + 9, sizeof(int32_t));

    const int D         = (int)Q->ne[0];
    const int nTokensQ  = (int)Q->ne[1];
    // nCells: when V_PACKED, V is the raw buffer so ne[1] is capacity not nCells.
    // Compute nCells from the mask if available, else from V geometry.
    const bool v_packed = (v_scales_t != nullptr);

    int nCells;
    if (v_packed) {
        // V is packed i8 buffer: nCells is not directly in V->ne[].
        // The mask always carries nCells; require mask when V is packed.
        GGML_ASSERT(dst->src[3] != nullptr && "TQ K+V fused requires mask to determine nCells");
        nCells = (int)dst->src[3]->ne[0];
    } else {
        nCells = (int)V->ne[1]; // after SDPA permute: [D, nCells, nKVHeads]
    }

    // packedBytes for K: when outliers are active, the regular sub-block holds
    // (D - outlier_count) codes at `bits` each, padded to 4-byte multiple.
    // Must match regularPackedBytes() in turboquant_compressed.go AND the
    // stride computation in tq-encode.cu / tq-dequant.cu.
    int packedBytes;
    if (outlier_count > 0) {
        const int regular_count = D - outlier_count;
        const int raw = (regular_count * bits + 7) / 8;
        packedBytes = (raw + 3) & ~3;
    } else {
        packedBytes = (D * bits + 7) / 8;
    }
    // packedBytes for V (computed before nKVHeads so we can derive nKVHeads from packed V dims):
    const int v_packedBytes = v_packed ? ((D * v_bits + 7) / 8) : 0;
    // nKVHeads: for K-only fused, V is a permuted f16 tensor with ne[2]=nKVHeads.
    // For K+V fused, V is the raw packed i8 buffer with ne[0]=v_packedBytes*nKVHeads.
    const int nKVHeads  = v_packed ? (int)V->ne[0] / v_packedBytes : (int)V->ne[2];

    const float * v_scales_ptr   = v_scales_t   ? (const float *)v_scales_t->data   : nullptr;
    const float * v_codebook_ptr = v_codebook_t ? (const float *)v_codebook_t->data : nullptr;
    const float   * zeros_ptr          = zeros_t          ? (const float   *)zeros_t->data          : nullptr;
    const uint8_t * qjl_packed_ptr     = qjl_packed_t     ? (const uint8_t *)qjl_packed_t->data     : nullptr;
    const float   * qjl_norm_ptr       = qjl_norm_t       ? (const float   *)qjl_norm_t->data       : nullptr;
    const float   * qjl_projection_ptr = qjl_projection_t ? (const float   *)qjl_projection_t->data : nullptr;
    const int qjl_packed_raw   = (qjl_rows + 7) / 8;
    const int qjl_packedBytes  = (qjl_packed_raw + 3) & ~3;

    const uint8_t * outlier_packed_ptr  = outlier_packed_t  ? (const uint8_t *)outlier_packed_t->data  : nullptr;
    const float   * outlier_scales_ptr  = outlier_scales_t  ? (const float   *)outlier_scales_t->data  : nullptr;
    const int16_t * outlier_indices_ptr = outlier_indices_t ? (const int16_t *)outlier_indices_t->data : nullptr;
    const float   * outlier_zeros_ptr   = outlier_zeros_t   ? (const float   *)outlier_zeros_t->data   : nullptr;

    GGML_ASSERT((D == 64 || D == 128 || D == 256 || D == 512) && "TurboQuant fused kernel: unsupported head_dim (need 64, 128, 256, or 512)");

    if (logit_softcap != 0.0f) { scale /= logit_softcap; }

    // Q-tile size selection. Prefill amortises K+V decode across ncols Q-tokens;
    // bigger ncols = fewer K+V re-decodes for the same nTokensQ. ncols=2 is
    // the legacy size — for nTokensQ=2048 at ctx=16384 that means decoding all
    // nCells K+V values 1024 times per layer. ncols=8 cuts that 4×.
    //   nTokensQ == 1  → decode (ncols=1)
    //   nTokensQ < 8   → small batch, ncols=2 keeps register pressure low
    //   nTokensQ ≥ 8   → prefill, ncols=8 amortises decode
    int ncols;
    if (nTokensQ == 1)      ncols = 1;
    else if (nTokensQ < 8)  ncols = 2;
    else if (D == 512)    ncols = 2;  // D=512 prefill: ncols=8 smem=57KB > 48KB Pascal limit
    else                    ncols = 8;

    const bool has_outliers = (outlier_count > 0);

    // K-only outlier (V_PACKED=false, HAS_OUTLIERS=true) instantiations live in
    // tq-fattn-konly-outlier.cu to stay under the gas single-file size limit.
    if (!v_packed && has_outliers) {
        ggml_cuda_tq_flash_attn_ext_konly_outlier(ctx, dst);
        return;
    }

    // Outlier dual-stream decode (HAS_OUTLIERS=true) instantiated for
    // V_PACKED K+V presets at D=64/128/256/512.
    #define DISPATCH(DIM, NCOLS, SOFTCAP) \
        if (v_packed && has_outliers) { \
            tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, true, true>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                v_bits, v_packedBytes, v_scales_ptr, v_codebook_ptr, \
                zeros_ptr, asymmetric, \
                qjl_packed_ptr, qjl_norm_ptr, qjl_projection_ptr, qjl_rows, qjl_packedBytes, \
                outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr, \
                outlier_bits, outlier_count, outlier_packed_bytes); \
        } else if (v_packed) { \
            tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, true, false>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                v_bits, v_packedBytes, v_scales_ptr, v_codebook_ptr, \
                zeros_ptr, asymmetric, \
                qjl_packed_ptr, qjl_norm_ptr, qjl_projection_ptr, qjl_rows, qjl_packedBytes, \
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0); \
        } else { \
            tq_fattn_vec_launch<DIM, NCOLS, SOFTCAP, false, false>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                0, 0, nullptr, nullptr, \
                zeros_ptr, asymmetric, \
                qjl_packed_ptr, qjl_norm_ptr, qjl_projection_ptr, qjl_rows, qjl_packedBytes, \
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0); \
        }

    #define DISPATCH_NCOLS(DIM, SOFTCAP) \
        if      (ncols == 1) { DISPATCH(DIM, 1, SOFTCAP); } \
        else if (ncols == 2) { DISPATCH(DIM, 2, SOFTCAP); } \
        else                 { DISPATCH(DIM, 8, SOFTCAP); }

    #define DISPATCH_NCOLS_512(SOFTCAP) \
        if      (ncols == 1) { DISPATCH(512, 1, SOFTCAP); } \
        else                 { DISPATCH(512, 2, SOFTCAP); }

    if (D == 64) {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(64, false); }
        else                       { DISPATCH_NCOLS(64, true);  }
    } else if (D == 256) {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(256, false); }
        else                       { DISPATCH_NCOLS(256, true);  }
    } else if (D == 512) {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS_512(false); }
        else                       { DISPATCH_NCOLS_512(true);  }
    } else {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(128, false); }
        else                       { DISPATCH_NCOLS(128, true);  }
    }
    #undef DISPATCH_NCOLS_512
    #undef DISPATCH_NCOLS
    #undef DISPATCH

    CUDA_CHECK(cudaGetLastError());
}
