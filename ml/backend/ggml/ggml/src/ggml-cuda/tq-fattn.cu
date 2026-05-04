#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

// Launch the TQ fused flash-attention kernel for a given (D, ncols, use_logit_softcap, V_PACKED, HAS_OUTLIERS).
template<int D, int ncols, bool use_logit_softcap, bool V_PACKED, bool HAS_OUTLIERS>
static void tq_fattn_vec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                float scale, float logit_softcap,
                                int bits, int firstCell, int nCells, int nKVHeads, int packedBytes,
                                int v_bits, int v_packedBytes,
                                const float * v_scales_ptr, const float * v_codebook_ptr,
                                const float   * zeros_ptr,
                                int             asymmetric,
                                const uint8_t * qjl_packed_ptr,
                                const float   * qjl_norm_ptr,
                                const float   * qjl_projection_ptr,
                                int             qjl_rows,
                                int             qjl_packedBytes,
                                const uint8_t * outlier_packed_ptr,
                                const float   * outlier_scales_ptr,
                                const int8_t  * outlier_indices_ptr,
                                const float   * outlier_zeros_ptr,
                                int             outlier_bits,
                                int             outlier_count,
                                int             outlier_packed_bytes)
{
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_p  = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * scales   = dst->src[4];
    const ggml_tensor * codebook = dst->src[5];

    GGML_ASSERT(Q->ne[0] == D);
    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    if constexpr (!V_PACKED) {
        GGML_ASSERT(V->type  == GGML_TYPE_F16);
    } else {
        GGML_ASSERT(V->type  == GGML_TYPE_I8);
    }

    const int nTokensQ = (int)Q->ne[1];
    const int nHeadsQ  = (int)Q->ne[2];
    const int nSeq     = (int)Q->ne[3];

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    const uint3 ne01 = init_fastdiv_values((uint64_t)nTokensQ);

    const int ntiles_x = (nTokensQ + ncols - 1) / ncols;
    dim3 threads(WARP_SIZE, 4);

    // V strides: only used by !V_PACKED path; pass V strides for the f16 case.
    // For the V_PACKED case these are passed but ignored by the kernel.
    // Per-block smem layout (see tq-fattn-vec.cuh:431-434):
    //   [0, max(ne_KQ, ne_combine))  KQ + combine workspace
    //   [..., +ncols*D)              s_Q_fixed (pre-scaled Q tile)
    //   [..., +ncols*256)            s_dot_q_fixed (QJL projection scratch)
    // ne_KQ      = ncols * D                                    (Q·K result tile)
    // ne_combine = nwarps * V_cols_per_iter * D = 16 * D        (warp-VKQ reduce)
    constexpr int ne_KQ      = ncols * D;
    constexpr int ne_combine = 16 * D;  // nwarps(4) * V_cols_per_iter(4) * D
    constexpr size_t smem    = ((ne_KQ > ne_combine ? ne_KQ : ne_combine)
                                + ncols * D
                                + ncols * 256) * sizeof(float);

    // ---- Multi-block KV-split heuristic ----
    // The TQ inline-decode kernel previously launched a fixed
    // `(ntiles_x, 1, nHeadsQ*nSeq)` grid: at decode (Q=1, ntiles_x=1) the
    // grid only has nHeadsQ*nSeq blocks (e.g. 24 for llama3.2:3b), each
    // walking the entire K/V context serially. That leaves SMs idle and
    // makes decode O(nCells) per block. Adding a Y dimension (parallel_blocks)
    // splits K cells across additional blocks via interleaved striping;
    // flash_attn_combine_results merges the per-block partial VKQ + (max,sum)
    // meta into the final result. This mirrors the pattern stock CUDA FA
    // uses (fattn-common.cuh:904-955).
    //
    // Workspace (transient, per layer-decode):
    //   dst_tmp      = parallel_blocks * D * nHeadsQ * nTokensQ * nSeq floats
    //   dst_tmp_meta = parallel_blocks * nHeadsQ * nTokensQ * nSeq float2s
    // For llama3.2:3b decode at ctx=32k with parallel_blocks=8 this is
    // ~96 KiB total — three orders of magnitude below the model+KV footprint.
    constexpr int nthreads = 128;
    const int ntiles_total = ntiles_x * nHeadsQ * nSeq;
    const int ntiles_KQ = std::max(1, (nCells + nthreads - 1) / nthreads);

    auto kernel_ptr = tq_flash_attn_ext_vec<D, ncols, use_logit_softcap, V_PACKED, HAS_OUTLIERS>;

    int max_blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, kernel_ptr,
        (int)(threads.x * threads.y * threads.z), smem));
    GGML_ASSERT(max_blocks_per_sm > 0);

    const int dev_id = ggml_cuda_get_device();
    const int nsm    = ggml_cuda_info().devices[dev_id].nsm;
    const int blocks_per_wave = nsm * max_blocks_per_sm;

    // Default to single-block (no workspace, no combine kernel). Only consider
    // multi-block KV-split when the single-block layout obviously underutilises
    // the GPU (less than one full wave of blocks). Prefill — where ntiles_total
    // is already nTokensQ/ncols * nHeadsQ ≥ blocks_per_wave — therefore stays
    // on the original single-kernel path and incurs zero extra VRAM. Decode
    // (Q=1, ntiles_total = nHeadsQ*nSeq, often dozens of blocks) takes the
    // multi-block path with ~96 KiB transient workspace.
    int parallel_blocks = 1;
    if (ntiles_total < blocks_per_wave && ntiles_KQ > 1) {
        // Initial efficiency at parallel_blocks=1.
        int nwaves_best = 1;
        int eff_best    = (100 * ntiles_total) / blocks_per_wave;
        // Refine upward, mirroring fattn-common.cuh:929-948.
        for (int test = std::min(max_blocks_per_sm, ntiles_KQ); test <= ntiles_KQ; ++test) {
            const int nblocks_total = ntiles_total * test;
            const int nwaves        = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int eff_pct       = nwaves > 0 ? (100 * nblocks_total) / (nwaves * blocks_per_wave) : 0;
            if (eff_best >= 95 && nwaves > nwaves_best) {
                break;
            }
            if (eff_pct > eff_best) {
                nwaves_best     = nwaves;
                eff_best        = eff_pct;
                parallel_blocks = test;
            }
        }
    }

    // ---- Allocate workspace if multi-block ----
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);
    if (parallel_blocks > 1) {
        const size_t kqv_n    = (size_t)D * nHeadsQ * nTokensQ * nSeq;
        const size_t kqv_rows = (size_t)nHeadsQ * nTokensQ * nSeq;
        dst_tmp.alloc((size_t)parallel_blocks * kqv_n);
        dst_tmp_meta.alloc((size_t)parallel_blocks * kqv_rows);
    }

    dim3 blocks(ntiles_x, parallel_blocks, nHeadsQ * nSeq);

    float  * kernel_dst      = (parallel_blocks > 1) ? dst_tmp.ptr      : (float  *)dst->data;
    float2 * kernel_dst_meta = (parallel_blocks > 1) ? dst_tmp_meta.ptr : (float2 *)nullptr;

    kernel_ptr<<<blocks, threads, smem, ctx.stream()>>>(
        (const char    *)Q->data,
        (const uint8_t *)K_p->data,
        (const char    *)V->data,
        mask ? (const char *)mask->data : nullptr,
        kernel_dst,
        kernel_dst_meta,
        (const float *)scales->data,
        (const float *)codebook->data,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        (int32_t)Q->ne[0],
        ne01,
        (int32_t)Q->ne[2],
        (int32_t)Q->ne[3],
        (int32_t)Q->nb[1],
        (int32_t)Q->nb[2],
        (int64_t)Q->nb[3],
        (int32_t)V->nb[1],
        (int32_t)V->nb[2],
        (int64_t)V->nb[3],
        mask ? (int32_t)mask->ne[0] : 0,   // nCells (mask row width)
        mask ? (int32_t)mask->nb[1] : 0,   // bytes between token rows
        v_scales_ptr, v_codebook_ptr, v_bits, v_packedBytes,
        zeros_ptr, asymmetric,
        qjl_packed_ptr, qjl_norm_ptr, qjl_projection_ptr, qjl_rows, qjl_packedBytes,
        outlier_packed_ptr, outlier_scales_ptr, outlier_indices_ptr, outlier_zeros_ptr,
        outlier_bits, outlier_count, outlier_packed_bytes
    );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        // Combine per-block partials into final output. Block grid mirrors
        // the FA output's logical shape (col, head, seq); each block reduces
        // parallel_blocks slots into one D-wide row of the final dst.
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine((unsigned)nTokensQ, (unsigned)nHeadsQ, (unsigned)nSeq);
        const size_t nbytes_shared_combine = (size_t)parallel_blocks * sizeof(float2);
        flash_attn_combine_results<D>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, ctx.stream()>>>(
                dst_tmp.ptr, dst_tmp_meta.ptr, (float *)dst->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

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
    const int8_t  * outlier_indices_ptr = outlier_indices_t ? (const int8_t  *)outlier_indices_t->data : nullptr;
    const float   * outlier_zeros_ptr   = outlier_zeros_t   ? (const float   *)outlier_zeros_t->data   : nullptr;

    GGML_ASSERT((D == 64 || D == 128 || D == 256) && "TurboQuant fused kernel: unsupported head_dim (need 64, 128, or 256)");

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
    else                    ncols = 8;

    const bool has_outliers = (outlier_count > 0);

    // Outlier dual-stream decode (HAS_OUTLIERS=true) only instantiated for the
    // V_PACKED asymmetric D=128 combination. D=64 routes to HAS_OUTLIERS=false.
    #define DISPATCH(DIM, NCOLS, SOFTCAP) \
        if (v_packed && has_outliers && DIM == 128) { \
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

    if (D == 64) {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(64, false); }
        else                       { DISPATCH_NCOLS(64, true);  }
    } else if (D == 256) {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(256, false); }
        else                       { DISPATCH_NCOLS(256, true);  }
    } else {
        if (logit_softcap == 0.0f) { DISPATCH_NCOLS(128, false); }
        else                       { DISPATCH_NCOLS(128, true);  }
    }
    #undef DISPATCH_NCOLS
    #undef DISPATCH

    CUDA_CHECK(cudaGetLastError());
}
