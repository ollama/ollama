// SPDX-License-Identifier: MIT
// ze_buffer.hpp — RAII device-memory wrapper plus a size-bucketed buffer pool,
//                 and push-constant struct definitions for all Level Zero kernels.
//
// Push-constant structs (ADR-L0-001 Section 3.4):
//   ze_rope_pc     — Family 2: rope_f32 / rope_f16         (112 B, 8-byte aligned)
//   ze_rms_norm_pc — Family 3: rms_norm_f32 / rms_norm_f16  (88 B, 8-byte aligned)
//   ze_softmax_pc  — Family 4: softmax_f32                 (128 B, 8-byte aligned)
//   ze_binop_pc    — Family 5: add_f32 / add_f16 / mul_f32 (144 B, 8-byte aligned)
//
// Note: mul_mat_pc (Family 1, 160 B) is defined by embedded-firmware-engineer (Group C1)
// in ggml-level-zero.cpp lines 370-450. It is NOT defined here to avoid overlap per
// AC-7 file-ownership rule.
//
// Alignment rules per ADR-L0-001 Section 3.4:
//   All structs must be aligned to their largest scalar member (8 bytes for int64_t).
//   Total size must be a multiple of 8 bytes (the struct alignment).
//   All sizes verified against the ADR contract values above.
//
// Pool design (ADR-L0-004 §DSA):
//   - 23 buckets: bucket k covers allocations in range [2^(k+6), 2^(k+7))
//     i.e., bucket 0 = 64 B, bucket 22 = 256 MB.
//   - Per-bucket std::mutex for fine-grained locking (avoids a global lock).
//   - Allocation: round requested size up to the next power-of-two, take
//     the matching bucket index, pop from the free list. If empty, call
//     zeMemAllocHost for a new allocation.
//   - Free: push back to the matching bucket's free list.
//   - O(1) amortised alloc/free; at most 2× space overhead per allocation.
//
// Memory type: zeMemAllocHost (host-pinned).
//   - CPU can write directly (memcpy in set_tensor).
//   - GPU can read/write without explicit cache coherency barriers.
//   - No H2D staging copy needed; set_tensor is a plain memcpy.
//   - Slower than zeMemAllocDevice for GPU-only access, but correct on
//     Intel Arc (PCIe dGPU, Windows) where zeMemAllocDevice requires a
//     DMA-capable copy path that was unavailable from mmap source.
//
// Thread-safety: each bucket is independently protected by its own mutex.
// Concurrent allocs/frees in different size classes contend only on their
// respective mutex.
#pragma once

#include <array>
#include <vector>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <utility>
#include <bit>

// =============================================================================
// Push-constant structs for Group C2 kernels (ADR-L0-001 Section 3.4)
// =============================================================================
// All fields use int32_t (4 B) or int64_t (8 B) or float (4 B) — strictly POD.
// The dispatcher stack-allocates each struct, fills fields from the ggml_tensor,
// and passes it via zeKernelSetArgumentValue at argument index 0.
// =============================================================================

/**
 * ze_rope_pc — Family 2 push-constant for rope_f32 and rope_f16 kernels.
 *
 * Layout (112 bytes total, 8-byte aligned):
 *   ne[4]        : 4 x int32  = 16 B  — element counts per dimension
 *   nb_x[4]      : 4 x int64  = 32 B  — input (src0) byte strides
 *   nb_y[4]      : 4 x int64  = 32 B  — output (dst) byte strides
 *   freq_base    : float       =  4 B  — RoPE theta base (e.g. 10000.0 or 500000.0)
 *   freq_scale   : float       =  4 B  — frequency scale factor (usually 1.0)
 *   attn_factor  : float       =  4 B  — YaRN attention factor
 *   beta_fast    : float       =  4 B  — YaRN ramp upper boundary (fast heads)
 *   beta_slow    : float       =  4 B  — YaRN ramp lower boundary (slow heads)
 *   n_ctx_orig   : int32       =  4 B  — original context length for YaRN
 *   mode         : int32       =  4 B  — 0 = neox-style, 2 = interleaved
 *   n_dims       : int32       =  4 B  — number of rotary dimensions
 *
 * pos[] is bound as a separate int32 device buffer at argument index 2 per
 * ADR Section 3.4 (pos size is not known at compile time).
 *
 * Total: 16+32+32+4+4+4+4+4+4+4+4 = 112 B.  8-byte aligned (no padding needed).
 */
struct ze_rope_pc {
    int32_t  ne[4];        // element counts [ne0, ne1, ne2, ne3]   16 B
    int64_t  nb_x[4];      // src0 byte strides                      32 B
    int64_t  nb_y[4];      // dst byte strides                       32 B
    float    freq_base;    // theta base                              4 B
    float    freq_scale;   // frequency scale                         4 B
    float    attn_factor;  // YaRN attenuation factor                 4 B
    float    beta_fast;    // YaRN ramp upper                         4 B
    float    beta_slow;    // YaRN ramp lower                         4 B
    int32_t  n_ctx_orig;   // original context length                 4 B
    int32_t  mode;         // rotation mode (0=neox, 2=interleaved)   4 B
    int32_t  n_dims;       // rotary dimension count                  4 B
    // Total: 112 B — no trailing padding required
};
static_assert(sizeof(ze_rope_pc) == 112, "ze_rope_pc must be 112 bytes per ADR-L0-001 Section 3.4");
static_assert(alignof(ze_rope_pc) >= 4,  "ze_rope_pc must be at least 4-byte aligned");

/**
 * ze_rms_norm_pc — Family 3 push-constant for rms_norm_f32 and rms_norm_f16.
 *
 * Layout (88 bytes total, 8-byte aligned):
 *   ne[4]   : 4 x int32  = 16 B  — element counts per dimension
 *   nb_x[4] : 4 x int64  = 32 B  — input byte strides
 *   nb_y[4] : 4 x int64  = 32 B  — output byte strides
 *   eps     : float       =  4 B  — stability epsilon (Bug #10 invariant: this is
 *                                   the ONLY per-op param; NO weight field)
 *   _pad    : int32       =  4 B  — 4-byte pad to maintain 8-byte struct alignment
 *
 * Bug #10 invariant (commit 32f6fac9): no weight argument in any rms_norm kernel
 * signature.  The learnable gamma is applied by a downstream GGML_OP_MUL node.
 * grep "weight" rms_norm.cl MUST return zero matches.
 *
 * Total: 16+32+32+4+4 = 88 B.  8-byte aligned.
 */
struct ze_rms_norm_pc {
    int32_t  ne[4];    // element counts [ne0, ne1, ne2, ne3]   16 B
    int64_t  nb_x[4];  // src0 byte strides                      32 B
    int64_t  nb_y[4];  // dst byte strides                       32 B
    float    eps;      // RMS stability epsilon                    4 B
    int32_t  _pad;     // padding to align struct to 8 bytes       4 B
    // Total: 88 B
};
static_assert(sizeof(ze_rms_norm_pc) == 88, "ze_rms_norm_pc must be 88 bytes per ADR-L0-001 Section 3.4");
static_assert(alignof(ze_rms_norm_pc) >= 4, "ze_rms_norm_pc must be at least 4-byte aligned");

/**
 * ze_softmax_pc — Family 4 push-constant for softmax_f32.
 *
 * Layout (128 bytes total, 8-byte aligned):
 *   ne[4]      : 4 x int32  = 16 B  — element counts [ne0, ne1, ne2, ne3]
 *   nb_x[4]    : 4 x int64  = 32 B  — input byte strides
 *   nb_y[4]    : 4 x int64  = 32 B  — output byte strides
 *   nb_mask[4] : 4 x int64  = 32 B  — mask byte strides (zeroed when has_mask=0)
 *   scale      : float       =  4 B  — pre-scale (typically 1/sqrt(d_k))
 *   max_bias   : float       =  4 B  — ALiBi max bias (0.0 if no ALiBi)
 *   has_mask   : int32       =  4 B  — 1 if mask buffer is valid, else 0
 *   has_alibi  : int32       =  4 B  — 1 if ALiBi slope is applied, else 0
 *
 * Total: 16+32+32+32+4+4+4+4 = 128 B.  8-byte aligned (no trailing padding).
 */
struct ze_softmax_pc {
    int32_t  ne[4];       // element counts [ne0, ne1, ne2, ne3]    16 B
    int64_t  nb_x[4];     // input byte strides                      32 B
    int64_t  nb_y[4];     // output byte strides                     32 B
    int64_t  nb_mask[4];  // mask byte strides (zero = no mask)      32 B
    float    scale;       // input scaling factor                      4 B
    float    max_bias;    // ALiBi slope max (0 = no ALiBi)            4 B
    int32_t  has_mask;    // 0 or 1                                    4 B
    int32_t  has_alibi;   // 0 or 1                                    4 B
    // Total: 128 B — no trailing padding required
};
static_assert(sizeof(ze_softmax_pc) == 128, "ze_softmax_pc must be 128 bytes per ADR-L0-001 Section 3.4");
static_assert(alignof(ze_softmax_pc) >= 4,  "ze_softmax_pc must be at least 4-byte aligned");

/**
 * ze_binop_pc — Family 5 push-constant for add_f32, add_f16, mul_f32, mul_f16.
 *
 * Layout (144 bytes total, 8-byte aligned):
 *   ne_a[4] : 4 x int32  = 16 B  — src0 element counts
 *   ne_b[4] : 4 x int32  = 16 B  — src1 element counts
 *   ne_d[4] : 4 x int32  = 16 B  — dst element counts (== ne_a for these ops)
 *   nb_a[4] : 4 x int64  = 32 B  — src0 byte strides
 *   nb_b[4] : 4 x int64  = 32 B  — src1 byte strides (0 on broadcast dimensions)
 *   nb_d[4] : 4 x int64  = 32 B  — dst byte strides
 *
 * Broadcast encoding (Null Object pattern per ADR Section 8):
 *   Setting nb_b[k] = 0 for dimension k causes the IDX macro to produce address 0
 *   for that axis, effectively broadcasting src1 across that dimension.
 *   No explicit branch is needed in the kernel body — zero stride is transparent.
 *
 * Total: 16+16+16+32+32+32 = 144 B.  8-byte aligned (no trailing padding).
 */
struct ze_binop_pc {
    int32_t  ne_a[4];  // src0 element counts                    16 B
    int32_t  ne_b[4];  // src1 element counts                    16 B
    int32_t  ne_d[4];  // dst  element counts (== ne_a)          16 B
    int64_t  nb_a[4];  // src0 byte strides                      32 B
    int64_t  nb_b[4];  // src1 byte strides (0 = broadcast dim)  32 B
    int64_t  nb_d[4];  // dst  byte strides                      32 B
    // Total: 144 B — no trailing padding required
};
static_assert(sizeof(ze_binop_pc) == 144, "ze_binop_pc must be 144 bytes per ADR-L0-001 Section 3.4");
static_assert(alignof(ze_binop_pc) >= 4,  "ze_binop_pc must be at least 4-byte aligned");

// Forward-declared L0 types.
struct _ze_context_handle_t;
struct _ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_device_handle_t  *ze_device_handle_t;

// Function-pointer typedefs resolved at runtime.
typedef int32_t (*PFN_zeMemAllocDevice)(
    ze_context_handle_t, const void *, size_t, size_t, ze_device_handle_t, void **);
// zeMemAllocShared: (ctx, device_desc, host_desc, size, align, device, ptr)
// Used for UMA shared allocations accessible from both CPU and GPU without
// an explicit H2D DMA copy (Intel Arc unified-memory model).
typedef int32_t (*PFN_zeMemAllocShared)(
    ze_context_handle_t, const void *, const void *, size_t, size_t, ze_device_handle_t, void **);
// zeMemAllocHost: (ctx, host_desc, size, align, ptr)
// Pinned host memory — always coherent between CPU and GPU; GPU accesses via IOMMU.
typedef int32_t (*PFN_zeMemAllocHost)(ze_context_handle_t, const void *, size_t, size_t, void **);
typedef int32_t (*PFN_zeMemFree)(ze_context_handle_t, void *);

// Number of power-of-two size buckets.  Bucket 0 = 64 B, bucket 22 = 256 MB.
static constexpr uint32_t ZE_BUF_BUCKET_COUNT = 23u;
// Base size for bucket 0 (= 2^6 bytes).
static constexpr uint64_t ZE_BUF_BASE_BYTES   = 64u;

/**
 * ZeBuffer — RAII owner of a zeMemAlloc* allocation.
 * Freed via zeMemFree in the destructor.
 */
class ZeBuffer {
public:
    ZeBuffer() noexcept : ptr_(nullptr), size_(0), ctx_(nullptr), fn_free_(nullptr) {}

    ZeBuffer(void *ptr, size_t size,
             ze_context_handle_t ctx, PFN_zeMemFree fn_free) noexcept
        : ptr_(ptr), size_(size), ctx_(ctx), fn_free_(fn_free) {}

    ZeBuffer(const ZeBuffer &)            = delete;
    ZeBuffer &operator=(const ZeBuffer &) = delete;

    ZeBuffer(ZeBuffer &&o) noexcept
        : ptr_(o.ptr_), size_(o.size_), ctx_(o.ctx_), fn_free_(o.fn_free_) {
        o.ptr_ = nullptr;
    }
    ZeBuffer &operator=(ZeBuffer &&o) noexcept {
        if (this != &o) {
            release();
            ptr_     = o.ptr_;
            size_    = o.size_;
            ctx_     = o.ctx_;
            fn_free_ = o.fn_free_;
            o.ptr_   = nullptr;
        }
        return *this;
    }

    ~ZeBuffer() noexcept { release(); }

    void  *data()  const noexcept { return ptr_;  }
    size_t size()  const noexcept { return size_; }
    bool   valid() const noexcept { return ptr_ != nullptr; }

    /**
     * Relinquish ownership without freeing.
     * Used when the pool takes the pointer back to its free list.
     */
    void *disown() noexcept {
        void *p = ptr_;
        ptr_    = nullptr;
        return p;
    }

private:
    void release() noexcept {
        if (ptr_ && fn_free_ && ctx_) {
            fn_free_(ctx_, ptr_);
            ptr_ = nullptr;
        }
    }

    void                *ptr_;
    size_t               size_;
    ze_context_handle_t  ctx_;
    PFN_zeMemFree        fn_free_;
};

/**
 * ZeBufferPool — size-bucketed free list for host-pinned memory allocations.
 *
 * Uses zeMemAllocHost so allocations are accessible from both CPU and GPU
 * without cache-coherency barriers.  set_tensor can write via plain memcpy
 * and GPU kernels read the data correctly via IOMMU.
 *
 * One instance per active ZeDevice (created in ze_ollama_device_open).
 * Destructor frees all pooled allocations via zeMemFree.
 */
class ZeBufferPool {
public:
    ZeBufferPool() noexcept : ctx_(nullptr), fn_alloc_host_(nullptr), fn_free_(nullptr) {}

    ZeBufferPool(ze_context_handle_t ctx,
                 ze_device_handle_t  /*dev_unused*/,
                 PFN_zeMemAllocHost   fn_alloc_host,
                 PFN_zeMemFree        fn_free) noexcept
        : ctx_(ctx), fn_alloc_host_(fn_alloc_host), fn_free_(fn_free) {}

    ZeBufferPool(const ZeBufferPool &)            = delete;
    ZeBufferPool &operator=(const ZeBufferPool &) = delete;
    ZeBufferPool(ZeBufferPool &&)                 = delete;
    ZeBufferPool &operator=(ZeBufferPool &&)      = delete;

    ~ZeBufferPool() noexcept {
        // Free all pooled allocations.
        for (uint32_t b = 0; b < ZE_BUF_BUCKET_COUNT; ++b) {
            std::lock_guard<std::mutex> lk(bucket_mutex_[b]);
            for (void *p : buckets_[b]) {
                if (p && fn_free_ && ctx_) {
                    fn_free_(ctx_, p);
                }
            }
            buckets_[b].clear();
        }
    }

    /**
     * Allocate at least `bytes` of host-pinned memory (zeMemAllocHost).
     * Rounds up to the next power-of-two bucket; pops from the free list or
     * allocates fresh.  Returns nullptr on failure.
     */
    void *alloc(size_t bytes) noexcept {
        if (!fn_alloc_host_ || !ctx_) return nullptr;
        uint32_t b          = bucket_index(bytes);
        size_t   alloc_size = bucket_size(b);

        {
            std::lock_guard<std::mutex> lk(bucket_mutex_[b]);
            if (!buckets_[b].empty()) {
                void *p = buckets_[b].back();
                buckets_[b].pop_back();
                return p;
            }
        }

        // Host-pinned allocation descriptor.
        // Layout matches ze_host_mem_alloc_desc_t {stype, pNext, flags}.
        uint32_t host_desc[3] = { 0x0016u, 0u, 0u };

        void   *ptr    = nullptr;
        int32_t result = fn_alloc_host_(
            ctx_,
            reinterpret_cast<const void *>(host_desc),
            alloc_size,
            256u,
            &ptr);
        return (result == 0) ? ptr : nullptr;
    }

    /**
     * Return a host-pinned pointer to the pool.
     * The pointer must have been obtained from alloc() on this pool instance.
     */
    void free(void *ptr, size_t bytes) noexcept {
        if (!ptr) return;
        uint32_t b = bucket_index(bytes);
        std::lock_guard<std::mutex> lk(bucket_mutex_[b]);
        buckets_[b].push_back(ptr);
    }

    /**
     * Query total tracked live allocation size (approximate).
     * For scheduler free-memory reporting.
     */
    uint64_t tracked_bytes() const noexcept {
        return tracked_live_bytes_.load(std::memory_order_relaxed);
    }

    void track_alloc(size_t n) noexcept {
        tracked_live_bytes_.fetch_add(n, std::memory_order_relaxed);
    }
    void track_free(size_t n) noexcept {
        tracked_live_bytes_.fetch_sub(n, std::memory_order_relaxed);
    }

private:
    /**
     * Compute the bucket index for a requested byte size.
     * bucket 0 = 64 B, bucket 22 = 256 MB.
     * Sizes above 256 MB go into bucket 22 (clamped).
     */
    static uint32_t bucket_index(size_t bytes) noexcept {
        if (bytes <= ZE_BUF_BASE_BYTES) return 0u;
        // Next power of two >= bytes.
        size_t   pow2 = std::bit_ceil(bytes);
        // log2(pow2) - log2(ZE_BUF_BASE_BYTES) = index.
        uint32_t idx  = static_cast<uint32_t>(
            std::countr_zero(static_cast<unsigned long long>(pow2)) -
            std::countr_zero(static_cast<unsigned long long>(ZE_BUF_BASE_BYTES)));
        return (idx < ZE_BUF_BUCKET_COUNT) ? idx : (ZE_BUF_BUCKET_COUNT - 1u);
    }

    static size_t bucket_size(uint32_t b) noexcept {
        return static_cast<size_t>(ZE_BUF_BASE_BYTES) << b;
    }

    ze_context_handle_t  ctx_;
    PFN_zeMemAllocHost   fn_alloc_host_;
    PFN_zeMemFree        fn_free_;

    std::array<std::vector<void *>, ZE_BUF_BUCKET_COUNT> buckets_;
    std::array<std::mutex,          ZE_BUF_BUCKET_COUNT> bucket_mutex_;
    std::atomic<uint64_t>                                 tracked_live_bytes_{0};
};
