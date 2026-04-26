// SPDX-License-Identifier: MIT
// ze_buffer.hpp — RAII device-memory wrapper plus a size-bucketed buffer pool.
//
// Pool design (ADR-L0-004 §DSA):
//   - 23 buckets: bucket k covers allocations in range [2^(k+6), 2^(k+7))
//     i.e., bucket 0 = 64 B, bucket 22 = 256 MB.
//   - Per-bucket std::mutex for fine-grained locking (avoids a global lock).
//   - Allocation: round requested size up to the next power-of-two, take
//     the matching bucket index, pop from the free list. If empty, call
//     zeMemAllocDevice for a new allocation.
//   - Free: push back to the matching bucket's free list.
//   - O(1) amortised alloc/free; at most 2× space overhead per allocation.
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

// Forward-declared L0 types.
struct _ze_context_handle_t;
struct _ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_device_handle_t  *ze_device_handle_t;

// Function-pointer typedefs resolved at runtime.
typedef int32_t (*PFN_zeMemAllocDevice)(
    ze_context_handle_t, const void *, size_t, size_t, ze_device_handle_t, void **);
typedef int32_t (*PFN_zeMemFree)(ze_context_handle_t, void *);

// Number of power-of-two size buckets.  Bucket 0 = 64 B, bucket 22 = 256 MB.
static constexpr uint32_t ZE_BUF_BUCKET_COUNT = 23u;
// Base size for bucket 0 (= 2^6 bytes).
static constexpr uint64_t ZE_BUF_BASE_BYTES   = 64u;

/**
 * ZeBuffer — RAII owner of a zeMemAllocDevice allocation.
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
 * ZeBufferPool — size-bucketed free list for device-memory allocations.
 *
 * One instance per active ZeDevice (created in ze_ollama_device_open).
 * Destructor frees all pooled allocations via zeMemFree.
 */
class ZeBufferPool {
public:
    ZeBufferPool() noexcept : ctx_(nullptr), dev_(nullptr), fn_alloc_(nullptr), fn_free_(nullptr) {}

    ZeBufferPool(ze_context_handle_t ctx,
                 ze_device_handle_t  dev,
                 PFN_zeMemAllocDevice fn_alloc,
                 PFN_zeMemFree        fn_free) noexcept
        : ctx_(ctx), dev_(dev), fn_alloc_(fn_alloc), fn_free_(fn_free) {}

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
     * Allocate at least `bytes` of device memory.
     * Rounds up to the next power-of-two bucket; pops from the free list or
     * calls zeMemAllocDevice.
     * Returns nullptr on failure (zeMemAllocDevice error).
     * Device allocations are 256-byte aligned per L0 specification.
     */
    void *alloc(size_t bytes) noexcept {
        if (!fn_alloc_ || !ctx_ || !dev_) return nullptr;
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

        // Descriptor for device-local, uncached allocation.
        // Using a plain uint8_t array to avoid including ze_api.h here.
        // Layout must match ze_device_mem_alloc_desc_t {stype, pNext, flags, ordinal}.
        uint32_t desc[4] = {
            0x0003u, // ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
            0u, 0u, 0u
        };

        void   *ptr    = nullptr;
        int32_t result = fn_alloc_(
            ctx_,
            reinterpret_cast<const void *>(desc),
            alloc_size,
            256u, // 256-byte alignment (L0 device memory requirement)
            dev_,
            &ptr);
        return (result == 0) ? ptr : nullptr;
    }

    /**
     * Return a device-memory pointer to the pool.
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

    /**
     * Byte size of allocations stored in bucket b.
     */
    static size_t bucket_size(uint32_t b) noexcept {
        return static_cast<size_t>(ZE_BUF_BASE_BYTES) << b;
    }

    ze_context_handle_t  ctx_;
    ze_device_handle_t   dev_;
    PFN_zeMemAllocDevice fn_alloc_;
    PFN_zeMemFree        fn_free_;

    std::array<std::vector<void *>, ZE_BUF_BUCKET_COUNT> buckets_;
    std::array<std::mutex, ZE_BUF_BUCKET_COUNT>          bucket_mutex_;
    std::atomic<uint64_t>                                 tracked_live_bytes_{0u};
};
