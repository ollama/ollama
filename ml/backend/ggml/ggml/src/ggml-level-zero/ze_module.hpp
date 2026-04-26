// SPDX-License-Identifier: MIT
// ze_module.hpp — RAII wrappers for ze_module_handle_t / ze_kernel_handle_t
// plus a 256-entry LRU kernel cache (Flyweight + LRU — ADR-L0-004 §DSA).
//
// Cache key = SHA-256( SPIR-V_IL_blob || device_UUID_bytes || entry_name_cstr
//                      || build_options_cstr ).
// Collision probability for 256-entry cache with 256-bit key: ~2^(-248).
// As a secondary guard, the full (device_uuid, entry_name) pair is also
// compared on a cache hit to catch the practically-impossible collision.
//
// SHA-256 implementation selection (set by CMakeLists.txt):
//   ZE_OLLAMA_SHA256_OPENSSL  — uses OpenSSL EVP_Digest (Linux default)
//   ZE_OLLAMA_SHA256_BCRYPT   — uses BCryptHash         (Windows default)
//   ZE_OLLAMA_SHA256_BUILTIN  — minimal public-domain fallback
#pragma once

#include <array>
#include <list>
#include <unordered_map>
#include <mutex>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <utility>
#include <functional>

// Forward-declared L0 types.
struct _ze_module_handle_t;
struct _ze_kernel_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;

// Function-pointer typedefs resolved at runtime.
typedef int32_t (*PFN_zeModuleDestroy)(ze_module_handle_t);
typedef int32_t (*PFN_zeKernelDestroy)(ze_kernel_handle_t);

// SHA-256 digest type: 32 bytes = 256 bits.
using SHA256Digest = std::array<uint8_t, 32>;

// Maximum number of cached kernel entries.
static constexpr size_t ZE_KERNEL_CACHE_CAP = 256u;

/**
 * ZeModule — RAII owner of a ze_module_handle_t.
 * Destructor calls zeModuleDestroy.
 */
class ZeModule {
public:
    ZeModule() noexcept : mod_(nullptr), fn_destroy_(nullptr) {}

    ZeModule(ze_module_handle_t m, PFN_zeModuleDestroy fn) noexcept
        : mod_(m), fn_destroy_(fn) {}

    ZeModule(const ZeModule &)            = delete;
    ZeModule &operator=(const ZeModule &) = delete;

    ZeModule(ZeModule &&o) noexcept : mod_(o.mod_), fn_destroy_(o.fn_destroy_) {
        o.mod_ = nullptr;
    }
    ZeModule &operator=(ZeModule &&o) noexcept {
        if (this != &o) {
            destroy_if_owned();
            mod_        = o.mod_;
            fn_destroy_ = o.fn_destroy_;
            o.mod_      = nullptr;
        }
        return *this;
    }
    ~ZeModule() noexcept { destroy_if_owned(); }

    ze_module_handle_t get()   const noexcept { return mod_; }
    bool               valid() const noexcept { return mod_ != nullptr; }

private:
    void destroy_if_owned() noexcept {
        if (mod_ && fn_destroy_) { fn_destroy_(mod_); mod_ = nullptr; }
    }
    ze_module_handle_t mod_;
    PFN_zeModuleDestroy fn_destroy_;
};

/**
 * ZeKernel — RAII owner of a ze_kernel_handle_t.
 * Destructor calls zeKernelDestroy.
 */
class ZeKernel {
public:
    ZeKernel() noexcept : kern_(nullptr), fn_destroy_(nullptr) {}

    ZeKernel(ze_kernel_handle_t k, PFN_zeKernelDestroy fn) noexcept
        : kern_(k), fn_destroy_(fn) {}

    ZeKernel(const ZeKernel &)            = delete;
    ZeKernel &operator=(const ZeKernel &) = delete;

    ZeKernel(ZeKernel &&o) noexcept : kern_(o.kern_), fn_destroy_(o.fn_destroy_) {
        o.kern_ = nullptr;
    }
    ZeKernel &operator=(ZeKernel &&o) noexcept {
        if (this != &o) {
            destroy_if_owned();
            kern_       = o.kern_;
            fn_destroy_ = o.fn_destroy_;
            o.kern_     = nullptr;
        }
        return *this;
    }
    ~ZeKernel() noexcept { destroy_if_owned(); }

    ze_kernel_handle_t get()   const noexcept { return kern_; }
    bool               valid() const noexcept { return kern_ != nullptr; }

private:
    void destroy_if_owned() noexcept {
        if (kern_ && fn_destroy_) { fn_destroy_(kern_); kern_ = nullptr; }
    }
    ze_kernel_handle_t kern_;
    PFN_zeKernelDestroy fn_destroy_;
};

// ---------------------------------------------------------------------------
// SHA-256 helper — thin wrapper selecting the platform implementation.
// ---------------------------------------------------------------------------
namespace ze_sha256 {

/**
 * Compute SHA-256 over `count` (data, length) pairs chained together.
 * This allows hashing several blobs without concatenation.
 *
 * Implementation is selected at compile time via CMakeLists.txt defines.
 */
SHA256Digest hash_chain(
    const std::pair<const uint8_t *, size_t> *parts,
    size_t count) noexcept;

} // namespace ze_sha256

// ---------------------------------------------------------------------------
// LRU kernel cache.
// ---------------------------------------------------------------------------

/**
 * KernelEntry — a single cached module + kernel object.
 */
struct KernelEntry {
    SHA256Digest key;
    std::string  device_uuid;
    std::string  entry_name;
    ZeModule     module;
    ZeKernel     kernel;
};

/**
 * ZeKernelCache — 256-entry LRU cache for compiled ZeModule + ZeKernel pairs.
 *
 * Key = SHA-256(SPIR-V IL || device UUID || entry name || build options).
 * On a cache hit the entry is moved to the front (MRU position).
 * On a cache miss the caller must compile the module/kernel, then call
 * insert() to add the result. If the cache is at capacity, the LRU entry
 * (back of the list) is evicted — its RAII destructors run, calling
 * zeKernelDestroy and zeModuleDestroy.
 *
 * Thread-safety: a single std::mutex protects all operations because cache
 * accesses are infrequent (> 99% hit rate after warmup) and the lock is
 * held for O(1) time per operation.
 */
class ZeKernelCache {
public:
    /**
     * Look up a kernel by its SHA-256 key.
     * On a hit, moves the entry to MRU position and returns a pointer to it.
     * On a miss, returns nullptr.
     * The returned pointer is valid only while the mutex is held — callers
     * must use the kernel handle immediately and not cache the raw pointer.
     */
    const KernelEntry *get(const SHA256Digest &key,
                           const std::string  &device_uuid,
                           const std::string  &entry_name) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;

        // Secondary guard against hash collision.
        KernelEntry &e = *it->second;
        if (e.device_uuid != device_uuid || e.entry_name != entry_name) {
            return nullptr;
        }

        // Move to front (MRU).
        list_.splice(list_.begin(), list_, it->second);
        return &e;
    }

    /**
     * Insert a newly compiled kernel entry.
     * Takes ownership of module and kernel via move semantics.
     * If the cache is at capacity, evicts the LRU entry first.
     */
    void insert(SHA256Digest key,
                std::string  device_uuid,
                std::string  entry_name,
                ZeModule   &&mod,
                ZeKernel   &&kern) {
        std::lock_guard<std::mutex> lk(mu_);
        if (list_.size() >= ZE_KERNEL_CACHE_CAP) {
            // Evict LRU (tail).  RAII destructors call zeKernelDestroy + zeModuleDestroy.
            map_.erase(list_.back().key);
            list_.pop_back();
        }
        list_.push_front(KernelEntry{
            std::move(key),
            std::move(device_uuid),
            std::move(entry_name),
            std::move(mod),
            std::move(kern)
        });
        map_.emplace(list_.front().key, list_.begin());
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return list_.size();
    }

private:
    struct DigestHash {
        size_t operator()(const SHA256Digest &d) const noexcept {
            // Fold the 32-byte digest into a single size_t using FNV-1a.
            size_t h = 14695981039346656037ull;
            for (uint8_t b : d) {
                h ^= static_cast<size_t>(b);
                h *= 1099511628211ull;
            }
            return h;
        }
    };

    mutable std::mutex                                               mu_;
    std::list<KernelEntry>                                           list_;
    std::unordered_map<SHA256Digest,
                       std::list<KernelEntry>::iterator,
                       DigestHash>                                   map_;
};
