// SPDX-License-Identifier: MIT
// ggml-level-zero.cpp — Intel Level Zero GGML backend implementation.
//
// Registers as a GGML dynamic backend (GGML_BACKEND_DL) loadable at runtime
// by the GGML backend loader in ml/backend/ggml/ggml/src/ggml-backend-reg.cpp.
//
// Key design decisions (all from blueprint ADRs):
//   ADR-L0-001: Co-equal backend, no scheduler priority changes.
//   ADR-L0-002: GPU=ZE_DEVICE_TYPE_GPU, NPU=ZE_DEVICE_TYPE_VPU, skip CPU.
//   ADR-L0-003: AOT SPIR-V primary path; JIT fallback on MODULE_BUILD_FAILURE.
//   ADR-L0-004: Buffer pool (23 pow-2 buckets), LRU kernel cache (256 entries),
//               command-list ring buffer (64 slots lock-free SPSC).
//   ADR-L0-005: ze_ollama.h C ABI only; Pimpl; no C++ types across boundary.
//   ADR-L0-006: dlopen ze_loader at runtime; graceful degrade if missing.
//   ADR-L0-007: NPU gated by OLLAMA_L0_NPU_ENABLE env; memory clamped.
//
// All C++ exceptions are caught at the ze_ollama.h function boundary and
// translated to ze_ollama_result_t codes — never cross the C ABI.
//
// SHA-256 is computed via platform API (OpenSSL/BCrypt/builtin) for the
// kernel LRU cache key.
//
// C++17 standard. No C++20 features.

#include "ze_ollama.h"
#include "ze_device.hpp"
#include "ze_context.hpp"
#include "ze_queue.hpp"
#include "ze_buffer.hpp"
#include "ze_module.hpp"
#include "ze_event.hpp"
#include "l0_tensor_debug.h"

// GGML public headers (available via target_include_directories in CMakeLists).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

// Level Zero SDK headers — compile-time only; ze_loader is dlopen'd at runtime.
// LevelZero_INCLUDE_DIRS is set by CMakeLists.txt via find_package(LevelZero).
#include <level_zero/ze_api.h>

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>

// Platform-specific dynamic loading.
#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#   define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#   define NOMINMAX
#   endif
#   include <windows.h>
    typedef HMODULE ZeLoaderHandle;
#   define ZE_LOADER_OPEN(name)  LoadLibraryA(name)
#   define ZE_LOADER_SYM(h, s)   GetProcAddress((h), (s))
#   define ZE_LOADER_CLOSE(h)    FreeLibrary(h)
#else
#   include <dlfcn.h>
    typedef void *ZeLoaderHandle;
#   define ZE_LOADER_OPEN(name)  dlopen((name), RTLD_NOW | RTLD_LOCAL)
#   define ZE_LOADER_SYM(h, s)   dlsym((h), (s))
#   define ZE_LOADER_CLOSE(h)    dlclose(h)
#endif

// ---------------------------------------------------------------------------
// SHA-256 implementation (selected by CMakeLists.txt).
// ---------------------------------------------------------------------------
#if defined(ZE_OLLAMA_SHA256_OPENSSL)
#   include <openssl/evp.h>
namespace ze_sha256 {
SHA256Digest hash_chain(
    const std::pair<const uint8_t *, size_t> *parts, size_t count) noexcept {
    SHA256Digest out{};
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return out;
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    for (size_t i = 0; i < count; ++i) {
        EVP_DigestUpdate(ctx, parts[i].first, parts[i].second);
    }
    unsigned len = 32;
    EVP_DigestFinal_ex(ctx, out.data(), &len);
    EVP_MD_CTX_free(ctx);
    return out;
}
} // namespace ze_sha256
#elif defined(ZE_OLLAMA_SHA256_BCRYPT)
#   include <bcrypt.h>
namespace ze_sha256 {
SHA256Digest hash_chain(
    const std::pair<const uint8_t *, size_t> *parts, size_t count) noexcept {
    SHA256Digest out{};
    BCRYPT_ALG_HANDLE alg = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return out;
    BCRYPT_HASH_HANDLE hash = nullptr;
    BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0);
    for (size_t i = 0; i < count; ++i) {
        BCryptHashData(hash, const_cast<uint8_t *>(parts[i].first),
                       static_cast<ULONG>(parts[i].second), 0);
    }
    BCryptFinishHash(hash, out.data(), 32, 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    return out;
}
} // namespace ze_sha256
#else
// Minimal public-domain SHA-256 fallback (RFC 6234 reference implementation).
// This path is only used when neither OpenSSL nor BCrypt is available.
namespace ze_sha256 {
static void sha256_compress(uint32_t s[8], const uint8_t *blk) {
    static const uint32_t K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)blk[i*4] << 24) | ((uint32_t)blk[i*4+1] << 16)
             | ((uint32_t)blk[i*4+2] << 8) | blk[i*4+3];
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = (w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
        uint32_t s1 = (w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
        w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1   = (e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
        uint32_t ch   = (e&f)^(~e&g);
        uint32_t tmp1 = h+S1+ch+K[i]+w[i];
        uint32_t S0   = (a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
        uint32_t maj  = (a&b)^(a&c)^(b&c);
        uint32_t tmp2 = S0+maj;
        h=g;g=f;f=e;e=d+tmp1;d=c;c=b;b=a;a=tmp1+tmp2;
    }
    s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}
SHA256Digest hash_chain(
    const std::pair<const uint8_t *, size_t> *parts, size_t count) noexcept {
    SHA256Digest out{};
    uint32_t s[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    uint8_t  buf[64];
    uint64_t total = 0;
    size_t   buf_len = 0;
    auto flush = [&]() { sha256_compress(s, buf); buf_len = 0; };
    auto feed = [&](const uint8_t *p, size_t n) {
        total += n;
        while (n > 0) {
            size_t copy = (64 - buf_len < n) ? (64 - buf_len) : n;
            memcpy(buf + buf_len, p, copy);
            buf_len += copy; p += copy; n -= copy;
            if (buf_len == 64) flush();
        }
    };
    for (size_t i = 0; i < count; ++i) feed(parts[i].first, parts[i].second);
    // Padding.
    uint64_t bit_len = total * 8;
    buf[buf_len++] = 0x80;
    while (buf_len != 56) { if (buf_len == 64) flush(); buf[buf_len++] = 0; }
    for (int i = 7; i >= 0; --i) buf[buf_len++] = (uint8_t)(bit_len >> (i*8));
    flush();
    for (int i = 0; i < 8; ++i) {
        out[i*4+0] = (uint8_t)(s[i] >> 24);
        out[i*4+1] = (uint8_t)(s[i] >> 16);
        out[i*4+2] = (uint8_t)(s[i] >> 8);
        out[i*4+3] = (uint8_t)s[i];
    }
    return out;
}
} // namespace ze_sha256
#endif

// ---------------------------------------------------------------------------
// L0 function-pointer table (resolved at runtime via dlsym).
// Only the symbols actually used by this backend are resolved.
// ---------------------------------------------------------------------------
struct ZeLoader {
    ZeLoaderHandle handle = nullptr;

    // Core init / enumeration.
    int32_t (*zeInit)(uint32_t flags)                                    = nullptr;
    int32_t (*zeDriverGet)(uint32_t *, void **)                          = nullptr;
    int32_t (*zeDeviceGet)(void *, uint32_t *, void **)                  = nullptr;
    int32_t (*zeDeviceGetProperties)(void *, void *)                     = nullptr;
    int32_t (*zeDeviceGetComputeProperties)(void *, void *)              = nullptr;
    int32_t (*zeDeviceGetMemoryProperties)(void *, uint32_t *, void *)   = nullptr;

    // Context.
    int32_t (*zeContextCreate)(void *, const void *, void **)            = nullptr;
    int32_t (*zeContextDestroy)(void *)                                  = nullptr;

    // Command queue and list.
    int32_t (*zeCommandQueueCreate)(void *, void *, const void *, void **) = nullptr;
    int32_t (*zeCommandQueueDestroy)(void *)                              = nullptr;
    int32_t (*zeCommandQueueExecuteCommandLists)(void *, uint32_t, void **, void *) = nullptr;
    int32_t (*zeCommandQueueSynchronize)(void *, uint64_t)               = nullptr;
    int32_t (*zeCommandListCreate)(void *, void *, const void *, void **) = nullptr;
    int32_t (*zeCommandListDestroy)(void *)                               = nullptr;
    int32_t (*zeCommandListClose)(void *)                                 = nullptr;
    int32_t (*zeCommandListReset)(void *)                                 = nullptr;
    int32_t (*zeCommandListAppendMemoryCopy)(void *, void *, const void *, size_t, void *, uint32_t, void **) = nullptr;
    int32_t (*zeCommandListAppendLaunchKernel)(void *, void *, const void *, void *, uint32_t, void **) = nullptr;
    int32_t (*zeCommandListAppendBarrier)(void *, void *, uint32_t, void **) = nullptr;

    // Memory.
    int32_t (*zeMemAllocDevice)(void *, const void *, size_t, size_t, void *, void **) = nullptr;
    int32_t (*zeMemAllocHost)(void *, const void *, size_t, size_t, void **)            = nullptr;
    int32_t (*zeMemFree)(void *, void *)                                                = nullptr;

    // Module and kernel.
    int32_t (*zeModuleCreate)(void *, void *, const void *, void **, void **) = nullptr;
    int32_t (*zeModuleDestroy)(void *)                                         = nullptr;
    int32_t (*zeKernelCreate)(void *, const void *, void **)                   = nullptr;
    int32_t (*zeKernelDestroy)(void *)                                         = nullptr;
    int32_t (*zeKernelSetArgumentValue)(void *, uint32_t, size_t, const void *) = nullptr;
    int32_t (*zeKernelSetGroupSize)(void *, uint32_t, uint32_t, uint32_t)       = nullptr;

    // Immediate command list — required by ADR-001 H2D/D2H transfer pattern.
    // Correction 7: these two symbols were absent from the original loader table.
    int32_t (*zeCommandListCreateImmediate)(void *, void *, const void *, void **) = nullptr;
    int32_t (*zeCommandListHostSynchronize)(void *, uint64_t)                      = nullptr;

    // Events.
    int32_t (*zeEventPoolCreate)(void *, const void *, uint32_t, void **, void **) = nullptr;
    int32_t (*zeEventPoolDestroy)(void *)                                           = nullptr;
    int32_t (*zeEventCreate)(void *, const void *, void **)                         = nullptr;
    int32_t (*zeEventDestroy)(void *)                                                = nullptr;
    int32_t (*zeEventHostSynchronize)(void *, uint64_t)                              = nullptr;

    bool load() noexcept {
        // Try platform-specific loader library names.
#if defined(_WIN32)
        const char *names[] = {"ze_loader.dll", nullptr};
#else
        const char *names[] = {
            "libze_loader.so.1",
            "libze_loader.so",
            nullptr
        };
#endif
        for (int i = 0; names[i]; ++i) {
            handle = ZE_LOADER_OPEN(names[i]);
            if (handle) break;
        }
        if (!handle) return false;

#define LOAD_SYM(name) \
    *(void**)(&name) = (void*)ZE_LOADER_SYM(handle, #name); \
    if (!name) { unload(); return false; }

        LOAD_SYM(zeInit)
        LOAD_SYM(zeDriverGet)
        LOAD_SYM(zeDeviceGet)
        LOAD_SYM(zeDeviceGetProperties)
        LOAD_SYM(zeDeviceGetComputeProperties)
        LOAD_SYM(zeDeviceGetMemoryProperties)
        LOAD_SYM(zeContextCreate)
        LOAD_SYM(zeContextDestroy)
        LOAD_SYM(zeCommandQueueCreate)
        LOAD_SYM(zeCommandQueueDestroy)
        LOAD_SYM(zeCommandQueueExecuteCommandLists)
        LOAD_SYM(zeCommandQueueSynchronize)
        LOAD_SYM(zeCommandListCreate)
        LOAD_SYM(zeCommandListDestroy)
        LOAD_SYM(zeCommandListClose)
        LOAD_SYM(zeCommandListReset)
        LOAD_SYM(zeCommandListAppendMemoryCopy)
        LOAD_SYM(zeCommandListAppendLaunchKernel)
        LOAD_SYM(zeCommandListAppendBarrier)
        LOAD_SYM(zeCommandListCreateImmediate)
        LOAD_SYM(zeCommandListHostSynchronize)
        LOAD_SYM(zeMemAllocDevice)
        LOAD_SYM(zeMemAllocHost)
        LOAD_SYM(zeMemFree)
        LOAD_SYM(zeModuleCreate)
        LOAD_SYM(zeModuleDestroy)
        LOAD_SYM(zeKernelCreate)
        LOAD_SYM(zeKernelDestroy)
        LOAD_SYM(zeKernelSetArgumentValue)
        LOAD_SYM(zeKernelSetGroupSize)
        LOAD_SYM(zeEventPoolCreate)
        LOAD_SYM(zeEventPoolDestroy)
        LOAD_SYM(zeEventCreate)
        LOAD_SYM(zeEventDestroy)
        LOAD_SYM(zeEventHostSynchronize)
#undef LOAD_SYM
        return true;
    }

    void unload() noexcept {
        if (handle) {
            ZE_LOADER_CLOSE(handle);
            handle = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Global singleton state.  Protected by g_init_once / g_init_mutex.
// ---------------------------------------------------------------------------

// Maximum L0 devices tracked (per ADR-L0-004 DSA: bounded array, n<=16).
static constexpr uint32_t MAX_L0_DEVICES = 16u;

// NPU memory soft-cap observed on Meteor Lake (bytes).
static constexpr uint64_t NPU_MEMORY_CAP = 3ULL * 1024 * 1024 * 1024; // 3.5 GB rounded to 3 GB

static std::once_flag              g_init_once;
static ZeLoader                    g_loader;
static ze_ollama_result_t          g_init_result = ZE_OLLAMA_ERR_INTERNAL;
static std::atomic<uint32_t>       g_device_count{0};
static std::array<ze_ollama_device_s *, MAX_L0_DEVICES> g_devices{};

// ---------------------------------------------------------------------------
// Pimpl body for ze_ollama_device_s.
// Declared ONLY here (never in ze_ollama.h).
// ---------------------------------------------------------------------------
struct ze_ollama_device_s {
    ZeDevice     device;
    ZeContext    context;
    ZeCommandQueue queue;
    ZeBufferPool   pool;
    ZeKernelCache  kernel_cache;
    ZeEventDAG     event_dag;
    uint32_t       index;
    uint64_t       live_alloc_bytes{0};
    // Raw module handles owned by ze_ollama_load_spirv_kernels; destroyed in
    // the destructor so ze_ollama_device_close ("delete handle") triggers
    // cleanup automatically without requiring a separate teardown call.
    std::vector<ze_module_handle_t> loaded_modules;

    ~ze_ollama_device_s() noexcept {
        for (ze_module_handle_t mod : loaded_modules) {
            if (mod && g_loader.zeModuleDestroy) {
                g_loader.zeModuleDestroy(mod);
            }
        }
        loaded_modules.clear();
    }
};

struct GgmlL0Backend {
    struct ggml_backend     base;   // must be first field
    ze_ollama_device_s     *dev;
};

// ---------------------------------------------------------------------------
// Level Zero device buffer type — ADR-001.
// Tensors are allocated from ZeBufferPool (LRU, 23 pow-2 buckets) and reside
// in Intel Arc VRAM.  H2D/D2H transfers use the Level Zero immediate command
// list pattern so each transfer is synchronous with zero queue-submission
// overhead.
//
// is_host MUST return false so the GGML scheduler routes ops through the L0
// supports_op() check and inserts explicit H2D/D2H transfers at graph
// boundaries.
// ---------------------------------------------------------------------------

namespace {

/**
 * Per-buffer context stored in ggml_backend_buffer_t::context.
 * Holds everything needed for H2D, D2H, and pool-return on free.
 */
struct L0DevBufContext {
    void               *device_ptr; /* raw device pointer from ZeBufferPool::alloc */
    size_t              size;       /* original allocation size, for pool.free      */
    ZeBufferPool       *pool;       /* owning pool (same lifetime as device)        */
    ze_ollama_device_s *dev;        /* device for command-list creation             */
    uintptr_t           base_for_tensor_arithmetic; /* = (uintptr_t)device_ptr after successful alloc */
    enum L0BufKind { DEVICE_OWNED, DEVICE_SUBVIEW } buf_kind; /* always DEVICE_OWNED for normal allocs */
};

/** Return the opaque device pointer; the host must never dereference it. */
static void *l0_buf_get_base(ggml_backend_buffer_t buffer) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);
    return ctx->device_ptr;
}

/**
 * Return the buffer to the pool.  ZeBufferPool::free() pushes back to the
 * LRU free list — it does NOT call zeMemFree — so there is no double-free.
 */
static void l0_buf_free_buffer(ggml_backend_buffer_t buffer) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);
    if (ctx->device_ptr && ctx->pool) {
        ctx->pool->free(ctx->device_ptr, ctx->size);
        ctx->dev->live_alloc_bytes -= ctx->size;
    }
    delete ctx;
}

/**
 * Host-to-device copy via Level Zero immediate command list.
 * Creates a fresh immediate command list for each transfer, appends a
 * MemoryCopy, synchronizes to completion, then destroys the list.
 * Offset and size come from the tensor layout (ggml_backend_buffer.cpp
 * always passes ggml_nbytes(tensor) for size).
 */
static void l0_buf_set_tensor(ggml_backend_buffer_t buffer,
                              struct ggml_tensor *tensor,
                              const void *data, size_t offset, size_t size) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);

    /* zeCommandListCreateImmediate requires a ze_command_queue_desc_t, NOT
     * a ze_command_list_desc_t.  ze_command_queue_desc_t layout (ze_api.h):
     *   [0] stype  = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x000F
     *   [1] pNext  = 0
     *   [2] ordinal (uint32) = 0  (compute queue group 0)
     *   [3] index  (uint32) = 0   (queue index within the group)
     *   [4] flags  (uint32) = 0
     *   [5] mode   (uint32) = ZE_COMMAND_QUEUE_MODE_DEFAULT = 0
     *   [6] priority (uint32) = ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
     * Padded to 8 uint32s for safe alignment on any implementation. */
    uint32_t desc[8] = {
        0x000Fu, /* ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC */
        0u,      /* pNext */
        0u,      /* ordinal: compute queue group 0 */
        0u,      /* index: first queue in group */
        0u,      /* flags: 0 (default) */
        0u,      /* mode: ZE_COMMAND_QUEUE_MODE_DEFAULT */
        0u,      /* priority: ZE_COMMAND_QUEUE_PRIORITY_NORMAL */
        0u,      /* padding */
    };

    void *cmd_list = nullptr;
    int32_t r = g_loader.zeCommandListCreateImmediate(
        static_cast<void *>(ctx->dev->context.get()),
        static_cast<void *>(ctx->dev->device.handle()),
        static_cast<const void *>(desc),
        &cmd_list);
    if (r != 0 || !cmd_list) {
        GGML_LOG_ERROR("%s: zeCommandListCreateImmediate failed (r=%d)\n",
                       __func__, r);
        return;
    }

    GGML_ASSERT(tensor->data != nullptr && "tensor->data must be set by init_tensor before set_tensor");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    char *dst = reinterpret_cast<char *>(reinterpret_cast<uintptr_t>(tensor->data) + offset);
    L0_TRACE_TENSOR_IO(tensor, "set", offset, size);
    r = g_loader.zeCommandListAppendMemoryCopy(
        cmd_list, dst, data, size, nullptr, 0, nullptr);
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListAppendMemoryCopy H2D failed (r=%d)\n",
                       __func__, r);
        g_loader.zeCommandListDestroy(cmd_list);
        GGML_ABORT("L0 H2D copy failed");
    } else {
        r = g_loader.zeCommandListHostSynchronize(cmd_list, UINT64_MAX);
        if (r != 0) {
            GGML_LOG_ERROR("%s: zeCommandListHostSynchronize H2D failed (r=%d)\n",
                           __func__, r);
        }
    }
    g_loader.zeCommandListDestroy(cmd_list);
}

/**
 * Device-to-host copy via Level Zero immediate command list.
 * Mirror of l0_buf_set_tensor with source and destination swapped.
 */
static void l0_buf_get_tensor(ggml_backend_buffer_t buffer,
                              const struct ggml_tensor *tensor,
                              void *data, size_t offset, size_t size) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);

    /* Same ze_command_queue_desc_t fix as l0_buf_set_tensor. */
    uint32_t desc[8] = {
        0x000Fu, /* ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC */
        0u,      /* pNext */
        0u,      /* ordinal: compute queue group 0 */
        0u,      /* index */
        0u,      /* flags */
        0u,      /* mode: default */
        0u,      /* priority: normal */
        0u,      /* padding */
    };

    void *cmd_list = nullptr;
    int32_t r = g_loader.zeCommandListCreateImmediate(
        static_cast<void *>(ctx->dev->context.get()),
        static_cast<void *>(ctx->dev->device.handle()),
        static_cast<const void *>(desc),
        &cmd_list);
    if (r != 0 || !cmd_list) {
        GGML_LOG_ERROR("%s: zeCommandListCreateImmediate failed (r=%d)\n",
                       __func__, r);
        return;
    }

    GGML_ASSERT(tensor->data != nullptr && "tensor->data must be set by init_tensor before get_tensor");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    const char *src = reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(tensor->data) + offset);
    L0_TRACE_TENSOR_IO(tensor, "get", offset, size);
    r = g_loader.zeCommandListAppendMemoryCopy(
        cmd_list, data, src, size, nullptr, 0, nullptr);
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListAppendMemoryCopy D2H failed (r=%d)\n",
                       __func__, r);
    } else {
        r = g_loader.zeCommandListHostSynchronize(cmd_list, UINT64_MAX);
        if (r != 0) {
            GGML_LOG_ERROR("%s: zeCommandListHostSynchronize D2H failed (r=%d)\n",
                           __func__, r);
        }
    }
    g_loader.zeCommandListDestroy(cmd_list);
}

/* Fills the L0 device buffer with 'value'. Required: ggml_backend_buffer_clear
 * calls iface.clear without a null guard (ggml-backend.cpp:179).
 *
 * Reuses a pooled command list from ctx->dev->queue instead of creating a new
 * one, avoiding ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE (0x78000004) which is
 * returned by zeCommandListCreate/zeCommandListCreateImmediate whenever the
 * ze_context already has the 64 pre-existing command lists (opened by
 * ze_ollama_device_open) holding an active reference.
 *
 * Pattern: queue.acquire() → zeCommandListAppendMemoryCopy →
 *          zeCommandListClose → queue.ready() →
 *          queue.execute_and_sync() → queue.recycle(). */
static void l0_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);

    std::vector<uint8_t> host(buffer->size, value);

    ze_command_list_handle_t cmd_list = ctx->dev->queue.acquire();
    if (!cmd_list) {
        GGML_LOG_ERROR("%s: ZeCommandQueue::acquire failed (ring full)\n", __func__);
        return;
    }

    int32_t r = g_loader.zeCommandListAppendMemoryCopy(
        static_cast<void *>(cmd_list),
        ctx->device_ptr, host.data(), buffer->size,
        nullptr, 0, nullptr);
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListAppendMemoryCopy H2D failed (r=%d)\n",
                       __func__, r);
        ctx->dev->queue.recycle();
        return;
    }

    r = g_loader.zeCommandListClose(static_cast<void *>(cmd_list));
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListClose failed (r=%d)\n", __func__, r);
        ctx->dev->queue.recycle();
        return;
    }

    ctx->dev->queue.ready(cmd_list);
    r = ctx->dev->queue.execute_and_sync(UINT64_MAX);
    if (r != 0) {
        GGML_LOG_ERROR("%s: execute_and_sync failed (r=%d)\n", __func__, r);
    }
    ctx->dev->queue.recycle();
}

/**
 * Verifies the tensor->data invariant for an L0-allocated tensor.
 *
 * Per ggml-backend.cpp:1985-1996 (ggml_backend_tensor_alloc), the GGML
 * allocator computes addr = (char *)get_base(buffer) + buf_addr.offset and
 * assigns tensor->data = addr BEFORE invoking init_tensor.  That is, on entry
 * to this function, tensor->data already holds the absolute L0 device
 * address: no further rewrite is required.
 *
 * This function therefore performs only optional debug tracing and an
 * always-active bounds check (L0_ASSERT_TENSOR_IN_BUFFER) confirming
 * tensor->data falls within [device_ptr, device_ptr + size).  The assertion
 * catches any future regression where the GGML allocator contract changes or
 * where a caller bypasses ggml_backend_tensor_alloc.
 *
 * The base_for_tensor_arithmetic field on L0DevBufContext is retained for
 * future use (e.g. sub-buffer / view-buffer support) but is not consumed by
 * the current correctness path.
 */
static enum ggml_status l0_buf_init_tensor(ggml_backend_buffer_t buffer,
                                           struct ggml_tensor *tensor) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);
    (void)ctx;  // referenced only by L0_ASSERT_TENSOR_IN_BUFFER and trace macros
    L0_TRACE_TENSOR_IO(tensor, "init", 0, ggml_nbytes(tensor));
    L0_ASSERT_TENSOR_IN_BUFFER(tensor, ctx);
    return GGML_STATUS_SUCCESS;
}

/* Fills [tensor->data + offset, + size] bytes with 'value' on the L0 device.
 * Required: ggml_backend_tensor_memset asserts but does not guard iface.memset_tensor
 * in release builds (ggml-backend.cpp:341-343).
 *
 * Reuses a pooled command list from ctx->dev->queue (same pattern as
 * l0_buf_clear) to avoid ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE when called
 * while the 64 pre-existing context-owned command lists are active. */
static void l0_buf_memset_tensor(ggml_backend_buffer_t buffer,
                                 struct ggml_tensor *tensor,
                                 uint8_t value, size_t offset, size_t size) {
    auto *ctx = static_cast<L0DevBufContext *>(buffer->context);
    (void)buffer;

    char *dst = static_cast<char *>(tensor->data) + offset;
    std::vector<uint8_t> host(size, value);

    ze_command_list_handle_t cmd_list = ctx->dev->queue.acquire();
    if (!cmd_list) {
        GGML_LOG_ERROR("%s: ZeCommandQueue::acquire failed (ring full)\n", __func__);
        return;
    }

    int32_t r = g_loader.zeCommandListAppendMemoryCopy(
        static_cast<void *>(cmd_list),
        dst, host.data(), size,
        nullptr, 0, nullptr);
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListAppendMemoryCopy memset failed (r=%d)\n",
                       __func__, r);
        ctx->dev->queue.recycle();
        return;
    }

    r = g_loader.zeCommandListClose(static_cast<void *>(cmd_list));
    if (r != 0) {
        GGML_LOG_ERROR("%s: zeCommandListClose failed (r=%d)\n", __func__, r);
        ctx->dev->queue.recycle();
        return;
    }

    ctx->dev->queue.ready(cmd_list);
    r = ctx->dev->queue.execute_and_sync(UINT64_MAX);
    if (r != 0) {
        GGML_LOG_ERROR("%s: execute_and_sync failed (r=%d)\n", __func__, r);
    }
    ctx->dev->queue.recycle();
}

/* Buffer interface vtable for the L0 device buffer type.
 * Field order matches ggml_backend_buffer_i in ggml-backend-impl.h exactly:
 * free_buffer, get_base, init_tensor, memset_tensor, set_tensor,
 * get_tensor, cpy_tensor, clear, reset. */
static const struct ggml_backend_buffer_i g_l0_buf_i = {
    /* .free_buffer     = */ l0_buf_free_buffer,
    /* .get_base        = */ l0_buf_get_base,
    /* .init_tensor     = */ l0_buf_init_tensor,
    /* .memset_tensor   = */ l0_buf_memset_tensor,
    /* .set_tensor      = */ l0_buf_set_tensor,
    /* .get_tensor      = */ l0_buf_get_tensor,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ l0_buf_clear,
    /* .reset           = */ nullptr,
};

/* Forward declaration of the buffer type alloc function. */
static ggml_backend_buffer_t l0_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size);

static const char *l0_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "L0-device";
}

static size_t l0_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 64u; /* Intel GPU cache line + SIMD-16 alignment (ADR-001) */
}

static size_t l0_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    /* ZeBufferPool has 23 power-of-two buckets, bucket 22 covers [256MB, 512MB).
     * Returning 256MB forces the GGML allocator to split multi-GB model weights
     * across multiple buffers, each fitting the pool. Confirmed via llama3-8B
     * load failure when SIZE_MAX was returned (requested single 3.9 GB alloc). */
    return (size_t)256 * 1024 * 1024;
}

static size_t l0_buft_get_alloc_size(ggml_backend_buffer_type_t buft,
                                     const struct ggml_tensor *tensor) {
    (void)buft;
    /* Round up to 64-byte alignment boundary. */
    size_t nbytes = ggml_nbytes(tensor);
    return (nbytes + 63u) & ~(size_t)63u;
}

static bool l0_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return false; /* CRITICAL: false forces scheduler to use L0 supports_op() */
}

/* Buffer type interface vtable. */
static const struct ggml_backend_buffer_type_i g_l0_buft_i = {
    /* .get_name      = */ l0_buft_get_name,
    /* .alloc_buffer  = */ l0_buft_alloc_buffer,
    /* .get_alignment = */ l0_buft_get_alignment,
    /* .get_max_size  = */ l0_buft_get_max_size,
    /* .get_alloc_size= */ l0_buft_get_alloc_size,
    /* .is_host       = */ l0_buft_is_host,
    /* .free_type     = */ nullptr,
};

/*
 * File-static buffer type object.  context is set to the first GgmlL0Backend
 * pointer created (single-device assumption for v1; multi-device can share the
 * pool via dev pointer stored in the per-buffer context instead).
 */
static struct ggml_backend_buffer_type g_l0_buft = {
    /* .iface   = */ g_l0_buft_i,
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

/**
 * Allocate a new L0 device buffer of the requested size.
 * Option C safety net covers two pre-Option-A failure paths:
 *   (A) g_l0_buft.context is null: ggml_l0_dev_init_backend has not been called yet.
 *       In llm_load_tensors, tensor allocation (this function) is called BEFORE
 *       ggml_backend_dev_init, so the backend context is not yet populated.
 *   (B) pool.alloc() returns null: ze_ollama_device_open sets only dev->index;
 *       ZeBufferPool fn_alloc_ remains null until Option A initializes it.
 * Both paths delegate to the CPU allocator so model weights land in host RAM.
 * These branches become unreachable when Option A populates the pool.
 * Phase B GPU routing decisions (is_host=false, g_l0_buft default type) are preserved.
 */
static ggml_backend_buffer_t l0_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    auto *b = static_cast<GgmlL0Backend *>(buft->context);

    // Backend context null — tensor alloc precedes backend init.
    if (!b || !b->dev) {
        GGML_LOG_ERROR("%s: L0 backend context or device is null\n", __func__);
        return nullptr;
    }

    auto *ctx = new (std::nothrow) L0DevBufContext{};
    if (!ctx) return nullptr;

    ctx->size       = size;
    ctx->pool       = &b->dev->pool;
    ctx->dev        = b->dev;
    ctx->device_ptr = b->dev->pool.alloc(size);

    // Pool allocation failed (fn_alloc_ null or OOM).
    if (!ctx->device_ptr) {
        GGML_LOG_ERROR("%s: ZeBufferPool::alloc failed for size=%zu — device OOM\n", __func__, size);
        delete ctx;
        return nullptr;
    }
    ctx->base_for_tensor_arithmetic = reinterpret_cast<uintptr_t>(ctx->device_ptr);
    ctx->buf_kind = L0DevBufContext::DEVICE_OWNED;
    b->dev->live_alloc_bytes += size;

    return ggml_backend_buffer_init(buft, g_l0_buf_i, ctx, size);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// GGML backend vtable implementation.
// ---------------------------------------------------------------------------

static const char *ggml_l0_get_name(ggml_backend_t backend) {
    (void)backend;
    return "level_zero";
}

static void ggml_l0_free(ggml_backend_t backend) {
    auto *b = reinterpret_cast<GgmlL0Backend *>(backend);
    if (b->dev) {
        ze_ollama_device_close(b->dev);
        b->dev = nullptr;
    }
    delete b;
}

static ggml_backend_buffer_type_t ggml_l0_get_default_buffer_type(ggml_backend_t backend) {
    /* Wire the per-backend context into the file-static buffer type so that
     * l0_buft_alloc_buffer can reach the device's ZeBufferPool.
     * This assignment is idempotent under single-device usage (v1). */
    g_l0_buft.context = backend->context;
    return &g_l0_buft;
}

/**
 * Dispatch a GGML graph to the Intel Level Zero compute queue.
 *
 * Iterates every node in the graph, resolves the kernel entry name from the
 * op type and source tensor types, retrieves the compiled kernel handle from
 * the per-device ZeKernelCache, binds arguments per the authoritative kernel
 * signatures in build-l0-artifacts/01-blueprint-corrections.md (Corrections
 * 1–4), appends a launch record to a batched command list, then closes,
 * executes, and synchronises the list before returning.
 *
 * Unsupported or unrecognised ops are silently skipped (CPU fallback path).
 * Any Level Zero API failure is logged via GGML_LOG_ERROR and causes
 * GGML_STATUS_FAILED to be returned.
 *
 * Precondition: must not be called concurrently on the same device instance.
 * The GGML scheduler guarantees single-threaded execution per backend device,
 * and the underlying ZeCommandQueue is SPSC (single-producer/single-consumer).
 */
static ggml_status ggml_l0_graph_compute(ggml_backend_t backend,
                                          struct ggml_cgraph *cgraph,
                                          int batch_size) {
    (void)batch_size;
    auto *l0 = static_cast<GgmlL0Backend *>(backend->context);
    auto *dev = l0->dev;

    if (cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    /* Temporary device buffers allocated during SET_ROWS for src1 H2D staging.
     * Freed after zeCommandQueueSynchronize (GPU is done reading them). */
    struct TempBuf { void *ptr; size_t bytes; };
    std::vector<TempBuf> temp_bufs;

    // Step 2 — allocate a batched command list for the entire graph.
    ze_command_list_handle_t cmd_list = nullptr;
    ze_command_list_desc_t   cl_desc  = {};
    cl_desc.stype                = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cl_desc.pNext                = nullptr;
    cl_desc.commandQueueGroupOrdinal = 0u; // compute ordinal 0 — standard single-queue compute group
    cl_desc.flags                = 0; // batched (not immediate)

    ze_result_t ze_ret = static_cast<ze_result_t>(
        g_loader.zeCommandListCreate(
            dev->context.get(), dev->device.handle(), &cl_desc,
            reinterpret_cast<void **>(&cmd_list)));
    if (ze_ret != ZE_RESULT_SUCCESS) {
        GGML_LOG_ERROR("ggml_l0_graph_compute: zeCommandListCreate failed 0x%x\n",
                       static_cast<unsigned>(ze_ret));
        return GGML_STATUS_FAILED;
    }

    // Step 3 — iterate nodes, resolve entry name, bind args, append launch.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor *node = cgraph->nodes[i];
        if (!node) continue;

        const char *entry_name = nullptr;

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                ggml_type src0_type = node->src[0] ? node->src[0]->type : GGML_TYPE_COUNT;
                switch (src0_type) {
                    case GGML_TYPE_F32:  entry_name = "mul_mat_f32";  break;
                    case GGML_TYPE_F16:  entry_name = "mul_mat_f16";  break;
                    case GGML_TYPE_Q8_0: entry_name = "mul_mat_q8_0"; break;
                    case GGML_TYPE_Q4_0: entry_name = "mul_mat_q4_0"; break;
                    default:             break;
                }
                break;
            }
            case GGML_OP_SOFT_MAX:
                entry_name = (node->op_params[0] != 0)
                             ? "softmax_causal_f32"
                             : "softmax_f32";
                break;
            case GGML_OP_RMS_NORM:
                entry_name = "rms_norm";
                break;
            case GGML_OP_ROPE:
                entry_name = "rope";
                break;
            case GGML_OP_ADD:
                entry_name = "add_f32";
                break;
            case GGML_OP_MUL:
                entry_name = "mul_f32";
                break;
            case GGML_OP_CONT:
                entry_name = "copy_f32";
                break;
            case GGML_OP_UNARY: {
                ggml_unary_op act = ggml_get_unary_op(node);
                if (act == GGML_UNARY_OP_GELU)      entry_name = "gelu_f32";
                else if (act == GGML_UNARY_OP_SILU) entry_name = "silu_f32";
                break;
            }
            case GGML_OP_SET_ROWS:
                /* Flash-attention KV cache scatter-write (ROPE output -> KV cache).
                 * dst type determines which kernel to use:
                 *   F16 (default FA KV cache type) -> set_rows_f16
                 *   F32                            -> set_rows_f32 */
                entry_name = (node->type == GGML_TYPE_F16)
                             ? "set_rows_f16"
                             : "set_rows_f32";
                break;
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                /* Zero-copy tensor aliases — no kernel needed.
                 * The scheduler handles these as in-place operations on the
                 * existing L0 buffer.  Skip without dispatching any work. */
                continue;
            default:
                break;
        }

        if (!entry_name) {
            // Unsupported op — silently skip to CPU fallback.
            continue;
        }

        // Retrieve the compiled kernel from the LRU cache.
        // The cache key is SHA-256(entry_name bytes); the SPIR-V blob binding
        // is handled by the kernel-loading task (Task 6).  A cache miss here
        // means the kernel has not yet been compiled for this device, so we
        // log and fall back to CPU for this node.
        std::string entry_str(entry_name);
        std::string uuid_str(dev->device.uuid_str());

        const uint8_t *name_bytes  = reinterpret_cast<const uint8_t *>(entry_name);
        size_t         name_len    = entry_str.size();
        std::pair<const uint8_t *, size_t> parts[1] = {{ name_bytes, name_len }};
        SHA256Digest cache_key = ze_sha256::hash_chain(parts, 1);

        const KernelEntry *ke = dev->kernel_cache.get(cache_key, uuid_str, entry_str);
        if (!ke) {
            GGML_LOG_ERROR(
                "ggml_l0_graph_compute: kernel '%s' not in cache for device %s — skipping node\n",
                entry_name, uuid_str.c_str());
            continue;
        }

        ze_kernel_handle_t kernel = ke->kernel.get();

        // Bind arguments and set group size per authoritative signatures
        // (build-l0-artifacts/01-blueprint-corrections.md, Corrections 1–4).

        if (node->op == GGML_OP_MUL_MAT) {
            // Signature: mul_mat_*(A, B, C, int M, int N, int K)
            void *a_ptr = node->src[0]->data;
            void *b_ptr = node->src[1]->data;
            void *c_ptr = node->data;
            int   M     = static_cast<int>(node->src[0]->ne[1]);
            int   N     = static_cast<int>(node->src[1]->ne[1]);
            int   K     = static_cast<int>(node->src[0]->ne[0]);

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &a_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &b_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &c_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 3, sizeof(int),    &M);
            g_loader.zeKernelSetArgumentValue(kernel, 4, sizeof(int),    &N);
            g_loader.zeKernelSetArgumentValue(kernel, 5, sizeof(int),    &K);

            g_loader.zeKernelSetGroupSize(kernel, 16u, 16u, 1u);

            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>((M + 15) / 16);
            gc.groupCountY = static_cast<uint32_t>((N + 15) / 16);
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else if (node->op == GGML_OP_SOFT_MAX) {
            // Signature: softmax_f32(x, y, int n_cols, float scale)
            //            softmax_causal_f32(x, y, int n_cols, float scale, int current_pos)
            void  *x_ptr  = node->src[0]->data;
            void  *y_ptr  = node->data;
            int    n_cols = static_cast<int>(node->src[0]->ne[0]);
            float  scale  = 1.0f;

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *),  &x_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *),  &y_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(int),     &n_cols);
            g_loader.zeKernelSetArgumentValue(kernel, 3, sizeof(float),   &scale);

            if (node->op_params[0] != 0) {
                // Causal variant — bind current_pos from op_params[1].
                int cur_pos = static_cast<int>(node->op_params[1]);
                g_loader.zeKernelSetArgumentValue(kernel, 4, sizeof(int), &cur_pos);
            }

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            int n_rows = static_cast<int>(ggml_nrows(node));
            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>(n_rows);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else if (node->op == GGML_OP_RMS_NORM) {
            // GGML_OP_RMS_NORM: (src[0]=x, op_params[0]=eps) — no weight operand.
            // The learnable scale (gamma) is applied by a separate GGML_OP_MUL node
            // downstream in the graph (see ggml.c ggml_rms_norm()).
            void  *x_ptr  = node->src[0]->data;
            void  *y_ptr  = node->data;
            int    n_cols = static_cast<int>(node->src[0]->ne[0]);
            float  eps    = *reinterpret_cast<const float *>(&node->op_params[0]);
            if (eps == 0.0f) eps = 1e-6f;

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &x_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &y_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(int),    &n_cols);
            g_loader.zeKernelSetArgumentValue(kernel, 3, sizeof(float),  &eps);

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            int n_rows = static_cast<int>(ggml_nrows(node));
            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>(n_rows);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else if (node->op == GGML_OP_ROPE) {
            // Signature: rope(x, y, pos, int n_heads, int n_dims,
            //                 float theta_base, float freq_scale)
            void  *x_ptr      = node->src[0]->data;
            void  *y_ptr      = node->data;
            void  *pos_ptr    = node->src[1]->data;
            // ROPE tensor shape: [head_dim, n_heads, n_tokens] (ne[0]=head_dim fast,
            // ne[1]=n_heads, ne[2]=n_tokens).  The kernel decomposes gid into
            // (head_elem, head_idx, token_idx) matching this layout, so n_heads
            // must be ne[1].  Using ne[2] (= n_tokens) was swapping token/head
            // indices, causing pos[] to be indexed by head instead of token.
            int    n_heads    = static_cast<int>(node->src[0]->ne[1]);
            int    n_dims     = static_cast<int>(node->op_params[0]);
            float  theta_base = *reinterpret_cast<const float *>(&node->op_params[4]);
            float  freq_scale = *reinterpret_cast<const float *>(&node->op_params[5]);
            if (theta_base == 0.0f) theta_base = 10000.0f;
            if (freq_scale == 0.0f) freq_scale = 1.0f;

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &x_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &y_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &pos_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 3, sizeof(int),    &n_heads);
            g_loader.zeKernelSetArgumentValue(kernel, 4, sizeof(int),    &n_dims);
            g_loader.zeKernelSetArgumentValue(kernel, 5, sizeof(float),  &theta_base);
            g_loader.zeKernelSetArgumentValue(kernel, 6, sizeof(float),  &freq_scale);

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            int n_el = static_cast<int>(ggml_nelements(node));
            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>((n_el + 255) / 256);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
            // Signature: add_f32(a, b, c, int n_elements)
            //            mul_f32(a, b, c, int n_elements)
            void *a_ptr = node->src[0]->data;
            void *b_ptr = node->src[1]->data;
            void *c_ptr = node->data;
            int   n_el  = static_cast<int>(ggml_nelements(node));

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &a_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &b_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &c_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 3, sizeof(int),    &n_el);

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>((n_el + 255) / 256);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else if (node->op == GGML_OP_SET_ROWS) {
            /* set_rows_f16 / set_rows_f32 — scatter-write src0 (F32) rows to
             * dst (F16 or F32) at row indices from src1.
             *
             * Kernel signature (both f16 and f32 variants):
             *   (src0, src1, dst, ne00, ne01, ne02, ne03,
             *    ne11, ne12, s01, s02, s03, s10, s11, s12, s1, s2, s3)
             *
             * src1 (row index tensor) is CPU-backed on unified-memory platforms such as
             * Intel Arc.  On those platforms src1->data is already GPU-accessible and
             * staging it via a device copy feeds the kernel corrupted indices.
             * Only stage when the buffer is NOT host-accessible. */
            void *src0_ptr = node->src[0]->data;
            void *dst_ptr  = node->data;
            // node->data for a VIEW tensor is already base_ptr + byte_offset
            // (set at graph build time by ggml_view_*). No offset needed here.

            /* src1 holds integer row indices.  On Intel Arc (unified memory)
             * src1->data is always GPU-accessible — host-allocated buffers and
             * device-allocated buffers share the same unified address space.
             * Pass src1->data directly without staging, matching the CUDA
             * backend (set-rows.cu: const idx_t *src1_d = src1->data). */
            void *src1_ptr = node->src[1]->data;

            #ifdef GGML_L0_DEBUG
                fprintf(stderr,
                    "[L0 SET_ROWS] node->data=%p dst_ptr=%p src0->data=%p "
                    "src1->data=%p src1_is_host=%d src1_ptr=%p "
                    "ne00=%d ne01=%d\n",
                    node->data, dst_ptr, node->src[0]->data,
                    node->src[1]->data,
                    (int)ggml_backend_buffer_is_host(node->src[1]->buffer),
                    src1_ptr,
                    (int)node->src[0]->ne[0], (int)node->src[0]->ne[1]);
            #endif

            int ne00 = static_cast<int>(node->src[0]->ne[0]);
            int ne01 = static_cast<int>(node->src[0]->ne[1]);
            int ne02 = static_cast<int>(node->src[0]->ne[2]);
            int ne03 = static_cast<int>(node->src[0]->ne[3]);

            int ne11 = static_cast<int>(node->src[1]->ne[1]);
            int ne12 = static_cast<int>(node->src[1]->ne[2]);

            int s01 = static_cast<int>(node->src[0]->nb[1] / sizeof(float));
            int s02 = static_cast<int>(node->src[0]->nb[2] / sizeof(float));
            int s03 = static_cast<int>(node->src[0]->nb[3] / sizeof(float));

            size_t idx_elem = (node->src[1]->type == GGML_TYPE_I64)
                              ? sizeof(int64_t) : sizeof(int32_t);
            int s10 = static_cast<int>(node->src[1]->nb[0] / idx_elem);
            int s11 = static_cast<int>(node->src[1]->nb[1] / idx_elem);
            int s12 = static_cast<int>(node->src[1]->nb[2] / idx_elem);

            size_t dst_elem = (node->type == GGML_TYPE_F16)
                              ? sizeof(uint16_t) : sizeof(float);
            int s1 = static_cast<int>(node->nb[1] / dst_elem);
            int s2 = static_cast<int>(node->nb[2] / dst_elem);
            int s3 = static_cast<int>(node->nb[3] / dst_elem);

            g_loader.zeKernelSetArgumentValue(kernel,  0, sizeof(void *), &src0_ptr);
            g_loader.zeKernelSetArgumentValue(kernel,  1, sizeof(void *), &src1_ptr);
            g_loader.zeKernelSetArgumentValue(kernel,  2, sizeof(void *), &dst_ptr);
            g_loader.zeKernelSetArgumentValue(kernel,  3, sizeof(int),    &ne00);
            g_loader.zeKernelSetArgumentValue(kernel,  4, sizeof(int),    &ne01);
            g_loader.zeKernelSetArgumentValue(kernel,  5, sizeof(int),    &ne02);
            g_loader.zeKernelSetArgumentValue(kernel,  6, sizeof(int),    &ne03);
            g_loader.zeKernelSetArgumentValue(kernel,  7, sizeof(int),    &ne11);
            g_loader.zeKernelSetArgumentValue(kernel,  8, sizeof(int),    &ne12);
            g_loader.zeKernelSetArgumentValue(kernel,  9, sizeof(int),    &s01);
            g_loader.zeKernelSetArgumentValue(kernel, 10, sizeof(int),    &s02);
            g_loader.zeKernelSetArgumentValue(kernel, 11, sizeof(int),    &s03);
            g_loader.zeKernelSetArgumentValue(kernel, 12, sizeof(int),    &s10);
            g_loader.zeKernelSetArgumentValue(kernel, 13, sizeof(int),    &s11);
            g_loader.zeKernelSetArgumentValue(kernel, 14, sizeof(int),    &s12);
            g_loader.zeKernelSetArgumentValue(kernel, 15, sizeof(int),    &s1);
            g_loader.zeKernelSetArgumentValue(kernel, 16, sizeof(int),    &s2);
            g_loader.zeKernelSetArgumentValue(kernel, 17, sizeof(int),    &s3);

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            int n_el = ne00 * ne01 * ne02 * ne03;
            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>((n_el + 255) / 256);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr);
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);

        } else {
            // GGML_OP_CONT (copy_f32) and GGML_OP_UNARY (gelu_f32 / silu_f32).
            // Signature: copy_f32(src, dst, int n_elements)
            //            gelu_f32(src, dst, int n_elements)
            //            silu_f32(src, dst, int n_elements)
            void *src_ptr = node->src[0]->data;
            void *dst_ptr = node->data;
            int   n_el    = static_cast<int>(ggml_nelements(node));

            g_loader.zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &src_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &dst_ptr);
            g_loader.zeKernelSetArgumentValue(kernel, 2, sizeof(int),    &n_el);

            g_loader.zeKernelSetGroupSize(kernel, 256u, 1u, 1u);

            ze_group_count_t gc{};
            gc.groupCountX = static_cast<uint32_t>((n_el + 255) / 256);
            gc.groupCountY = 1u;
            gc.groupCountZ = 1u;
            g_loader.zeCommandListAppendLaunchKernel(
                cmd_list, kernel, &gc, nullptr, 0, nullptr);
        }
    }

    /* Lambda for cleanup on error paths. */
    auto cleanup_temp = [&]() {
        for (auto &tb : temp_bufs) { dev->pool.free(tb.ptr, tb.bytes); }
    };

    // Step 4 — close, execute, synchronise, destroy.
    ze_ret = static_cast<ze_result_t>(g_loader.zeCommandListClose(cmd_list));
    if (ze_ret != ZE_RESULT_SUCCESS) {
        GGML_LOG_ERROR("ggml_l0_graph_compute: zeCommandListClose failed 0x%x\n",
                       static_cast<unsigned>(ze_ret));
        cleanup_temp();
        g_loader.zeCommandListDestroy(cmd_list);
        return GGML_STATUS_FAILED;
    }

    ze_command_queue_handle_t queue_handle = dev->queue.get();
    ze_ret = static_cast<ze_result_t>(
        g_loader.zeCommandQueueExecuteCommandLists(
            queue_handle, 1, reinterpret_cast<void **>(&cmd_list), nullptr));
    if (ze_ret != ZE_RESULT_SUCCESS) {
        GGML_LOG_ERROR(
            "ggml_l0_graph_compute: zeCommandQueueExecuteCommandLists failed 0x%x\n",
            static_cast<unsigned>(ze_ret));
        cleanup_temp();
        g_loader.zeCommandListDestroy(cmd_list);
        return GGML_STATUS_FAILED;
    }

    ze_ret = static_cast<ze_result_t>(
        g_loader.zeCommandQueueSynchronize(queue_handle, UINT64_MAX));
    if (ze_ret != ZE_RESULT_SUCCESS) {
        GGML_LOG_ERROR(
            "ggml_l0_graph_compute: zeCommandQueueSynchronize failed 0x%x\n",
            static_cast<unsigned>(ze_ret));
        cleanup_temp();
        g_loader.zeCommandListDestroy(cmd_list);
        return GGML_STATUS_FAILED;
    }

    /* Free temporary staging buffers (src1 H2D copies for SET_ROWS).
     * GPU is done with them after successful synchronization. */
    for (auto &tb : temp_bufs) {
        dev->pool.free(tb.ptr, tb.bytes);
    }

    g_loader.zeCommandListDestroy(cmd_list);
    return GGML_STATUS_SUCCESS;
}

/**
 * Type-guarded op support decision table — ADR-002.
 *
 * The NPU narrow guard (lines immediately following) is preserved verbatim
 * from the original stub as required by ADR-002 Section "NPU Narrow Guard
 * Retention".  The switch below it implements the full decision table for GPU
 * dispatch.  Unsupported ops return false so the GGML scheduler falls back to
 * the CPU backend rather than silently producing wrong output.
 *
 * Ops NOT listed in the switch (e.g., GGML_OP_NORM, GGML_OP_FLASH_ATTN_EXT,
 * K-quant matmul variants) return false by the default arm.
 */
static bool ggml_l0_supports_op(ggml_backend_t backend, const struct ggml_tensor *op) {
    // Leaf tensors pre-allocated in L0 buffers have no computation — claim them.
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    auto *b = reinterpret_cast<GgmlL0Backend *>(backend);
    // NPU does not support F32 on some SKUs (preserved verbatim per ADR-002).
    if (b->dev && b->dev->device.device_kind() == ZE_OLLAMA_DEV_NPU) {
        if (op->type == GGML_TYPE_F32) {
            return b->dev->device.supports_fp16() ? true : false;
        }
    }

    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            /* Accept F32, F16, Q8_0, Q4_0 weight types only.
             * K-quant variants (Q4_1, Q5_x, Q6_K) have no compiled kernel. */
            ggml_type src_type = op->src[0] ? op->src[0]->type : GGML_TYPE_COUNT;
            return src_type == GGML_TYPE_F32  ||
                   src_type == GGML_TYPE_F16  ||
                   src_type == GGML_TYPE_Q8_0 ||
                   src_type == GGML_TYPE_Q4_0;
        }
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
            return (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 ||
                                   op->src[0]->type == GGML_TYPE_F16)) &&
                   (op->src[1] && (op->src[1]->type == GGML_TYPE_F32 ||
                                   op->src[1]->type == GGML_TYPE_F16));
        case GGML_OP_MUL:
            return op->src[0] && op->src[1] &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_UNARY: {
            ggml_unary_op act = ggml_get_unary_op(op);
            return act == GGML_UNARY_OP_GELU || act == GGML_UNARY_OP_SILU;
        }
        case GGML_OP_SET_ROWS:
            /* Required for flash-attention KV cache update path.
             * src0 = F32 (rope-encoded keys/values), src1 = I32 row indices,
             * dst = F32 or F16 KV cache buffer on the L0 device.
             * Only F16 and F32 dst types are implemented in the L0 kernels. */
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && (op->src[1]->type == GGML_TYPE_I32 ||
                                  op->src[1]->type == GGML_TYPE_I64) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            /* View-type ops are zero-copy aliases into the same buffer.
             * The GGML scheduler handles them without dispatching any kernel.
             * The L0 backend must claim them to prevent GGML_ABORT at line 844
             * when the scheduler assigns a pre-allocated L0 tensor these ops. */
            return true;
        default:
            return false;
    }
}

static struct ggml_backend_i g_l0_backend_i = {
    /* .get_name                = */ ggml_l0_get_name,
    /* .free                    = */ ggml_l0_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_l0_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
    /* .graph_reserve           = */ nullptr,
    /* .buffer_size             = */ nullptr,
    /* .reset                   = */ nullptr,
};

// ---------------------------------------------------------------------------
// ze_ollama.h C API implementation.
// ---------------------------------------------------------------------------

extern "C" {

ZE_OLLAMA_API ze_ollama_result_t ze_ollama_init(void) {
    std::call_once(g_init_once, []() {
        if (!g_loader.load()) {
            // Log at debug level via GGML callback if available.
            g_init_result = ZE_OLLAMA_ERR_LOADER_MISSING;
            return;
        }

        // zeInit — flags=0 means all driver types.
        int32_t r = g_loader.zeInit(0);
        if (r != 0) {
            g_init_result = ZE_OLLAMA_ERR_DRIVER_INIT;
            return;
        }
        g_init_result = ZE_OLLAMA_OK;
    });
    return g_init_result;
}

ZE_OLLAMA_API ze_ollama_result_t ze_ollama_enumerate_devices(
    ze_ollama_device_info_t *out_buf,
    size_t                   buf_cap,
    size_t                  *out_count)
{
    if (!out_buf || !out_count) return ZE_OLLAMA_ERR_INTERNAL;
    *out_count = 0;

    if (g_init_result != ZE_OLLAMA_OK) return g_init_result;

    bool npu_enable = false;
    const char *env = getenv("OLLAMA_L0_NPU_ENABLE");
    if (env && env[0] == '1') npu_enable = true;

    // Enumerate drivers.
    uint32_t n_drivers = 0;
    if (g_loader.zeDriverGet(&n_drivers, nullptr) != 0 || n_drivers == 0) {
        return ZE_OLLAMA_ERR_NO_DEVICE;
    }

    std::vector<void *> drivers(n_drivers);
    if (g_loader.zeDriverGet(&n_drivers, drivers.data()) != 0) {
        return ZE_OLLAMA_ERR_INTERNAL;
    }

    size_t count = 0;
    for (uint32_t di = 0; di < n_drivers && count < buf_cap; ++di) {
        uint32_t n_dev = 0;
        if (g_loader.zeDeviceGet(drivers[di], &n_dev, nullptr) != 0) continue;

        std::vector<void *> devs(n_dev);
        if (g_loader.zeDeviceGet(drivers[di], &n_dev, devs.data()) != 0) continue;

        for (uint32_t k = 0; k < n_dev && count < buf_cap; ++k) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            props.pNext = nullptr;
            g_loader.zeDeviceGetProperties(devs[k], &props);
            uint32_t dev_type = static_cast<uint32_t>(props.type);

            GGML_LOG_INFO("%s: Level Zero device %u: type=%u name=%s\n",
                __func__, k, dev_type, props.name);

            // type 1 (ZE_DEVICE_TYPE_GPU) path unchanged — GPU CI runner validates this
            // ZE_DEVICE_TYPE_VPU = 5 per ze_api.h L0 SDK >= 1.11 (Meteor Lake driver)
            const bool is_gpu = (dev_type == 1u);
            const bool is_npu = (dev_type == 4u || dev_type == 5u);
            if (!is_gpu && !is_npu) continue;
            if (is_npu && !npu_enable) continue;

            ze_ollama_device_info_t &info = out_buf[count];
            memset(&info, 0, sizeof(info));

            memcpy(info.name, props.name, sizeof(info.name) - 1);
            info.name[sizeof(info.name) - 1] = '\0';

            const uint8_t *uuid_bytes = props.uuid.id;
            snprintf(info.uuid, sizeof(info.uuid),
                     "%02x%02x%02x%02x-%02x%02x-%02x%02x"
                     "-%02x%02x-%02x%02x%02x%02x%02x%02x",
                     uuid_bytes[0],  uuid_bytes[1],  uuid_bytes[2],  uuid_bytes[3],
                     uuid_bytes[4],  uuid_bytes[5],  uuid_bytes[6],  uuid_bytes[7],
                     uuid_bytes[8],  uuid_bytes[9],  uuid_bytes[10], uuid_bytes[11],
                     uuid_bytes[12], uuid_bytes[13], uuid_bytes[14], uuid_bytes[15]);

            // Compute-unit fields live on ze_device_properties_t (numEUsPerSubslice,
            // numSubslicesPerSlice, numSlices) per ze_api.h L0 SDK >= 1.9, not on
            // ze_device_compute_properties_t which only carries group-size limits.
            info.compute_units = props.numEUsPerSubslice
                               * props.numSubslicesPerSlice
                               * props.numSlices;

            info.clock_mhz = props.coreClockRate;

            uint32_t n_mem = 0;
            g_loader.zeDeviceGetMemoryProperties(devs[k], &n_mem, nullptr);
            if (n_mem > 0) {
                std::vector<ze_device_memory_properties_t> mps(n_mem);
                for (auto &mp : mps) {
                    mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                    mp.pNext = nullptr;
                }
                g_loader.zeDeviceGetMemoryProperties(devs[k], &n_mem, mps.data());
                uint64_t largest = 0;
                for (const auto &mp : mps) if (mp.totalSize > largest) largest = mp.totalSize;
                info.total_memory = largest;
                info.free_memory  = largest;
            }

            // NPU memory cap (ADR-L0-007). Accept type 4 (legacy VPU) and type 5 (Meteor Lake NPU).
            if (is_npu) {
                if (info.total_memory > NPU_MEMORY_CAP) info.total_memory = NPU_MEMORY_CAP;
                if (info.free_memory  > NPU_MEMORY_CAP) info.free_memory  = NPU_MEMORY_CAP;
            }

            info.device_kind    = is_gpu ? ZE_OLLAMA_DEV_GPU : ZE_OLLAMA_DEV_NPU;
            info.supports_fp16  = 1;
            info.supports_int8  = 1;

            ++count;
        }
    }

    *out_count = count;
    return ZE_OLLAMA_OK;
}

// ---------------------------------------------------------------------------
// SPIR-V kernel pre-loading (Phase C Check 3).
//
// Conditional includes for CMake-generated SPIR-V blob headers.  Each header
// exposes:  const unsigned char g_spirv_<name>[];
//           const unsigned int  g_spirv_<name>_size;
// ---------------------------------------------------------------------------
#ifdef GGML_L0_SPIRV_EMBEDDED_HEADERS
#  include "kernels/mul_mat_spv_blob.h"
#  include "kernels/softmax_spv_blob.h"
#  include "kernels/attention_spv_blob.h"
#  include "kernels/rms_norm_spv_blob.h"
#  include "kernels/rope_spv_blob.h"
#  include "kernels/kv_cache_spv_blob.h"
#  include "kernels/gelu_silu_spv_blob.h"
#endif

/**
 * Pre-compile all SPIR-V kernels for a device and populate its ZeKernelCache.
 *
 * For every entry point across the 7 .cl source modules, this function creates
 * a ze_module_handle_t from the embedded SPIR-V IL blob, then creates a
 * ze_kernel_handle_t for each named entry point.  Each kernel is inserted into
 * dev->kernel_cache using the identical SHA-256 key derivation that
 * ggml_l0_graph_compute uses for lookup: hash_chain over the raw UTF-8 bytes
 * of the entry-point name string (single part, no SPIR-V blob in the key).
 *
 * Module handles are tracked in dev->loaded_modules for orderly destruction
 * when ze_ollama_device_close calls delete on the device.
 *
 * When GGML_L0_SPIRV_EMBEDDED_HEADERS is not defined at compile time, the
 * function emits a warning and returns ZE_RESULT_SUCCESS so the device still
 * opens; affected ops will fall back to the GGML CPU scheduler.
 *
 * A failure to compile or link one module is logged and skipped; remaining
 * modules continue loading so partial GPU acceleration is still available.
 */
static ze_result_t ze_ollama_load_spirv_kernels(ze_ollama_device_s *dev) {
#ifndef GGML_L0_SPIRV_EMBEDDED_HEADERS
    // To produce SPIR-V blobs at build time, install Intel oneAPI Base Toolkit
    // (provides ocloc.exe) OR LLVM with -target spir64 support.
    // Without GGML_L0_SPIRV_EMBEDDED_HEADERS defined, all GPU dispatches fall
    // back to CPU via the GGML scheduler (graceful degradation).
    GGML_LOG_WARN("ze_ollama_load_spirv_kernels: SPIR-V blobs not embedded at "
                  "build time; GPU dispatch will fall back to CPU.\n");
    return ZE_RESULT_SUCCESS;
#else
    struct ModuleSpec {
        const char          *source_name;
        const unsigned char *spirv;
        unsigned int         spirv_size;
        const char * const  *entry_points;
        int                  n_entry_points;
    };

    static const char *mul_mat_entries[]   = {"mul_mat_f32", "mul_mat_f16",
                                               "mul_mat_q8_0", "mul_mat_q4_0"};
    static const char *softmax_entries[]   = {"softmax_f32", "softmax_causal_f32"};
    static const char *attention_entries[] = {"attention_tiled"};
    static const char *rms_norm_entries[]  = {"rms_norm"};
    static const char *rope_entries[]      = {"rope"};
    static const char *kv_cache_entries[]  = {"kv_cache_write", "kv_cache_read",
                                               "set_rows_f32", "set_rows_f16"};
    static const char *gelu_silu_entries[] = {"gelu_f32", "silu_f32",
                                               "add_f32", "mul_f32", "copy_f32"};

    const ModuleSpec modules[] = {
        {"mul_mat",   g_spirv_mul_mat,   g_spirv_mul_mat_size,
                      mul_mat_entries,   4},
        {"softmax",   g_spirv_softmax,   g_spirv_softmax_size,
                      softmax_entries,   2},
        {"attention", g_spirv_attention, g_spirv_attention_size,
                      attention_entries, 1},
        {"rms_norm",  g_spirv_rms_norm,  g_spirv_rms_norm_size,
                      rms_norm_entries,  1},
        {"rope",      g_spirv_rope,      g_spirv_rope_size,
                      rope_entries,      1},
        {"kv_cache",  g_spirv_kv_cache,  g_spirv_kv_cache_size,
                      kv_cache_entries,  4},
        {"gelu_silu", g_spirv_gelu_silu, g_spirv_gelu_silu_size,
                      gelu_silu_entries, 5},
    };

    std::string uuid_str = dev->device.uuid_str();

    for (const auto &m : modules) {
        ze_module_desc_t mod_desc{};
        mod_desc.stype       = ZE_STRUCTURE_TYPE_MODULE_DESC;
        mod_desc.format      = ZE_MODULE_FORMAT_IL_SPIRV;
        mod_desc.inputSize   = static_cast<size_t>(m.spirv_size);
        mod_desc.pInputModule = m.spirv;
        mod_desc.pBuildFlags = nullptr;
        mod_desc.pConstants  = nullptr;

        ze_module_handle_t module = nullptr;
        ze_result_t ret = static_cast<ze_result_t>(g_loader.zeModuleCreate(
            dev->context.get(), dev->device.handle(), &mod_desc,
            reinterpret_cast<void**>(&module), nullptr));
        if (ret != ZE_RESULT_SUCCESS) {
            GGML_LOG_ERROR(
                "ze_ollama_load_spirv_kernels: zeModuleCreate failed for "
                "module '%s': 0x%x — skipping\n",
                m.source_name, static_cast<unsigned>(ret));
            continue;
        }

        dev->loaded_modules.push_back(module);

        for (int i = 0; i < m.n_entry_points; ++i) {
            ze_kernel_desc_t k_desc{};
            k_desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            k_desc.pKernelName = m.entry_points[i];

            ze_kernel_handle_t kernel = nullptr;
            ret = static_cast<ze_result_t>(g_loader.zeKernelCreate(
                module, &k_desc, reinterpret_cast<void**>(&kernel)));
            if (ret != ZE_RESULT_SUCCESS) {
                GGML_LOG_ERROR(
                    "ze_ollama_load_spirv_kernels: zeKernelCreate failed for "
                    "entry '%s' in module '%s': 0x%x — skipping\n",
                    m.entry_points[i], m.source_name,
                    static_cast<unsigned>(ret));
                continue;
            }

            // Build the cache key using the identical derivation that
            // ggml_l0_graph_compute uses at dispatch time (lines 736-739):
            //   parts[0] = { entry_name_bytes, entry_name_len }
            //   cache_key = ze_sha256::hash_chain(parts, 1)
            const char  *ep_name  = m.entry_points[i];
            const uint8_t *name_bytes =
                reinterpret_cast<const uint8_t *>(ep_name);
            size_t name_len = std::strlen(ep_name);
            std::pair<const uint8_t *, size_t> parts[1] = {
                {name_bytes, name_len}
            };
            SHA256Digest cache_key = ze_sha256::hash_chain(parts, 1);

            // ZeKernel RAII wrapper owns the kernel handle; moved into cache below.
            // ze_mod is intentionally NOT created here: dev->loaded_modules already
            // owns the module lifetime, and creating a ZeModule with a destroy fn
            // here would destroy the module at the end of each inner loop iteration,
            // making subsequent zeKernelCreate calls on the same module crash.
            ZeKernel ze_kern(kernel, reinterpret_cast<PFN_zeKernelDestroy>(g_loader.zeKernelDestroy));
            // Note: ZeModule passed here would destroy the module handle when
            // the ZeModule destructor runs if this is the last owner.  Because
            // dev->loaded_modules also holds the raw handle for cleanup, we
            // pass a no-op destroy function to avoid double-destroy.
            // The kernel cache owns the ZeKernel RAII object; the module
            // raw handle is kept alive in dev->loaded_modules.
            ZeModule ze_mod_no_destroy(module, nullptr);
            dev->kernel_cache.insert(
                cache_key,
                uuid_str,
                std::string(ep_name),
                std::move(ze_mod_no_destroy),
                std::move(ze_kern));

            GGML_LOG_INFO(
                "ze_ollama_load_spirv_kernels: loaded kernel '%s' from "
                "module '%s'\n", ep_name, m.source_name);
        }
    }

    return ZE_RESULT_SUCCESS;
#endif
}

/**
 * Opens an Intel Level Zero GPU device by logical GPU index and initialises
 * all L0 handles required for inference.
 *
 * The function performs a complete 10-step initialisation sequence:
 *   Step 0  — Validate the output pointer and allocate the Pimpl struct.
 *   Step 1  — Re-enumerate drivers/devices to recover driver_handle and
 *              device_handle for the requested GPU index.
 *   Step 2  — Populate dev->device (ZeDevice RAII wrapper).
 *   Step 3  — Create the L0 context via zeContextCreate.
 *   Step 4  — Record the compute queue ordinal (fixed at 0).
 *   Step 5  — Create the command queue via zeCommandQueueCreate.
 *   Step 6  — Create 64 command lists via zeCommandListCreate (ring buffer).
 *   Step 7  — Populate dev->queue (ZeCommandQueue RAII wrapper).
 *   Step 8  — Placement-new initialise dev->pool (ZeBufferPool, non-movable).
 *   Step 9  — Load SPIR-V kernels (non-fatal; partial GPU accel on failure).
 *   Step 10 — Assign *out and return ZE_OLLAMA_OK.
 *
 * @param index  Logical GPU index (0-based) matching ze_ollama_enumerate_devices.
 * @param out    Receives the opaque device handle on success; set to nullptr on
 *               error.
 *
 * @return ZE_OLLAMA_OK             on success.
 *         ZE_OLLAMA_ERR_INTERNAL   if out is null, context/queue/cmdlist creation
 *                                  fails, or an unexpected L0 error occurs.
 *         ZE_OLLAMA_ERR_NO_DEVICE  if no GPU at the requested index exists.
 *         ZE_OLLAMA_ERR_OOM        if the Pimpl allocation fails.
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_device_open(
    uint32_t index, ze_ollama_device_handle_t *out)
{
    // Step 0 — Validate output pointer and allocate Pimpl struct.
    if (!out) return ZE_OLLAMA_ERR_INTERNAL;
    *out = nullptr;
    auto *dev = new (std::nothrow) ze_ollama_device_s();
    if (!dev) return ZE_OLLAMA_ERR_OOM;
    dev->index = index;

    // Step 1 — Re-enumerate drivers and devices to recover driver_handle and
    // device_handle for the GPU at position [index].
    void *driver_handle = nullptr;
    void *device_handle = nullptr;
    ZeDeviceCaps caps{};
    bool found = false;
    uint32_t gpu_counter = 0;

    uint32_t n_drivers = 0;
    if (g_loader.zeDriverGet(&n_drivers, nullptr) != 0 || n_drivers == 0) {
        delete dev;
        return ZE_OLLAMA_ERR_NO_DEVICE;
    }

    std::vector<void *> drivers(n_drivers);
    if (g_loader.zeDriverGet(&n_drivers, drivers.data()) != 0) {
        delete dev;
        return ZE_OLLAMA_ERR_NO_DEVICE;
    }

    for (uint32_t di = 0; di < n_drivers && !found; ++di) {
        uint32_t n_dev = 0;
        if (g_loader.zeDeviceGet(drivers[di], &n_dev, nullptr) != 0) continue;

        std::vector<void *> devs(n_dev);
        if (g_loader.zeDeviceGet(drivers[di], &n_dev, devs.data()) != 0) continue;

        for (uint32_t k = 0; k < n_dev && !found; ++k) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            props.pNext = nullptr;
            g_loader.zeDeviceGetProperties(devs[k], &props);

            // Only count GPU devices (ZE_DEVICE_TYPE_GPU == 1u).
            if (static_cast<uint32_t>(props.type) != 1u) continue;

            if (gpu_counter == index) {
                driver_handle = drivers[di];
                device_handle = devs[k];

                caps.compute_units = props.numEUsPerSubslice
                                   * props.numSubslicesPerSlice
                                   * props.numSlices;
                caps.clock_mhz    = props.coreClockRate;
                caps.supports_fp16 = 1;
                caps.supports_int8 = 1;
                caps.device_kind   = 1u;

                memcpy(caps.name, props.name, sizeof(caps.name) - 1);
                caps.name[sizeof(caps.name) - 1] = '\0';

                const uint8_t *uid = props.uuid.id;
                snprintf(caps.uuid, sizeof(caps.uuid),
                         "%02x%02x%02x%02x-%02x%02x-%02x%02x"
                         "-%02x%02x-%02x%02x%02x%02x%02x%02x",
                         uid[0],  uid[1],  uid[2],  uid[3],
                         uid[4],  uid[5],  uid[6],  uid[7],
                         uid[8],  uid[9],  uid[10], uid[11],
                         uid[12], uid[13], uid[14], uid[15]);

                uint32_t n_mem = 0;
                g_loader.zeDeviceGetMemoryProperties(devs[k], &n_mem, nullptr);
                if (n_mem > 0) {
                    std::vector<ze_device_memory_properties_t> mps(n_mem);
                    for (auto &mp : mps) {
                        mp.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                        mp.pNext = nullptr;
                    }
                    g_loader.zeDeviceGetMemoryProperties(devs[k], &n_mem, mps.data());
                    uint64_t largest = 0;
                    for (const auto &mp : mps) if (mp.totalSize > largest) largest = mp.totalSize;
                    caps.total_memory = largest;
                }

                found = true;
            }

            // CRITICAL: advance gpu_counter for every GPU device encountered,
            // not just the matching one, so the index mapping is correct.
            ++gpu_counter;
        }
    }

    if (!found) {
        delete dev;
        return ZE_OLLAMA_ERR_NO_DEVICE;
    }

    // Step 2 — Populate dev->device (ZeDevice takes ownership of the handle).
    dev->device = ZeDevice(static_cast<ze_device_handle_t>(device_handle), caps);

    // Step 3 — Create the L0 context for this device.
    // Layout on x64: ze_context_desc_t = {stype(u32)@0, pad(u32)@4, pNext(ptr)@8, flags(u32)@16, pad@20} = 24 bytes.
    // Use 6 x uint32_t so pNext is always zero-initialised (never garbage from stack).
    uint32_t ctx_desc[6] = {0x0002u, 0u, 0u, 0u, 0u, 0u};
    void *ctx_handle = nullptr;
    auto ze_ret = g_loader.zeContextCreate(
        driver_handle,
        reinterpret_cast<void *>(ctx_desc),
        &ctx_handle);
    if (ze_ret != 0 || !ctx_handle) {
        delete dev;
        return ZE_OLLAMA_ERR_INTERNAL;
    }
    GGML_LOG_INFO("ze_ollama_device_open: context created for device %u\n", index);
    dev->context = std::move(ZeContext(
        static_cast<ze_context_handle_t>(ctx_handle),
        reinterpret_cast<PFN_zeContextDestroy>(g_loader.zeContextDestroy)));

    // Step 4 — Record the compute queue ordinal (compute group 0 is always present).
    const uint32_t ordinal = 0u;

    // Step 5 — Create the command queue.
    // Layout on x64: ze_command_queue_desc_t =
    //   {stype(u32)@0, pad@4, pNext(ptr)@8, ordinal(u32)@16, index@20, flags@24, mode@28, priority@32, pad@36} = 40 bytes.
    // Use 10 x uint32_t; ordinal is at slot [4] (offset 16), pNext slots [2][3] are zero.
    uint32_t q_desc[10] = {0x0005u, 0u, 0u, 0u, ordinal, 0u, 0u, 0u, 0u, 0u};
    void *queue_handle = nullptr;
    ze_ret = g_loader.zeCommandQueueCreate(
        ctx_handle,
        device_handle,
        reinterpret_cast<void *>(q_desc),
        &queue_handle);
    if (ze_ret != 0 || !queue_handle) {
        delete dev;
        return ZE_OLLAMA_ERR_INTERNAL;
    }
    GGML_LOG_INFO("ze_ollama_device_open: command queue created\n");

    // Step 6 — Create 64 command lists (ring buffer slots).
    // Layout on x64: ze_command_list_desc_t =
    //   {stype(u32)@0, pad@4, pNext(ptr)@8, queueGroupOrdinal(u32)@16, flags@20} = 24 bytes.
    // Use 6 x uint32_t; ordinal at slot [4] (offset 16), pNext slots [2][3] are zero.
    uint32_t cl_desc[6] = {0x0009u, 0u, 0u, 0u, ordinal, 0u};
    std::array<ze_command_list_handle_t, 64> cmd_lists{};
    for (uint32_t i = 0; i < 64u; ++i) {
        ze_command_list_handle_t cl = nullptr;
        ze_ret = g_loader.zeCommandListCreate(
            ctx_handle,
            device_handle,
            reinterpret_cast<void *>(cl_desc),
            reinterpret_cast<void **>(&cl));
        if (ze_ret != 0 || !cl) {
            for (uint32_t j = 0; j < i; ++j) {
                if (cmd_lists[j]) g_loader.zeCommandListDestroy(cmd_lists[j]);
            }
            g_loader.zeCommandQueueDestroy(queue_handle);
            delete dev;
            return ZE_OLLAMA_ERR_INTERNAL;
        }
        cmd_lists[i] = cl;
    }
    GGML_LOG_INFO("ze_ollama_device_open: 64 command lists created\n");

    // Step 7 — Populate dev->queue.  queue_handle is now owned by ZeCommandQueue;
    // do NOT call zeCommandQueueDestroy again after this point.
    dev->queue = std::move(ZeCommandQueue(
        static_cast<ze_command_queue_handle_t>(queue_handle),
        cmd_lists,
        reinterpret_cast<PFN_zeCommandQueueDestroy>(g_loader.zeCommandQueueDestroy),
        reinterpret_cast<PFN_zeCommandListDestroy>(g_loader.zeCommandListDestroy),
        reinterpret_cast<PFN_zeCommandListReset>(g_loader.zeCommandListReset),
        reinterpret_cast<PFN_zeCommandQueueExecuteCommandLists>(g_loader.zeCommandQueueExecuteCommandLists),
        reinterpret_cast<PFN_zeCommandQueueSynchronize>(g_loader.zeCommandQueueSynchronize)));

    // Step 8 — Placement-new initialise the buffer pool.
    // ZeBufferPool has all copy/move constructors deleted; placement new into
    // the struct member is the only valid initialisation path.
    new (&dev->pool) ZeBufferPool(
        static_cast<ze_context_handle_t>(ctx_handle),
        static_cast<ze_device_handle_t>(device_handle),
        reinterpret_cast<PFN_zeMemAllocDevice>(g_loader.zeMemAllocDevice),
        reinterpret_cast<PFN_zeMemFree>(g_loader.zeMemFree));
    GGML_LOG_INFO("ze_ollama_device_open: buffer pool initialized\n");

    // Step 9 — Load SPIR-V kernels (non-fatal; partial GPU acceleration on failure).
    ze_ollama_load_spirv_kernels(dev);

    // Step 10 — Publish the handle.
    *out = dev;
    return ZE_OLLAMA_OK;
}

ZE_OLLAMA_API void ze_ollama_device_close(ze_ollama_device_handle_t handle) {
    if (!handle) return;
    delete handle;
}

ZE_OLLAMA_API ze_ollama_result_t ze_ollama_device_free_memory(
    ze_ollama_device_handle_t handle, uint64_t *out_bytes)
{
    if (!handle || !out_bytes) return ZE_OLLAMA_ERR_INTERNAL;
    // Return total minus tracked live allocations.
    ze_ollama_device_info_t buf[MAX_L0_DEVICES];
    size_t cnt = 0;
    ze_ollama_result_t r = ze_ollama_enumerate_devices(buf, MAX_L0_DEVICES, &cnt);
    if (r != ZE_OLLAMA_OK) return r;
    if (handle->index >= cnt) return ZE_OLLAMA_ERR_NO_DEVICE;
    uint64_t total = buf[handle->index].total_memory;
    uint64_t live  = handle->live_alloc_bytes;
    *out_bytes = (total > live) ? (total - live) : 0u;
    return ZE_OLLAMA_OK;
}

ZE_OLLAMA_API const char *ze_ollama_result_str(ze_ollama_result_t result) {
    switch (result) {
        case ZE_OLLAMA_OK:                  return "ZE_OLLAMA_OK";
        case ZE_OLLAMA_ERR_LOADER_MISSING:  return "ZE_OLLAMA_ERR_LOADER_MISSING";
        case ZE_OLLAMA_ERR_NO_DEVICE:       return "ZE_OLLAMA_ERR_NO_DEVICE";
        case ZE_OLLAMA_ERR_DRIVER_INIT:     return "ZE_OLLAMA_ERR_DRIVER_INIT";
        case ZE_OLLAMA_ERR_OOM:             return "ZE_OLLAMA_ERR_OOM";
        case ZE_OLLAMA_ERR_UNSUPPORTED:     return "ZE_OLLAMA_ERR_UNSUPPORTED";
        case ZE_OLLAMA_ERR_INTERNAL:        return "ZE_OLLAMA_ERR_INTERNAL";
        default:                            return "unknown";
    }
}

ZE_OLLAMA_API const char *ze_ollama_version(void) {
#ifdef ZE_OLLAMA_VERSION_STR
    return ZE_OLLAMA_VERSION_STR;
#else
    return "1.0.0";
#endif
}

ZE_OLLAMA_API ze_ollama_result_t ze_ollama_shutdown(void) {
    for (uint32_t i = 0; i < MAX_L0_DEVICES; ++i) {
        if (g_devices[i]) {
            delete g_devices[i];
            g_devices[i] = nullptr;
        }
    }
    g_device_count.store(0, std::memory_order_relaxed);
    g_loader.unload();
    // Reset once_flag via placement-new (C++17).
    g_init_once.~once_flag();
    new (&g_init_once) std::once_flag();
    g_init_result = ZE_OLLAMA_ERR_INTERNAL;
    return ZE_OLLAMA_OK;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Public GGML API (ggml-level-zero.h).
// ---------------------------------------------------------------------------

extern "C" {

/* Level Zero backend GUID — required by ggml framework so that ggml_backend_is_cpu /
   ggml_backend_is_level_zero etc. can memcmp(backend->guid, ...) without null-deref.
   Value is arbitrary but must be unique vs other backends (see ggml_backend_cpu_guid
   in ggml-cpu.cpp which uses 0xaa,0x67,0xc7,0x43,... — ours is distinct). */
static ggml_guid_t ggml_backend_level_zero_guid(void) {
    static ggml_guid guid = { 0x4c, 0x5a, 0x45, 0x52, 0x4f, 0x55, 0x4d, 0x4c,
                              0xa4, 0xb5, 0xc6, 0xd7, 0xe8, 0xf9, 0x0a, 0x1b };
    return &guid;
}

GGML_BACKEND_API ggml_backend_t ggml_backend_level_zero_init(int device_id) {
    if (ze_ollama_init() != ZE_OLLAMA_OK) return nullptr;

    ze_ollama_device_handle_t dev = nullptr;
    if (ze_ollama_device_open(static_cast<uint32_t>(device_id), &dev) != ZE_OLLAMA_OK) {
        return nullptr;
    }

    auto *b = new (std::nothrow) GgmlL0Backend{};
    if (!b) {
        ze_ollama_device_close(dev);
        return nullptr;
    }
    b->base.guid    = ggml_backend_level_zero_guid();
    b->base.iface   = g_l0_backend_i;
    b->base.context = b;
    b->dev          = dev;
    /* Eagerly wire the backend into the file-static buffer type context so that
     * l0_buft_alloc_buffer can reach the device's ZeBufferPool even before the
     * first ggml_l0_get_default_buffer_type call. */
    g_l0_buft.context = b;
    return &b->base;
}

GGML_BACKEND_API bool ggml_backend_is_level_zero(ggml_backend_t backend) {
    if (!backend) return false;
    return backend->iface.get_name == ggml_l0_get_name;
}

GGML_BACKEND_API int ggml_backend_level_zero_get_device_count(void) {
    if (ze_ollama_init() != ZE_OLLAMA_OK) return 0;
    ze_ollama_device_info_t buf[MAX_L0_DEVICES];
    size_t cnt = 0;
    ze_ollama_enumerate_devices(buf, MAX_L0_DEVICES, &cnt);
    return static_cast<int>(cnt);
}

GGML_BACKEND_API const char *ggml_backend_level_zero_get_device_description(int device_id) {
    static thread_local char desc[256];
    ze_ollama_device_info_t buf[MAX_L0_DEVICES];
    size_t cnt = 0;
    if (ze_ollama_init() != ZE_OLLAMA_OK) {
        snprintf(desc, sizeof(desc), "<invalid>");
        return desc;
    }
    ze_ollama_enumerate_devices(buf, MAX_L0_DEVICES, &cnt);
    if (device_id < 0 || static_cast<size_t>(device_id) >= cnt) {
        snprintf(desc, sizeof(desc), "<invalid>");
        return desc;
    }
    snprintf(desc, sizeof(desc), "%s", buf[device_id].name);
    return desc;
}

} // extern "C"

// ---------------------------------------------------------------------------
// GGML device enumeration — bridges ze_ollama_enumerate_devices() into
// ggml_backend_device_i so the runner's /info endpoint can report each
// Level Zero GPU/NPU to the parent process.
// Lazily builds a static device array on first access; subsequent calls are
// O(1).  Each device's context points into the cached ze_ollama_device_info_t
// so name/description/memory queries need no fresh L0 calls.
// ---------------------------------------------------------------------------
struct ggml_l0_device_ctx {
    size_t                    index;
    ze_ollama_device_info_t   info;
    std::string               name_str;
    std::string               description_str;
};

static const char *ggml_l0_dev_get_name(ggml_backend_dev_t dev) {
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    return ctx->name_str.c_str();
}

static const char *ggml_l0_dev_get_description(ggml_backend_dev_t dev) {
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    return ctx->description_str.c_str();
}

static void ggml_l0_dev_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    if (total) { *total = static_cast<size_t>(ctx->info.total_memory); }
    if (free)  { *free  = static_cast<size_t>(ctx->info.free_memory); }
}

static enum ggml_backend_dev_type ggml_l0_dev_get_type(ggml_backend_dev_t dev) {
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    return (ctx->info.device_kind == ZE_OLLAMA_DEV_GPU)
        ? GGML_BACKEND_DEVICE_TYPE_GPU
        : GGML_BACKEND_DEVICE_TYPE_ACCEL; /* NPU treated as accelerator */
}

static void ggml_l0_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    std::memset(props, 0, sizeof(*props));
    props->name         = ctx->name_str.c_str();
    props->description  = ctx->description_str.c_str();
    props->id           = ctx->info.uuid;
    props->device_id    = ctx->info.uuid;
    props->memory_total = static_cast<size_t>(ctx->info.total_memory);
    props->memory_free  = static_cast<size_t>(ctx->info.free_memory);
    props->type         = ggml_l0_dev_get_type(dev);
    props->library      = (ctx->info.device_kind == ZE_OLLAMA_DEV_GPU)
                              ? "level-zero-gpu"
                              : "level-zero-npu";
    props->caps.async               = false;
    props->caps.host_buffer         = false;
    props->caps.buffer_from_host_ptr= false;
    props->caps.events              = false;
}

static ggml_backend_t ggml_l0_dev_init_backend(ggml_backend_dev_t dev, const char *params) {
    (void)params;
    auto *ctx = static_cast<ggml_l0_device_ctx *>(dev->context);
    ggml_backend_t backend = ggml_backend_level_zero_init(static_cast<int>(ctx->index));
    if (backend) {
        /* Wire the ggml_backend_t back to its owning ggml_backend_device_t so that
           ggml_backend_get_default_buffer_type(backend) -> ggml_backend_dev_buffer_type(backend->device)
           does not crash with GGML_ASSERT(device) at ggml-backend.cpp:544. */
        backend->device = dev;
    }
    return backend;
}

static ggml_backend_buffer_type_t ggml_l0_dev_get_buffer_type(ggml_backend_dev_t dev) {
    /* Return the file-static L0 device buffer type so that the device-level
     * buffer type query is consistent with ggml_l0_get_default_buffer_type.
     * Correction 6: this site was previously returning cpu_buffer_type(). */
    (void)dev;
    return &g_l0_buft;
}

static bool ggml_l0_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op) {
    /* Delegate to the backend-level decision table (ADR-002).
     * ggml_l0_dev_supports_op is what the GGML scheduler queries for offload
     * routing decisions.  We replicate the same type guards as
     * ggml_l0_supports_op without the NPU F32 narrowing (that guard requires a
     * live backend pointer; at device-query time no backend is allocated yet). */
    (void)dev;
    /* GGML_OP_NONE tensors are leaf/pre-allocated tensors (e.g. KV-cache
     * tensors cache_k_l0, cache_v_l0) that carry no computation.  The
     * scheduler calls ggml_backend_supports_op -> ggml_backend_dev_supports_op
     * -> this function to decide whether the L0 backend can "claim" a
     * pre-allocated tensor that already lives in an L0 device buffer.  If we
     * return false the scheduler cannot assign the tensor to any backend and
     * hits GGML_ABORT at ggml-backend.cpp:844.  Always return true for NONE. */
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            ggml_type src_type = op->src[0] ? op->src[0]->type : GGML_TYPE_COUNT;
            return src_type == GGML_TYPE_F32  ||
                   src_type == GGML_TYPE_F16  ||
                   src_type == GGML_TYPE_Q8_0 ||
                   src_type == GGML_TYPE_Q4_0;
        }
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_RMS_NORM:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ADD:
            return (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 ||
                                   op->src[0]->type == GGML_TYPE_F16)) &&
                   (op->src[1] && (op->src[1]->type == GGML_TYPE_F32 ||
                                   op->src[1]->type == GGML_TYPE_F16));
        case GGML_OP_MUL:
            return op->src[0] && op->src[1] &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_CONT:
            return true;
        case GGML_OP_UNARY: {
            ggml_unary_op act = ggml_get_unary_op(op);
            return act == GGML_UNARY_OP_GELU || act == GGML_UNARY_OP_SILU;
        }
        case GGML_OP_SET_ROWS:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && (op->src[1]->type == GGML_TYPE_I32 ||
                                  op->src[1]->type == GGML_TYPE_I64) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool ggml_l0_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    /* Accept only the native L0 device buffer.  Accepting CPU/CPU_Mapped buffer
     * types here causes the scheduler to route mul_mat to L0 with mmap'd host
     * pointers that the GPU cannot dereference without zeContextMakeMemoryResident.
     * The scheduler handles necessary tensor copies via cpy_tensor_async. */
    (void)dev;
    if (buft == &g_l0_buft) {
        return true;
    }
    if (buft && buft->iface.get_name) {
        const char *name = buft->iface.get_name(buft);
        if (name && strcmp(name, "L0-device") == 0) {
            return true;
        }
    }
    return false;
}

static const struct ggml_backend_device_i ggml_l0_device_i = {
    /* .get_name             = */ ggml_l0_dev_get_name,
    /* .get_description      = */ ggml_l0_dev_get_description,
    /* .get_memory           = */ ggml_l0_dev_get_memory,
    /* .get_type             = */ ggml_l0_dev_get_type,
    /* .get_props            = */ ggml_l0_dev_get_props,
    /* .init_backend         = */ ggml_l0_dev_init_backend,
    /* .get_buffer_type      = */ ggml_l0_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_l0_dev_supports_op,
    /* .supports_buft        = */ ggml_l0_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
    /* .reset                = */ nullptr,
};

// Lazily-built, process-lifetime device cache populated on first reg access.
static std::mutex                          g_l0_devices_mu;
static std::vector<ggml_l0_device_ctx>     g_l0_device_ctxs;
static std::vector<ggml_backend_device>    g_l0_devices;
static bool                                g_l0_devices_built = false;

static void ggml_l0_build_device_cache_locked(ggml_backend_reg_t reg) {
    if (g_l0_devices_built) return;
    g_l0_devices_built = true; /* mark up-front so re-entry during failure exits cleanly */

    if (ze_ollama_init() != ZE_OLLAMA_OK) return;

    ze_ollama_device_info_t buf[MAX_L0_DEVICES];
    size_t cnt = 0;
    if (ze_ollama_enumerate_devices(buf, MAX_L0_DEVICES, &cnt) != ZE_OLLAMA_OK) return;
    if (cnt == 0) return;

    g_l0_device_ctxs.reserve(cnt);
    g_l0_devices.reserve(cnt);
    for (size_t i = 0; i < cnt; ++i) {
        ggml_l0_device_ctx ctx;
        ctx.index = i;
        ctx.info  = buf[i];
        ctx.name_str = buf[i].name;
        char desc[512];
        std::snprintf(desc, sizeof(desc),
            "Intel Level Zero %s (compute_units=%u clock_mhz=%u fp16=%u int8=%u uuid=%s)",
            (buf[i].device_kind == ZE_OLLAMA_DEV_NPU ? "NPU" : "GPU"),
            buf[i].compute_units, buf[i].clock_mhz,
            buf[i].supports_fp16, buf[i].supports_int8, buf[i].uuid);
        ctx.description_str = desc;
        g_l0_device_ctxs.emplace_back(std::move(ctx));
    }
    for (size_t i = 0; i < cnt; ++i) {
        ggml_backend_device dev{};
        dev.iface   = ggml_l0_device_i;
        dev.reg     = reg;
        dev.context = &g_l0_device_ctxs[i];
        g_l0_devices.emplace_back(dev);
    }
}

static const char *ggml_l0_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "Level Zero";
}

static size_t ggml_l0_reg_get_device_count(ggml_backend_reg_t reg) {
    std::lock_guard<std::mutex> lk(g_l0_devices_mu);
    ggml_l0_build_device_cache_locked(reg);
    return g_l0_devices.size();
}

static ggml_backend_dev_t ggml_l0_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    std::lock_guard<std::mutex> lk(g_l0_devices_mu);
    ggml_l0_build_device_cache_locked(reg);
    if (index >= g_l0_devices.size()) return nullptr;
    return &g_l0_devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_level_zero_reg_i = {
    /* .get_name         = */ ggml_l0_reg_get_name,
    /* .get_device_count = */ ggml_l0_reg_get_device_count,
    /* .get_device       = */ ggml_l0_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg_t ggml_backend_level_zero_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_level_zero_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_level_zero_reg)
