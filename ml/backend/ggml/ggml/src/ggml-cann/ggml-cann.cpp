/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cann.h"

#include <acl/acl.h>
#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <chrono>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/common.h"
#include "ggml.h"

#define GGML_COMMON_DECL_C

#include "ggml-common.h"

#define GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

/**
 * @brief Sets the device to be used by CANN.
 *
 * @param device The device ID to set.
 */
void ggml_cann_set_device(const int32_t device) {
    // TODO: uncomment these lines after empty context has fixed.
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //   return;
    // }
    ACL_CHECK(aclrtSetDevice(device));
}

/**
 * @brief Retrieves the current device ID.
 *
 * @return The current device ID.
 */
int32_t ggml_cann_get_device() {
    int32_t id;
    ACL_CHECK(aclrtGetDevice(&id));
    return id;
}

/**
 * @brief Initialize the CANN device information.
 *
 * This function initializes the CANN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrtGetDeviceCount((uint32_t*)&info.device_count);

    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n",
                __func__, aclGetRecentErrMsg());
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = id;
        prop.reserve = 0;
        err = aclrtMemGetAllocationGranularity(
            &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            &info.devices[id].vmm_granularity);
        info.devices[id].vmm = err == ACL_SUCCESS;

        size_t free, total;
        ggml_backend_cann_get_device_memory(id, &free, &total);
        info.devices[id].total_vram = free;
    }

    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the CANN device information.
 *
 * This function returns a reference to a structure containing the CANN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_cann_device_info& ggml_cann_info() {
    static ggml_cann_device_info info = ggml_cann_init();
    return info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_buf_prio : public ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    static const size_t min_free_margin = 1ull << 20;   // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    static const size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer.
        size_t size = 0;      ///< Size of the buffer.
        std::chrono::steady_clock::time_point last_used;  ///< Last used time.

        bool operator>(const ggml_cann_buffer& other) const {
            return size > other.size;
        }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<void*, size_t> buffer_pool;
    std::priority_queue<ggml_cann_buffer,
                        std::vector<ggml_cann_buffer>,
                        std::greater<>> free_buffers ;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf_prio(int device) : device(device) {
        disable_clean = getenv("GGML_CANN_DISABLE_BUF_POOL_CLEAN") != nullptr;
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_set_device(device);
        for (auto& [b_ptr, b_size] : buffer_pool) {
            aclrtFree(b_ptr);
            pool_size -= b_size;
        }
        buffer_pool.clear();
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        void* ptr = nullptr;
        auto now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_buffer> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        while (!free_buffers.empty()) {
            auto b = free_buffers.top();
            free_buffers.pop();

            if (b.size >= size) {
                // reuse the buffer if the size is enough
                const size_t margin = b.size - size;
                if (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    ptr = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr,
                        (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(margin, 1048576) / 1048576));
#endif
                    break;
                }
            }

            bool should_clean = !disable_clean &&
                                b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            if (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
                buffer_pool.erase(b.ptr);
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr,
                    (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t)(GGML_PAD(b.size, 1048576) / 1048576));
#endif
                continue;
            }
            free_buffers_rest.push_back(b);
        }
        for (ggml_cann_buffer &b : free_buffers_rest) {
            free_buffers.push(std::move(b));
        }

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device, (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576));
#endif
        if (ptr != nullptr) {
            return ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_set_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = size;
        pool_size += size;
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
            (uint32_t)(GGML_PAD(size, 1048576) / 1048576));
#endif
        buffer_pool.emplace(ptr, size);
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        GGML_UNUSED(size);
        auto it = buffer_pool.find(ptr);
        if (it == buffer_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        auto now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_buffer{ptr, it->second, now});
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr,
            (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576));
#endif
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_buf : public ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    static const size_t min_free_margin = 1ull << 20;   // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    static const size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer memory.
        size_t size = 0;      ///< Size of the buffer.
        bool used = false;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point last_used;  ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf(int device) : device(device) {
        disable_clean = getenv("GGML_CANN_DISABLE_BUF_POOL_CLEAN") != nullptr;
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                aclrtFree(b.ptr);
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        void* ptr = nullptr;
        auto now = std::chrono::steady_clock::now();

        int i = 0;
        for (; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                break;
            }
            if (b.used) {
                continue;
            }
            if (b.size >= size) {
                // reuse the buffer if the size is enough
                const size_t margin = b.size - size;
                if (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    b.used = true;
                    ptr = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr,
                        (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(margin, 1048576) / 1048576));
#endif
                    break;
                }
            }

            bool should_clean = !disable_clean &&
                                b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            if (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr,
                    (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t)(GGML_PAD(b.size, 1048576) / 1048576));
#endif
                b.ptr = nullptr;
            }
        }
        if (ptr != nullptr) {
            return ptr;
        }

        if (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_buffer& b = buffer_pool[i];
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
            pool_size += size;
            *actual_size = size;
            b.size = size;
            b.used = true;
            if (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr,
                (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                (uint32_t)(GGML_PAD(b.size, 1048576) / 1048576));
#endif
            return b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        GGML_UNUSED(size);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer& b = buffer_pool[i];
            if (b.ptr != ptr) {
                continue;
            }
            b.used = false;
            b.last_used = std::chrono::steady_clock::now();
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr,
                (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576));
#endif
            return;
        }
        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }
};

/**
 * @brief A pool of CANN buffers with virtual memory.
 *
 * This class manages a pool of CANN buffers with virtual memory for a specific
 * device.
 */
struct ggml_cann_pool_vmm : public ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    void* pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    size_t granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<void*> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(int device)
    : device(device) {
        auto dev = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size = dev.total_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        if (pool_addr != 0) {
            for (auto& offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            for (auto& handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrtReleaseMemAddress(pool_addr));
        }
    }

    /**
     * @brief Allocate a buffer of the given size in the virtual memory pool.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        const size_t alignment = 128;
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = GGML_PAD(reserve_size, granularity);

            GGML_ASSERT(pool_size + reserve_size <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr = ACL_HBM_MEM_HUGE;
            prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            prop.reserve = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                ACL_CHECK(aclrtReserveMemAddress(
                    &pool_addr, max_size, 0, NULL, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((char*)pool_addr + pool_size, reserve_size, 0,
                                  handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((char*)pool_addr + pool_size);

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_CANN_MALLOC
             GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                   device, (unsigned long long) (pool_size/1024/1024),
                   (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void* ptr = (void*)((char*)pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device,
               (unsigned long long)size, (unsigned long long)ptr);
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device,
               (unsigned long long)size, (unsigned long long)ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void*)((char*)pool_addr + pool_used));
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(
    int device) {
    bool disable_vmm = (getenv("GGML_CANN_DISABLE_VMM_POOL") != nullptr);
    if (!disable_vmm && ggml_cann_info().devices[device].vmm) {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_vmm(device));
    }
    bool enable_buf_prio = (getenv("GGML_CANN_ENABLE_BUF_PRIO_POOL") != nullptr);
    if (enable_buf_prio) {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf_prio(device));
    }
    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf(device));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
static void * ggml_cann_host_malloc(size_t size) {
    if (getenv("GGML_CANN_NO_PINNED") != nullptr) {
        return nullptr;
    }

    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void * hostPtr = nullptr;
    aclError err = aclrtMallocHost((void **) &hostPtr, size);
    if (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, aclGetRecentErrMsg());
        return nullptr;
    }
    return hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_cann_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * hostPtr = ggml_cann_host_malloc(size);

    if (hostPtr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cann_host_buffer_free;

    return buffer;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cann_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cann_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cann_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cann_buffer_type_host;
}

/**
 * @brief Computes the forward operation for a given tensor using CANN
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The CANN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_cann_compute_forward(ggml_backend_cann_context& ctx,
                                      struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
            ggml_cann_binary_op<aclnn_add>(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cann_binary_op<aclnn_sub>(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cann_binary_op<aclnn_mul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_binary_op<aclnn_div>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_UNARY_OP(Abs);
                    break;
                case GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_UNARY_OP(Neg);
                    break;
                case GGML_UNARY_OP_GELU:
                    GGML_CANN_CALL_UNARY_OP(Gelu);
                    break;
                case GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_UNARY_OP(Silu);
                    break;
                case GGML_UNARY_OP_GELU_QUICK: {
                    auto lambda = [](ggml_backend_cann_context& ctx,
                        aclTensor* acl_src,
                        aclTensor* acl_dst) {
                        GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                    };
                    ggml_cann_unary_op(lambda, ctx, dst);
                } break;
                case GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_UNARY_OP(Tanh);
                    break;
                case GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_UNARY_OP(Relu);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_UNARY_OP(Sigmoid);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_UNARY_OP(Hardsigmoid);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_UNARY_OP(Hardswish);
                    break;
                case GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_UNARY_OP(Exp);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_UNARY_OP(Sign);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cann_step(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == nullptr);
            dst->src[1] = dst->src[0];
            ggml_cann_binary_op<aclnn_mul>(ctx, dst);
            break;
        case GGML_OP_SQRT:
            GGML_CANN_CALL_UNARY_OP(Sqrt);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cann_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
        case GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cann_unary_op<aclnn_cos>(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cann_unary_op<aclnn_sin>(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_LOG:
            GGML_CANN_CALL_UNARY_OP(Log);
            break;
        case GGML_OP_MEAN:
            ggml_cann_mean(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cann_count_equal(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
/**
 * @brief Retrieves the name associated with the CANN backend.
 *
 * This function returns the name assigned to the CANN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
static const char* ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    delete cann_ctx;
    delete backend;
}

/**
 * @brief Structure defining the interface for the CANN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the CANN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

// backend device
struct ggml_backend_cann_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char* ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cann_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *)dev->context;
    ggml_backend_cann_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CANN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *)dev->context;
    return ggml_backend_cann_init(ctx->device);
}

/**
 * @brief Checks if the CANN backend supports a specific backend buffer type.
 *
 * This function determines whether the CANN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the CANN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the CANN backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_cann_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;
        ggml_backend_cann_buffer_type_context * buft_ctx =
                        (ggml_backend_cann_buffer_type_context *)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *)dev->context;
    return ggml_backend_cann_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cann_host_buffer_type();
}

/**
 * @brief Creates a new event for the CANN backend device.
 *
 * This function initializes a new event for the CANN backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the CANN backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_cann_device_event_new(
    ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    ggml_cann_set_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    return new ggml_backend_event{
        /* .device = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), dev_ctx->device),
        /* .context = */ event,
    };
}

/**
 * @brief Frees a CANN backend event.
 *
 * This function destroys the ACL runtime event associated with the given CANN
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent)event->context));

    delete event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_memory,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,    // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL, // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};


// backend reg
struct ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CANN_NAME;
}

static size_t ggml_backend_cann_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_cann_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // reserved for future use
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            aclInit(nullptr);
            ggml_backend_cann_reg_context * ctx = new ggml_backend_cann_reg_context;

            for (int i = 0; i < ggml_cann_info().device_count; i++) {
                ggml_backend_cann_device_context* dev_ctx = new ggml_backend_cann_device_context();
                dev_ctx->description = aclrtGetSocName();
                dev_ctx->device = i;
                dev_ctx->name = GGML_CANN_NAME + std::to_string(i);
                ggml_cann_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cann_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cann_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(nullptr);
    if (device < 0 || device >= ggml_backend_cann_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context* ctx = new ggml_backend_cann_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_cann_set_device(ctx->device);
    ggml_backend_t cann_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_cann_guid(),
                         /* .interface = */ ggml_backend_cann_interface,
                         /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                         /* .context   = */ ctx};

    return cann_backend;
}

bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_info().device_count;
}

void ggml_backend_cann_get_device_description(
    int32_t device, char* description, size_t description_size) {
    ggml_cann_set_device(device);
    const char* soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

void ggml_backend_cann_get_device_memory(int32_t device, size_t* free,
                                         size_t* total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)
