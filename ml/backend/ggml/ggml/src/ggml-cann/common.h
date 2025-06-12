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

#ifndef CANN_COMMON_H
#define CANN_COMMON_H

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>

#include "../include/ggml-cann.h"
#include "../include/ggml.h"
#include "../ggml-impl.h"

#define MATRIX_ROW_PADDING 512
#define GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, aclGetRecentErrMsg)

/**
 * @brief Contains information about CANN devices.
 */
struct ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_count;

    /**
     * @brief Information about a single CANN device.
     */
    struct cann_device_info {
        int cc;                 /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool vmm;               /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] =
        {}; /**< Array of CANN device information. */
};

const ggml_cann_device_info& ggml_cann_info();

void ggml_cann_set_device(int32_t device);
int32_t ggml_cann_get_device();

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
struct ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    virtual ~ggml_cann_pool() = default;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    virtual void* alloc(size_t size, size_t* actual_size) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    virtual void free(void* ptr, size_t size) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
struct ggml_cann_pool_alloc {
    ggml_cann_pool* pool = nullptr; /**< Pointer to the memory pool. */
    void* ptr = nullptr;    /**< Pointer to the allocated memory block. */
    size_t actual_size = 0; /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = default;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_cann_pool_alloc(ggml_cann_pool& pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_cann_pool_alloc(ggml_cann_pool& pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        return ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(ggml_cann_pool& pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    void* get() { return ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(const ggml_cann_pool_alloc&) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc&&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc& operator=(const ggml_cann_pool_alloc&) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc& operator=(ggml_cann_pool_alloc&&) = delete;
};

/**
 * @brief Function pointer type for ACLNN operator calls.
 */
using aclnn_func_t = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

/**
 * @brief Base class for all CANN tasks to be submitted to the task queue.
 *
 * Users should override the run_task() method with actual task logic.
 */
class cann_task {
public:
    virtual void run_task() {}
};

/**
 * @brief A lock-free ring-buffer based task queue for asynchronously executing cann_task instances.
 */
class cann_task_queue {
public:
    /**
     * @brief Constructs a task queue with a fixed power-of-two capacity for a specific device.
     *
     * @param capacity Queue capacity. Must be a power of 2.
     * @param device Target device ID (used for context setting).
     */
    explicit cann_task_queue(size_t capacity, int32_t device)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0),
          running_(false), device_(device) {
        GGML_ASSERT((capacity & (capacity - 1)) == 0 && "capacity must be power of 2");
        mask_ = capacity_ - 1;
    }

    /**
     * @brief Attempts to enqueue a task into the queue.
     *
     * @param item Unique pointer to the task.
     * @return true if the task was successfully enqueued, false if the queue was full.
     */
    bool enqueue(std::unique_ptr<cann_task>&& item) {
        size_t next_tail = (tail_ + 1) & mask_;

        if (next_tail == head_) {
            return false;
        }

        buffer_[tail_] = std::move(item);
        std::atomic_thread_fence(std::memory_order_release);
        tail_ = next_tail;

        return true;
    }

    /**
     * @brief Submits a task to the queue, and starts the worker thread if not already running.
     *
     * @param task Task to be submitted.
     */
    void submit_task(std::unique_ptr<cann_task>&& task) {
        while(!enqueue(std::move(task))) {
            std::this_thread::yield();
            continue;
        }

        if (!running_) {
            running_ = true;
            thread_ = std::thread(&cann_task_queue::execute, this);
        }

    }

    /**
     * @brief Waits until the queue is completely empty and no tasks are being processed.
     */
    void wait() {
        while (running_ && head_ != tail_) {
            std::this_thread::yield();
            continue;
        }
    }

    /**
     * @brief Stops the task queue and joins the worker thread.
     */
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    /**
     * @brief Worker thread function that continuously dequeues and executes tasks.
     */
    void execute() {
        ggml_cann_set_device(device_);

        while (running_) {
            if(head_ == tail_) {
                std::this_thread::yield();
                continue;
            }

            std::atomic_thread_fence(std::memory_order_acquire);
            buffer_[head_]->run_task();
            buffer_[head_].reset();
            head_ = (head_ + 1) & mask_;
        }
    }

    std::vector<std::unique_ptr<cann_task>> buffer_;
    const size_t capacity_;
    size_t mask_;
    size_t head_;
    size_t tail_;
    bool running_;
    std::thread thread_;
    int32_t device_;
};

/**
 * @brief Context for managing CANN backend operations.
 */
struct ggml_backend_cann_context {
    int32_t device;                  /**< Device ID. */
    std::string name;                /**< Name of the device. */
    std::string description;         /**< Description of the device. */
    aclrtEvent copy_event = nullptr; /**< Event for managing copy operations. */
    cann_task_queue task_queue;
    bool async_mode;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = {nullptr}; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(int device)
        : device(device), name("CANN" + std::to_string(device)), task_queue(1024, device) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();
        async_mode = (getenv("GGML_CANN_ASYNC_MODE") != nullptr);
        GGML_LOG_INFO("%s: device %d async operator submission is %s\n", __func__,
            device, async_mode ? "ON" : "OFF");
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        task_queue.stop();
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { return stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool>
        mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    static std::unique_ptr<ggml_cann_pool> new_pool_for_device(int device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool& pool() {
        if (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        return *mem_pool;
    }
};

#endif  // CANN_COMMON_H
