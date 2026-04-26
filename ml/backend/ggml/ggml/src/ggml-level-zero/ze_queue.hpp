// SPDX-License-Identifier: MIT
// ze_queue.hpp — RAII wrapper for ze_command_queue_handle_t plus a lock-free
// SPSC command-list ring buffer (capacity 64 slots, per ADR-L0-004 §DSA).
//
// Command-list lifecycle state machine (enforced via per-slot atomic):
//   EMPTY (0) → BUILDING (1) → READY (2) → EXECUTING (3) → DONE (4) → EMPTY (0)
//
// The ring buffer is single-producer/single-consumer (SPSC) because GGML
// single-threads graph compute per backend.  No mutex is required for the
// push/pop paths; only the state transition on each slot uses atomic CAS.
//
// Thread-safety summary:
//   - acquire() / release() : SPSC — called only from the GGML compute thread
//   - destroy in destructor  : called only during shutdown (no concurrent use)
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cassert>
#include <utility>

// Forward-declared L0 opaque types.
struct _ze_command_queue_handle_t;
struct _ze_command_list_handle_t;
typedef struct _ze_command_queue_handle_t *ze_command_queue_handle_t;
typedef struct _ze_command_list_handle_t  *ze_command_list_handle_t;

// Function-pointer typedefs resolved at runtime by ZeLoader::init().
typedef int32_t (*PFN_zeCommandQueueDestroy)(ze_command_queue_handle_t);
typedef int32_t (*PFN_zeCommandListDestroy)(ze_command_list_handle_t);
typedef int32_t (*PFN_zeCommandListReset)(ze_command_list_handle_t);
typedef int32_t (*PFN_zeCommandQueueExecuteCommandLists)(
    ze_command_queue_handle_t,
    uint32_t,
    ze_command_list_handle_t *,
    void *);
typedef int32_t (*PFN_zeCommandQueueSynchronize)(
    ze_command_queue_handle_t,
    uint64_t);

// Per-slot state machine values.
enum class CmdListState : uint32_t {
    EMPTY     = 0,
    BUILDING  = 1,
    READY     = 2,
    EXECUTING = 3,
    DONE      = 4
};

// Ring-buffer capacity — must be a power of 2 (used as bitmask).
static constexpr uint32_t ZE_CMD_RING_CAP = 64u;

/**
 * ZeCommandQueue — RAII owner of a ze_command_queue_handle_t plus a
 * lock-free ring buffer of 64 command-list slots.
 *
 * Usage pattern (SPSC, from GGML compute thread):
 *
 *   auto *list = queue.acquire();          // borrow a BUILDING slot
 *   // append operations to list...
 *   queue.ready(list);                     // mark READY
 *   queue.execute_and_sync(timeout_ns);    // submit READY lists, wait
 *   queue.recycle();                       // reset and return DONE lists
 */
class ZeCommandQueue {
public:
    ZeCommandQueue() noexcept
        : queue_(nullptr)
        , fns_{}
        , head_(0)
        , tail_(0) {
        for (auto &s : slot_state_) {
            s.store(static_cast<uint32_t>(CmdListState::EMPTY),
                    std::memory_order_relaxed);
        }
        slots_.fill(nullptr);
    }

    ZeCommandQueue(ze_command_queue_handle_t q,
                   std::array<ze_command_list_handle_t, ZE_CMD_RING_CAP> lists,
                   PFN_zeCommandQueueDestroy       fn_q_destroy,
                   PFN_zeCommandListDestroy        fn_l_destroy,
                   PFN_zeCommandListReset          fn_l_reset,
                   PFN_zeCommandQueueExecuteCommandLists fn_execute,
                   PFN_zeCommandQueueSynchronize   fn_sync) noexcept
        : queue_(q)
        , fns_{fn_q_destroy, fn_l_destroy, fn_l_reset, fn_execute, fn_sync}
        , head_(0)
        , tail_(0)
        , slots_(lists) {
        for (auto &s : slot_state_) {
            s.store(static_cast<uint32_t>(CmdListState::EMPTY),
                    std::memory_order_relaxed);
        }
    }

    // Non-copyable.
    ZeCommandQueue(const ZeCommandQueue &)            = delete;
    ZeCommandQueue &operator=(const ZeCommandQueue &) = delete;

    ZeCommandQueue(ZeCommandQueue &&o) noexcept { *this = std::move(o); }
    ZeCommandQueue &operator=(ZeCommandQueue &&o) noexcept {
        if (this != &o) {
            destroy_all();
            queue_      = o.queue_;
            fns_        = o.fns_;
            head_.store(o.head_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
            tail_.store(o.tail_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
            slots_ = o.slots_;
            for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
                slot_state_[i].store(
                    o.slot_state_[i].load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }
            o.queue_ = nullptr;
            o.slots_.fill(nullptr);
        }
        return *this;
    }

    ~ZeCommandQueue() noexcept { destroy_all(); }

    ze_command_queue_handle_t get()   const noexcept { return queue_; }
    bool                      valid() const noexcept { return queue_ != nullptr; }

    /**
     * Acquire a command-list slot in the BUILDING state.
     * Advances the tail pointer (producer side).
     * Returns nullptr if the ring is full (caller must drain first).
     */
    ze_command_list_handle_t acquire() noexcept {
        uint32_t t   = tail_.load(std::memory_order_relaxed);
        uint32_t idx = t & (ZE_CMD_RING_CAP - 1u);
        uint32_t expected = static_cast<uint32_t>(CmdListState::EMPTY);
        if (!slot_state_[idx].compare_exchange_strong(
                expected,
                static_cast<uint32_t>(CmdListState::BUILDING),
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            return nullptr; // ring full or slot not yet EMPTY
        }
        tail_.store(t + 1u, std::memory_order_release);
        return slots_[idx];
    }

    /**
     * Mark the slot for the given list as READY (done building).
     */
    void ready(ze_command_list_handle_t list) noexcept {
        uint32_t idx = find_slot(list);
        if (idx < ZE_CMD_RING_CAP) {
            slot_state_[idx].store(
                static_cast<uint32_t>(CmdListState::READY),
                std::memory_order_release);
        }
    }

    /**
     * Submit all READY command lists then synchronise (blocking).
     * timeout_ns: nanoseconds to wait; UINT64_MAX for infinite.
     * Returns ZE_RESULT_SUCCESS (0) on success.
     */
    int32_t execute_and_sync(uint64_t timeout_ns) noexcept {
        // Collect READY lists.
        ze_command_list_handle_t ready_lists[ZE_CMD_RING_CAP];
        uint32_t                 ready_count = 0;
        for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
            if (slot_state_[i].load(std::memory_order_acquire) ==
                static_cast<uint32_t>(CmdListState::READY)) {
                slot_state_[i].store(
                    static_cast<uint32_t>(CmdListState::EXECUTING),
                    std::memory_order_release);
                ready_lists[ready_count++] = slots_[i];
            }
        }
        if (!ready_count || !fns_.execute || !queue_) {
            return 0;
        }
        int32_t r = fns_.execute(queue_, ready_count, ready_lists, nullptr);
        if (r != 0) return r;
        if (fns_.sync) {
            r = fns_.sync(queue_, timeout_ns);
        }
        // Mark EXECUTING → DONE.
        for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
            uint32_t st = slot_state_[i].load(std::memory_order_acquire);
            if (st == static_cast<uint32_t>(CmdListState::EXECUTING)) {
                slot_state_[i].store(
                    static_cast<uint32_t>(CmdListState::DONE),
                    std::memory_order_release);
            }
        }
        return r;
    }

    /**
     * Reset all DONE command lists and return them to EMPTY.
     * Must be called after execute_and_sync completes.
     */
    void recycle() noexcept {
        for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
            if (slot_state_[i].load(std::memory_order_acquire) ==
                static_cast<uint32_t>(CmdListState::DONE)) {
                if (fns_.reset && slots_[i]) {
                    fns_.reset(slots_[i]);
                }
                slot_state_[i].store(
                    static_cast<uint32_t>(CmdListState::EMPTY),
                    std::memory_order_release);
            }
        }
    }

private:
    uint32_t find_slot(ze_command_list_handle_t list) const noexcept {
        for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
            if (slots_[i] == list) return i;
        }
        return ZE_CMD_RING_CAP; // not found
    }

    void destroy_all() noexcept {
        if (!queue_) return;
        // Drain any in-flight lists first.
        if (fns_.sync) {
            fns_.sync(queue_, UINT64_MAX);
        }
        for (uint32_t i = 0; i < ZE_CMD_RING_CAP; ++i) {
            if (slots_[i] && fns_.list_destroy) {
                fns_.list_destroy(slots_[i]);
                slots_[i] = nullptr;
            }
        }
        if (fns_.queue_destroy) {
            fns_.queue_destroy(queue_);
            queue_ = nullptr;
        }
    }

    struct FnPtrs {
        PFN_zeCommandQueueDestroy             queue_destroy;
        PFN_zeCommandListDestroy              list_destroy;
        PFN_zeCommandListReset                reset;
        PFN_zeCommandQueueExecuteCommandLists execute;
        PFN_zeCommandQueueSynchronize         sync;
    };

    ze_command_queue_handle_t                        queue_;
    FnPtrs                                           fns_;
    std::atomic<uint32_t>                            head_;
    std::atomic<uint32_t>                            tail_;
    std::array<ze_command_list_handle_t, ZE_CMD_RING_CAP> slots_;
    std::array<std::atomic<uint32_t>, ZE_CMD_RING_CAP>    slot_state_;
};
