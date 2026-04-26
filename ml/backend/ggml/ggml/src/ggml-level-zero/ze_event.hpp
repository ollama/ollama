// SPDX-License-Identifier: MIT
// ze_event.hpp — RAII wrappers for ze_event_pool_handle_t / ze_event_handle_t
// plus an async dependency DAG with Kahn's BFS topological submit order
// (ADR-L0-004 §DSA: adjacency list, Kahn's BFS O(V+E)).
//
// DAG semantics:
//   - Nodes are ze_event_handle_t values.
//   - An edge from A → B means "A must be signalled before B can start".
//   - add_dep(producer, consumer) adds the directed edge A → B.
//   - topo_order() returns all events in a submission order that respects
//     all dependencies (Kahn's BFS, O(V+E)).
//
// Observer integration:
//   The Observer pattern (§5 patterns table) is realised as a callback
//   registered at event-pool creation via register_completion_callback().
//   The Go layer may supply a C function pointer that is invoked from the
//   completion-check thread when an event is signalled.  This bridges the
//   ze_event_handle_t world to Go channels for future async interop.
#pragma once

#include <unordered_map>
#include <vector>
#include <queue>
#include <mutex>
#include <cstdint>
#include <cassert>
#include <functional>
#include <utility>

// Forward-declared L0 types.
struct _ze_event_pool_handle_t;
struct _ze_event_handle_t;
typedef struct _ze_event_pool_handle_t *ze_event_pool_handle_t;
typedef struct _ze_event_handle_t      *ze_event_handle_t;

// Function-pointer typedefs.
typedef int32_t (*PFN_zeEventPoolDestroy)(ze_event_pool_handle_t);
typedef int32_t (*PFN_zeEventDestroy)(ze_event_handle_t);

// Observer callback: invoked when an event transitions to signalled.
// first argument = user-supplied context pointer.
// second argument = the signalled event handle.
typedef void (*ZeEventCompletionCB)(void *, ze_event_handle_t);

/**
 * ZeEventPool — RAII owner of a ze_event_pool_handle_t.
 * Destructor calls zeEventPoolDestroy after all child events are destroyed.
 */
class ZeEventPool {
public:
    ZeEventPool() noexcept : pool_(nullptr), fn_destroy_(nullptr) {}

    ZeEventPool(ze_event_pool_handle_t p, PFN_zeEventPoolDestroy fn) noexcept
        : pool_(p), fn_destroy_(fn) {}

    ZeEventPool(const ZeEventPool &)            = delete;
    ZeEventPool &operator=(const ZeEventPool &) = delete;

    ZeEventPool(ZeEventPool &&o) noexcept : pool_(o.pool_), fn_destroy_(o.fn_destroy_) {
        o.pool_ = nullptr;
    }
    ZeEventPool &operator=(ZeEventPool &&o) noexcept {
        if (this != &o) {
            destroy_if_owned();
            pool_       = o.pool_;
            fn_destroy_ = o.fn_destroy_;
            o.pool_     = nullptr;
        }
        return *this;
    }
    ~ZeEventPool() noexcept { destroy_if_owned(); }

    ze_event_pool_handle_t get()   const noexcept { return pool_; }
    bool                   valid() const noexcept { return pool_ != nullptr; }

private:
    void destroy_if_owned() noexcept {
        if (pool_ && fn_destroy_) { fn_destroy_(pool_); pool_ = nullptr; }
    }
    ze_event_pool_handle_t pool_;
    PFN_zeEventPoolDestroy fn_destroy_;
};

/**
 * ZeEvent — RAII owner of a single ze_event_handle_t.
 * Destructor calls zeEventDestroy.
 */
class ZeEvent {
public:
    ZeEvent() noexcept : ev_(nullptr), fn_destroy_(nullptr) {}

    ZeEvent(ze_event_handle_t e, PFN_zeEventDestroy fn) noexcept
        : ev_(e), fn_destroy_(fn) {}

    ZeEvent(const ZeEvent &)            = delete;
    ZeEvent &operator=(const ZeEvent &) = delete;

    ZeEvent(ZeEvent &&o) noexcept : ev_(o.ev_), fn_destroy_(o.fn_destroy_) {
        o.ev_ = nullptr;
    }
    ZeEvent &operator=(ZeEvent &&o) noexcept {
        if (this != &o) {
            destroy_if_owned();
            ev_         = o.ev_;
            fn_destroy_ = o.fn_destroy_;
            o.ev_       = nullptr;
        }
        return *this;
    }
    ~ZeEvent() noexcept { destroy_if_owned(); }

    ze_event_handle_t get()   const noexcept { return ev_; }
    bool              valid() const noexcept { return ev_ != nullptr; }

private:
    void destroy_if_owned() noexcept {
        if (ev_ && fn_destroy_) { fn_destroy_(ev_); ev_ = nullptr; }
    }
    ze_event_handle_t ev_;
    PFN_zeEventDestroy fn_destroy_;
};

// ---------------------------------------------------------------------------
// Async DAG — adjacency list + Kahn's BFS topological sort.
// ---------------------------------------------------------------------------

/**
 * ZeEventDAG — directed acyclic graph of ze_event_handle_t dependencies.
 *
 * Edges represent "must-complete-before" ordering.
 * add_dep(producer, consumer): producer's completion is prerequisite for consumer.
 * topo_order(): returns all events in a valid submission order (Kahn's BFS).
 *
 * After topo_order() is called the caller submits the events in the returned
 * order to the command list using zeCommandListAppendWaitOnEvents.
 *
 * Thread-safety: protected by a single mutex.  The DAG is rebuilt per
 * graph-compute call and cleared after submission.
 */
class ZeEventDAG {
public:
    /**
     * Register a dependency: consumer must wait for producer.
     */
    void add_dep(ze_event_handle_t producer, ze_event_handle_t consumer) {
        std::lock_guard<std::mutex> lk(mu_);
        adj_[producer].push_back(consumer);
        in_degree_[consumer]++;
        // Ensure producer has an entry (even with zero in-degree).
        in_degree_.emplace(producer, in_degree_[producer]);
    }

    /**
     * Kahn's BFS topological sort.
     * Returns all events in submission order (sources first).
     * Returns an empty vector if a cycle is detected (should never happen
     * in a well-formed GGML graph; logged as an internal error in the caller).
     */
    std::vector<ze_event_handle_t> topo_order() const {
        std::lock_guard<std::mutex> lk(mu_);

        // Copy in-degrees (we mutate them during BFS).
        std::unordered_map<ze_event_handle_t, int32_t> deg = in_degree_;

        std::queue<ze_event_handle_t> q;
        for (auto &[ev, d] : deg) {
            if (d == 0) q.push(ev);
        }

        std::vector<ze_event_handle_t> order;
        order.reserve(deg.size());

        while (!q.empty()) {
            ze_event_handle_t ev = q.front();
            q.pop();
            order.push_back(ev);
            auto it = adj_.find(ev);
            if (it != adj_.end()) {
                for (ze_event_handle_t succ : it->second) {
                    if (--deg[succ] == 0) {
                        q.push(succ);
                    }
                }
            }
        }

        // Cycle detection: if order.size() < number of unique events, cycle exists.
        if (order.size() != deg.size()) {
            return {}; // cycle detected
        }
        return order;
    }

    /**
     * Clear all state. Called after each graph-compute submission.
     */
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        adj_.clear();
        in_degree_.clear();
    }

    /**
     * Register an Observer callback invoked when a given event is signalled.
     * The callback is stored per-event. Called from the compute thread after
     * zeEventHostSynchronize returns for that event.
     * ctx is passed as the first argument to cb.
     */
    void register_completion_callback(ze_event_handle_t ev,
                                      ZeEventCompletionCB cb,
                                      void *ctx) {
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_[ev] = {cb, ctx};
    }

    /**
     * Fire registered callbacks for signalled events.
     * Called by the backend after synchronisation, before clearing.
     */
    void notify_completed(const std::vector<ze_event_handle_t> &signalled) {
        std::lock_guard<std::mutex> lk(mu_);
        for (ze_event_handle_t ev : signalled) {
            auto it = callbacks_.find(ev);
            if (it != callbacks_.end()) {
                auto &[cb, ctx] = it->second;
                if (cb) cb(ctx, ev);
            }
        }
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<ze_event_handle_t, std::vector<ze_event_handle_t>> adj_;
    std::unordered_map<ze_event_handle_t, int32_t>                        in_degree_;
    std::unordered_map<ze_event_handle_t,
                       std::pair<ZeEventCompletionCB, void *>>             callbacks_;
};
