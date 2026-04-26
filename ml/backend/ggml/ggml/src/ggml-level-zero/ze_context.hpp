// SPDX-License-Identifier: MIT
// ze_context.hpp — RAII wrapper for ze_context_handle_t.
//
// One context is created per ze_driver_handle_t via zeContextCreate.
// Destructor calls zeContextDestroy, which flushes and frees all device-side
// resources allocated through this context.
//
// Design notes:
//   - Non-copyable (context handles must not be aliased).
//   - Move-constructible for optional<> and unique_ptr<> patterns.
//   - zeContextDestroy function pointer resolved at runtime via dlsym; the
//     pointer is set once by ZeLoader::init() before any ZeContext is created.
#pragma once

#include <cstdint>
#include <cassert>
#include <utility>

// Forward-declared L0 opaque types.
struct _ze_driver_handle_t;
struct _ze_context_handle_t;
typedef struct _ze_driver_handle_t  *ze_driver_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;

// Function-pointer typedefs for the L0 runtime calls used here.
// These are populated by ZeLoader::init() via dlsym at startup.
typedef int32_t (*PFN_zeContextDestroy)(ze_context_handle_t hContext);

/**
 * ZeContext — RAII owner of a ze_context_handle_t.
 *
 * Created by zeContextCreate in ze_ollama_device_open.
 * Destroyed (zeContextDestroy) in the destructor.
 *
 * The PFN_zeContextDestroy function pointer is injected at construction to
 * avoid a static-global dependency on a dynamically-resolved symbol.
 */
class ZeContext {
public:
    ZeContext() noexcept : ctx_(nullptr), fn_destroy_(nullptr) {}

    /**
     * Take ownership of hCtx.  fn_destroy is the zeContextDestroy function
     * pointer already resolved via dlsym by ZeLoader::init().
     */
    ZeContext(ze_context_handle_t hCtx, PFN_zeContextDestroy fn_destroy) noexcept
        : ctx_(hCtx), fn_destroy_(fn_destroy) {}

    // Non-copyable.
    ZeContext(const ZeContext &)            = delete;
    ZeContext &operator=(const ZeContext &) = delete;

    // Move semantics.
    ZeContext(ZeContext &&o) noexcept
        : ctx_(o.ctx_), fn_destroy_(o.fn_destroy_) {
        o.ctx_ = nullptr;
    }
    ZeContext &operator=(ZeContext &&o) noexcept {
        if (this != &o) {
            destroy_if_owned();
            ctx_        = o.ctx_;
            fn_destroy_ = o.fn_destroy_;
            o.ctx_      = nullptr;
        }
        return *this;
    }

    /**
     * Destructor calls zeContextDestroy if this instance owns the handle.
     * All device allocations made through this context must be freed before
     * the context is destroyed (enforcement is the responsibility of
     * ZeBuffer's destructor running before ZeContext's in RAII order).
     */
    ~ZeContext() noexcept { destroy_if_owned(); }

    ze_context_handle_t get()   const noexcept { return ctx_; }
    bool                valid() const noexcept { return ctx_ != nullptr; }

private:
    void destroy_if_owned() noexcept {
        if (ctx_ && fn_destroy_) {
            fn_destroy_(ctx_);
            ctx_ = nullptr;
        }
    }

    ze_context_handle_t ctx_;
    PFN_zeContextDestroy fn_destroy_;
};
