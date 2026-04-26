// SPDX-License-Identifier: MIT
// l0_tensor_debug.h — Debug instrumentation macros for L0 tensor I/O tracing.
//
// All macros compile to ((void)0) when OLLAMA_L0_DEBUG_TENSOR_IO is not defined,
// making this header production-safe with zero overhead.
//
// Usage:
//   To enable tracing, define OLLAMA_L0_DEBUG_TENSOR_IO before including this
//   header, or pass -DOLLAMA_L0_DEBUG_TENSOR_IO to the compiler.  This is
//   handled by the build system (cloud-engineer, T07) and must NOT be enabled
//   in production builds.
//
//   Macros:
//     L0_TRACE_TENSOR_IO(tensor, op, offset, size)
//       Prints one trace line per tensor I/O operation (init/set/get).
//     L0_HEXDUMP_FIRST_BYTES(ptr, n)
//       Prints the first n bytes of a host-readable buffer as hex.
//       WARNING: do NOT call on device-only (non-host-visible) memory —
//       the pointer is not dereferenceable from the host and will crash.
//     L0_ASSERT_TENSOR_IN_BUFFER(tensor, ctx)
//       Asserts that tensor->data is within [ctx->device_ptr, ctx->device_ptr +
//       ctx->size).  Calls GGML_ABORT on violation.  Always active (both debug
//       and release) — this is a safety invariant, not a debug-only check.

#ifndef OLLAMA_L0_TENSOR_DEBUG_H
#define OLLAMA_L0_TENSOR_DEBUG_H

#include <cstdint>
#include <cstdio>
#include "ggml.h"
#include "ggml-impl.h"

// ---------------------------------------------------------------------------
// L0_ASSERT_TENSOR_IN_BUFFER — always active (invariant assertion)
// ---------------------------------------------------------------------------

/**
 * Verifies that [tensor->data, tensor->data + ggml_nbytes(tensor)) lies
 * entirely within the buffer's device allocation [ctx->device_ptr,
 * ctx->device_ptr + ctx->size).  Calls GGML_ABORT with a descriptive message
 * if the invariant is violated.
 *
 * This macro is active in both debug and release builds because an
 * out-of-bounds device pointer causes silent GPU memory corruption that is
 * extremely difficult to diagnose post-hoc.
 *
 * MATH_DELEGATE: the address-range arithmetic below uses unsigned pointer
 * comparison.  No floating-point or non-trivial math is required.
 */
#define L0_ASSERT_TENSOR_IN_BUFFER(tensor, ctx)                                        \
    do {                                                                               \
        const uintptr_t _t_start = reinterpret_cast<uintptr_t>((tensor)->data);       \
        const uintptr_t _t_end   = _t_start + static_cast<uintptr_t>(ggml_nbytes(tensor)); \
        const uintptr_t _b_start = reinterpret_cast<uintptr_t>((ctx)->device_ptr);    \
        const uintptr_t _b_end   = _b_start + static_cast<uintptr_t>((ctx)->size);    \
        if (_t_start < _b_start || _t_end > _b_end) {                                 \
            GGML_ABORT(                                                                \
                "[L0] tensor->data=%p .. +%zu is outside buffer [%p .. +%zu] "        \
                "(tensor=%s)",                                                         \
                (tensor)->data,                                                        \
                static_cast<size_t>(ggml_nbytes(tensor)),                             \
                (ctx)->device_ptr,                                                     \
                (ctx)->size,                                                           \
                (tensor)->name);                                                       \
        }                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// Debug-only macros — compiled out in production builds
// ---------------------------------------------------------------------------

#ifdef OLLAMA_L0_DEBUG_TENSOR_IO

/**
 * Emits a single trace line to stderr for each tensor I/O operation.
 *
 * Output format:
 *   [L0_TRACE] op=<op> tensor=<ptr> name=<name> data=<ptr>
 *              offset=<N> size=<N> shape=[ne0,ne1,ne2,ne3]
 *
 * Parameters:
 *   tensor  - pointer to ggml_tensor
 *   op      - C string: "init", "set", or "get"
 *   offset  - byte offset within the tensor buffer
 *   size    - byte count being transferred
 */
#define L0_TRACE_TENSOR_IO(tensor, op, offset, size)                         \
    do {                                                                     \
        fprintf(stderr,                                                      \
                "[L0_TRACE] op=%s tensor=%p name=%s data=%p "               \
                "offset=%zu size=%zu shape=[%lld,%lld,%lld,%lld]\n",        \
                (op),                                                        \
                static_cast<const void *>(tensor),                          \
                (tensor)->name,                                              \
                (tensor)->data,                                              \
                static_cast<size_t>(offset),                                 \
                static_cast<size_t>(size),                                   \
                static_cast<long long>((tensor)->ne[0]),                     \
                static_cast<long long>((tensor)->ne[1]),                     \
                static_cast<long long>((tensor)->ne[2]),                     \
                static_cast<long long>((tensor)->ne[3]));                    \
    } while (0)

/**
 * Prints the first n bytes of a HOST-READABLE buffer as hex to stderr.
 *
 * WARNING: this macro dereferences ptr from the host CPU.  NEVER call this
 * on a device-only (non-host-visible) L0 device pointer — the dereference
 * will cause an access violation / segfault.  Only safe when ptr points to
 * host-visible (shared or host-allocated) memory.
 *
 * Parameters:
 *   ptr  - host-readable pointer to the start of the buffer
 *   n    - number of bytes to hex-dump (keep small, e.g., 16 or 32)
 */
#define L0_HEXDUMP_FIRST_BYTES(ptr, n)                                       \
    do {                                                                     \
        const unsigned char *_p = reinterpret_cast<const unsigned char *>(ptr); \
        fprintf(stderr, "[L0_HEXDUMP] ptr=%p bytes=%zu : ", (const void *)_p, \
                static_cast<size_t>(n));                                     \
        for (size_t _i = 0; _i < static_cast<size_t>(n); ++_i) {            \
            fprintf(stderr, "%02x ", _p[_i]);                                \
        }                                                                    \
        fprintf(stderr, "\n");                                               \
    } while (0)

#else /* OLLAMA_L0_DEBUG_TENSOR_IO not defined — production-safe no-ops */

#define L0_TRACE_TENSOR_IO(tensor, op, offset, size)  ((void)0)
#define L0_HEXDUMP_FIRST_BYTES(ptr, n)                ((void)0)

#endif /* OLLAMA_L0_DEBUG_TENSOR_IO */

#endif /* OLLAMA_L0_TENSOR_DEBUG_H */
