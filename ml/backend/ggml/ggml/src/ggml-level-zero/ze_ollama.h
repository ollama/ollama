// SPDX-License-Identifier: MIT
// ze_ollama.h — Ollama-specific C ABI for Intel Level Zero backend.
// Thin, stable interface consumed by Go CGO and automation-engineer.
// NO C++ types. NO L0 types. NO templates. Pure C99.
// All C++ state is hidden behind the opaque ze_ollama_device_s Pimpl.
#ifndef ZE_OLLAMA_H
#define ZE_OLLAMA_H
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Export macro — Windows needs __declspec(dllexport) to emit these in the
// DLL's export table so the Go-side discover/level_zero_info.c dlopen+dlsym
// shim can resolve ze_ollama_* symbols at runtime.  Non-Windows relies on
// default visibility.
#if defined(_WIN32) && !defined(__MINGW32__)
#  define ZE_OLLAMA_API __declspec(dllexport)
#else
#  define ZE_OLLAMA_API __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Error codes returned by all ze_ollama_* functions.
// Ze_OLLAMA_OK (0) is the only success value; all other values are failures.
// Callers must not treat unknown positive values as success.
// ---------------------------------------------------------------------------
typedef enum ze_ollama_result_t {
    ZE_OLLAMA_OK                 = 0,
    ZE_OLLAMA_ERR_LOADER_MISSING = 1,  // ze_loader.so.1 / ze_loader.dll not found
    ZE_OLLAMA_ERR_NO_DEVICE      = 2,  // Driver present but zero usable devices
    ZE_OLLAMA_ERR_DRIVER_INIT    = 3,  // zeInit or zeDriverGet returned error
    ZE_OLLAMA_ERR_OOM            = 4,  // zeMemAllocDevice returned out-of-memory
    ZE_OLLAMA_ERR_UNSUPPORTED    = 5,  // Operation not supported on this device
    ZE_OLLAMA_ERR_INTERNAL       = 99  // Unclassified internal error
} ze_ollama_result_t;

// ---------------------------------------------------------------------------
// Device classification — distinguishes GPU (discrete/integrated Arc) from
// NPU (Meteor Lake / Lunar Lake VPU) and is stored in device_info.device_kind.
// CPU-type L0 devices are never enumerated (see ADR-L0-002).
// ---------------------------------------------------------------------------
typedef enum ze_ollama_device_kind_t {
    ZE_OLLAMA_DEV_GPU = 1,
    ZE_OLLAMA_DEV_NPU = 2
} ze_ollama_device_kind_t;

// ---------------------------------------------------------------------------
// POD device descriptor filled by ze_ollama_enumerate_devices().
// Fixed-size fields guarantee ABI stability across library versions.
// _reserved[5] provides forward-compat padding for future fields.
// sizeof(ze_ollama_device_info_t) == 320 bytes (verify in unit test).
// ---------------------------------------------------------------------------
typedef struct ze_ollama_device_info_t {
    char     name[256];       // Null-terminated device name from ze_device_properties_t
    char     uuid[37];        // UUID as hex string with dashes: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    uint64_t total_memory;    // Total device memory in bytes from ze_device_memory_properties_t
    uint64_t free_memory;     // Estimated free memory in bytes at enumeration time
    uint32_t compute_units;   // numEUsPerSubSlice * numSubSlicesPerSlice * numSlices
    uint32_t clock_mhz;       // core_clock_rate from ze_device_properties_t
    uint8_t  device_kind;     // ze_ollama_device_kind_t value (GPU=1, NPU=2)
    uint8_t  supports_fp16;   // Non-zero if ZE_DEVICE_FP_ATOMIC_FLAG_GLOBAL is supported
    uint8_t  supports_int8;   // Non-zero if INT8 dot-product is reported
    uint8_t  _reserved[5];    // Must be zero; reserved for future ABI extension
} ze_ollama_device_info_t;

// ---------------------------------------------------------------------------
// Opaque device handle. The body (ze_ollama_device_s) is declared only in
// ggml-level-zero.cpp — never exposed to Go or any C header.  This is the
// Pimpl pattern: Go sees only the pointer; all C++ state lives behind it.
// ---------------------------------------------------------------------------
typedef struct ze_ollama_device_s *ze_ollama_device_handle_t;

// ---------------------------------------------------------------------------
// API — 8 functions total.
//
// Thread-safety contract:
//   ze_ollama_init / ze_ollama_shutdown: call once per process from a single
//   thread before / after all other calls.
//   ze_ollama_enumerate_devices: safe to call concurrently after init.
//   ze_ollama_device_open / device_close / device_free_memory: safe to call
//   concurrently on different handles.
//   ze_ollama_result_str / ze_ollama_version: always thread-safe (read-only).
// ---------------------------------------------------------------------------

/**
 * Initialize the Level Zero backend.
 * Calls dlopen("libze_loader.so.1", RTLD_NOW|RTLD_LOCAL) on Linux or
 * LoadLibrary("ze_loader.dll") on Windows. On success resolves all required
 * L0 function pointers via dlsym / GetProcAddress and calls zeInit(0).
 * Protected internally by std::call_once — safe to call multiple times;
 * only the first call performs initialization.
 *
 * Returns:
 *   ZE_OLLAMA_OK               — backend ready for enumeration
 *   ZE_OLLAMA_ERR_LOADER_MISSING — ze_loader shared library not found on system
 *   ZE_OLLAMA_ERR_DRIVER_INIT  — ze_loader found but zeInit failed
 *   ZE_OLLAMA_ERR_INTERNAL     — unexpected failure during init
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_init(void);

/**
 * Enumerate all usable L0 devices (GPU + optionally NPU).
 * Fills out_buf with up to buf_cap entries in device-index order.
 * NPU (VPU-type) devices are included only when the calling process has set
 * the OLLAMA_L0_NPU_ENABLE environment variable to "1".
 * CPU-type L0 devices are always excluded.
 *
 * Parameters:
 *   out_buf   — caller-allocated array; must hold at least buf_cap elements
 *   buf_cap   — capacity of out_buf; 16 is the recommended minimum
 *   out_count — on success, set to the actual number of devices written
 *
 * Returns:
 *   ZE_OLLAMA_OK            — enumeration succeeded (out_count may be zero)
 *   ZE_OLLAMA_ERR_NO_DEVICE — zeDriverGet returned zero drivers
 *   ZE_OLLAMA_ERR_INTERNAL  — unexpected failure
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_enumerate_devices(
    ze_ollama_device_info_t *out_buf,
    size_t                   buf_cap,
    size_t                  *out_count);

/**
 * Open a device by its enumeration index (0-based, matching out_buf order
 * from ze_ollama_enumerate_devices). Creates a ze_context_handle_t and
 * ze_command_queue_handle_t bound to this device and stores them in the
 * opaque handle.
 *
 * Parameters:
 *   index — device index; must be < out_count from the last enumeration
 *   out   — receives the opaque handle on success; unchanged on failure
 *
 * Returns:
 *   ZE_OLLAMA_OK              — handle is valid and ready for use
 *   ZE_OLLAMA_ERR_NO_DEVICE   — index out of range
 *   ZE_OLLAMA_ERR_DRIVER_INIT — zeContextCreate or zeCommandQueueCreate failed
 *   ZE_OLLAMA_ERR_OOM         — insufficient memory for internal structures
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_device_open(
    uint32_t                  index,
    ze_ollama_device_handle_t *out);

/**
 * Close a device handle opened by ze_ollama_device_open.
 * Destroys the associated ze_command_queue_handle_t and ze_context_handle_t.
 * The handle is invalid after this call; passing it to any other function
 * is undefined behaviour.
 * Passing NULL is a no-op.
 */
ZE_OLLAMA_API void ze_ollama_device_close(ze_ollama_device_handle_t handle);

/**
 * Query the current free device memory for a handle returned by
 * ze_ollama_device_open. Calls zeDeviceGetMemoryProperties and subtracts
 * the backend's tracked live allocations from the hardware total.
 *
 * Parameters:
 *   handle    — valid handle from ze_ollama_device_open
 *   out_bytes — set to estimated free bytes on success
 *
 * Returns:
 *   ZE_OLLAMA_OK           — out_bytes is valid
 *   ZE_OLLAMA_ERR_INTERNAL — query failed
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_device_free_memory(
    ze_ollama_device_handle_t handle,
    uint64_t                 *out_bytes);

/**
 * Return a short human-readable description of an error code.
 * The returned pointer is valid for the lifetime of the process; callers
 * must not free it. Thread-safe (points into a static array).
 * Returns "unknown" for unrecognised codes.
 */
ZE_OLLAMA_API const char *ze_ollama_result_str(ze_ollama_result_t result);

/**
 * Return the semantic version string of this ze_ollama ABI implementation.
 * Format: "MAJOR.MINOR.PATCH" (e.g. "1.0.0"). Never NULL.
 * The returned pointer is valid for the lifetime of the process.
 */
ZE_OLLAMA_API const char *ze_ollama_version(void);

/**
 * Shut down the Level Zero backend.
 * Synchronises all outstanding command queues, destroys all open contexts,
 * and unloads the ze_loader shared library.
 * Must not be called while any device handle is in use.
 * After this call ze_ollama_init may be called again to re-initialise.
 *
 * Returns:
 *   ZE_OLLAMA_OK           — shutdown clean
 *   ZE_OLLAMA_ERR_INTERNAL — unexpected failure during teardown
 */
ZE_OLLAMA_API ze_ollama_result_t ze_ollama_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* ZE_OLLAMA_H */
