// SPDX-License-Identifier: MIT
// level_zero_info.h — thin C shim consumed by Go CGO.
// Reproduces the types from ze_ollama.h inline so that the discover/ package
// does NOT need the full ggml subtree on its include path.  The shim
// dynamically loads libggml-level-zero at runtime via dlopen/LoadLibrary.
// No Intel L0 SDK headers are required to compile this file.
#ifndef LEVEL_ZERO_INFO_H
#define LEVEL_ZERO_INFO_H
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Mirror of ze_ollama_result_t from ze_ollama.h — kept in sync by
// automation-engineer (do not add values without updating ze_ollama.h).
// ---------------------------------------------------------------------------
typedef enum lz_result_t {
    LZ_OK                 = 0,
    LZ_ERR_LOADER_MISSING = 1,
    LZ_ERR_NO_DEVICE      = 2,
    LZ_ERR_DRIVER_INIT    = 3,
    LZ_ERR_OOM            = 4,
    LZ_ERR_UNSUPPORTED    = 5,
    LZ_ERR_INTERNAL       = 99
} lz_result_t;

// ---------------------------------------------------------------------------
// Mirror of ze_ollama_device_kind_t.
// ---------------------------------------------------------------------------
typedef enum lz_device_kind_t {
    LZ_DEV_GPU = 1,
    LZ_DEV_NPU = 2
} lz_device_kind_t;

// ---------------------------------------------------------------------------
// Mirror of ze_ollama_device_info_t.
// sizeof == 320 bytes (matches ze_ollama_device_info_t).
// ---------------------------------------------------------------------------
typedef struct lz_device_info_t {
    char     name[256];
    char     uuid[37];
    uint64_t total_memory;
    uint64_t free_memory;
    uint32_t compute_units;
    uint32_t clock_mhz;
    uint8_t  device_kind;
    uint8_t  supports_fp16;
    uint8_t  supports_int8;
    uint8_t  _reserved[5];
} lz_device_info_t;

// ---------------------------------------------------------------------------
// Opaque device handle — mirrors ze_ollama_device_handle_t.
// Stored as uintptr_t on the Go side; cast here for the C API.
// ---------------------------------------------------------------------------
typedef struct lz_device_s *lz_device_handle_t;

// ---------------------------------------------------------------------------
// Shim API — loaded dynamically from libggml-level-zero.so / ggml-level-zero.dll.
// Returns LZ_ERR_LOADER_MISSING when the shared library is absent.
// ---------------------------------------------------------------------------

lz_result_t lz_init(void);

lz_result_t lz_enumerate_devices(
    lz_device_info_t *out_buf,
    size_t            buf_cap,
    size_t           *out_count);

lz_result_t lz_device_open(
    uint32_t          index,
    lz_device_handle_t *out);

void lz_device_close(lz_device_handle_t handle);

lz_result_t lz_device_free_memory(
    lz_device_handle_t handle,
    uint64_t          *out_bytes);

const char *lz_result_str(lz_result_t result);

const char *lz_version(void);

lz_result_t lz_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* LEVEL_ZERO_INFO_H */
