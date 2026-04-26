// SPDX-License-Identifier: MIT
// ggml-level-zero.h — Public GGML interface for the Intel Level Zero backend.
// Mirrors the shape of ggml-vulkan.h. Included by runner/ and ml/backend/.
// The implementation is in ml/backend/ggml/ggml/src/ggml-level-zero/.
#ifndef GGML_LEVEL_ZERO_H
#define GGML_LEVEL_ZERO_H
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialise a GGML backend instance bound to the Level Zero device at
 * device_id (0-based index matching ze_ollama_enumerate_devices order).
 *
 * Returns NULL when:
 *   - ze_loader is not present on this system
 *   - device_id is out of range
 *   - zeContextCreate or zeCommandQueueCreate fails
 *
 * The returned ggml_backend_t must be released with ggml_backend_free().
 */
GGML_BACKEND_API ggml_backend_t ggml_backend_level_zero_init(int device_id);

/**
 * Returns true when backend was created by ggml_backend_level_zero_init().
 * Safe to call with any ggml_backend_t including NULL.
 */
GGML_BACKEND_API bool ggml_backend_is_level_zero(ggml_backend_t backend);

/**
 * Return the number of Level Zero devices visible to this process.
 * Calls ze_ollama_init() if necessary.  Thread-safe after the first call.
 * Returns 0 when ze_loader is absent or no compatible device is found.
 */
GGML_BACKEND_API int ggml_backend_level_zero_get_device_count(void);

/**
 * Retrieve a human-readable description string for device_id.
 * The pointer is valid until the next call with the same device_id from the
 * same thread.  Returns "<invalid>" for out-of-range IDs.
 */
GGML_BACKEND_API const char *ggml_backend_level_zero_get_device_description(int device_id);

#ifdef __cplusplus
}
#endif
#endif /* GGML_LEVEL_ZERO_H */
