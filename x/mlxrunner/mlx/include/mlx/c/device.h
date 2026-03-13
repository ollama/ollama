/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DEVICE_H
#define MLX_DEVICE_H

#include <stdbool.h>
#include <stddef.h>

#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_device Device
 * MLX device object.
 */
/**@{*/

/**
 * A MLX device object.
 */
typedef struct mlx_device_ {
  void* ctx;
} mlx_device;

/**
 * Device type.
 */
typedef enum mlx_device_type_ { MLX_CPU, MLX_GPU } mlx_device_type;

/**
 * Returns a new empty device.
 */
mlx_device mlx_device_new(void);

/**
 * Returns a new device of specified `type`, with specified `index`.
 */
mlx_device mlx_device_new_type(mlx_device_type type, int index);
/**
 * Free a device.
 */
int mlx_device_free(mlx_device dev);
/**
 * Set device to provided src device.
 */
int mlx_device_set(mlx_device* dev, const mlx_device src);
/**
 * Get device description.
 */
int mlx_device_tostring(mlx_string* str, mlx_device dev);
/**
 * Check if devices are the same.
 */
bool mlx_device_equal(mlx_device lhs, mlx_device rhs);
/**
 * Returns the index of the device.
 */
int mlx_device_get_index(int* index, mlx_device dev);
/**
 * Returns the type of the device.
 */
int mlx_device_get_type(mlx_device_type* type, mlx_device dev);
/**
 * Returns the default MLX device.
 */
int mlx_get_default_device(mlx_device* dev);
/**
 * Set the default MLX device.
 */
int mlx_set_default_device(mlx_device dev);
/**
 * Check if device is available.
 */
int mlx_device_is_available(bool* avail, mlx_device dev);
/**
 * Get the number of available devices for a device type.
 */
int mlx_device_count(int* count, mlx_device_type type);

/**
 * A MLX device info object.
 * Contains key-value pairs with device properties.
 * Keys vary by backend but common keys include:
 *   - device_name (string): Device name
 *   - architecture (string): Architecture identifier
 * Additional keys may be present depending on the backend.
 */
typedef struct mlx_device_info_ {
  void* ctx;
} mlx_device_info;

/**
 * Returns a new empty device info object.
 */
mlx_device_info mlx_device_info_new(void);
/**
 * Get device information for a device.
 */
int mlx_device_info_get(mlx_device_info* info, mlx_device dev);
/**
 * Free a device info object.
 */
int mlx_device_info_free(mlx_device_info info);
/**
 * Check if a key exists in the device info.
 * Returns 0 on success, 1 on error.
 * Sets *exists to true if the key exists, false otherwise.
 */
int mlx_device_info_has_key(
    bool* exists,
    mlx_device_info info,
    const char* key);
/**
 * Check if a value is a string type.
 * Returns 0 on success, 1 on error.
 * Sets *is_string to true if the value is a string, false if it's a size_t.
 */
int mlx_device_info_is_string(
    bool* is_string,
    mlx_device_info info,
    const char* key);
/**
 * Get a string value from device info.
 * Returns 0 on success, 1 on error, 2 if key not found or wrong type.
 */
int mlx_device_info_get_string(
    const char** value,
    mlx_device_info info,
    const char* key);
/**
 * Get a size_t value from device info.
 * Returns 0 on success, 1 on error, 2 if key not found or wrong type.
 */
int mlx_device_info_get_size(
    size_t* value,
    mlx_device_info info,
    const char* key);
/**
 * Get all keys from device info.
 * Returns 0 on success, 1 on error.
 */
int mlx_device_info_get_keys(mlx_vector_string* keys, mlx_device_info info);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
