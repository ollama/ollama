#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_KOMPUTE_MAX_DEVICES 16

struct ggml_vk_device {
    int index;
    int type; // same as VkPhysicalDeviceType
    size_t heapSize;
    const char * name;
    const char * vendor;
    int subgroupSize;
    uint64_t bufferAlignment;
    uint64_t maxAlloc;
};

struct ggml_vk_device * ggml_vk_available_devices(size_t memoryRequired, size_t * count);
bool ggml_vk_get_device(struct ggml_vk_device * device, size_t memoryRequired, const char * name);
bool ggml_vk_has_vulkan(void);
bool ggml_vk_has_device(void);
struct ggml_vk_device ggml_vk_current_device(void);

//
// backend API
//

// forward declaration
typedef struct ggml_backend * ggml_backend_t;

GGML_BACKEND_API ggml_backend_t ggml_backend_kompute_init(int device);

GGML_BACKEND_API bool ggml_backend_is_kompute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_kompute_reg(void);

#ifdef __cplusplus
}
#endif
