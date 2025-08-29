#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <array>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OPENVINO_NAME "OPENVINO"
#define GGML_OPENVINO_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device);

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU
// and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_openvino_get_device_count(void);
// GGML_BACKEND_API void ggml_backend_openvino_get_device_description(int device, char * description,
//                                                                    size_t description_size);
// GGML_BACKEND_API void ggml_backend_openvino_get_device_memory(int device, size_t * free, size_t * total);

// GGML_BACKEND_API bool ggml_backend_openvino_register_host_buffer(void * buffer, size_t size);
// GGML_BACKEND_API void ggml_backend_openvino_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void);

struct ggml_openvino_device_info {
    int device_count;

    struct openvino_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
    };

    openvino_device_info devices[GGML_OPENVINO_MAX_DEVICES] = {};

    std::array<float, GGML_OPENVINO_MAX_DEVICES> default_tensor_split = {};
};

const ggml_openvino_device_info & ggml_openvino_info();

#ifdef __cplusplus
}
#endif
