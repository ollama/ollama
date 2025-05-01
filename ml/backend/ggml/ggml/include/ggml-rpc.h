#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    2
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0
#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint,
                                                    const char * cache_dir,
                                                    size_t free_mem, size_t total_mem);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);

GGML_BACKEND_API ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint);

#ifdef  __cplusplus
}
#endif
