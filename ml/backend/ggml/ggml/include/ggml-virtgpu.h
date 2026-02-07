#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_REMOTING_FRONTEND_NAME "RemotingFrontend"

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_virtgpu_reg();

#ifdef  __cplusplus
}
#endif
