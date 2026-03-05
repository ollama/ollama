#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

#ifdef  __cplusplus
}
#endif
