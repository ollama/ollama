#include "ggml-backend.h"
#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void);
bool ggml_backend_amx_buft_is_amx(ggml_backend_buffer_type_t buft);
bool ggml_backend_amx_device_supports_op(const struct ggml_tensor * op);
void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

#endif

#ifdef __cplusplus
}
#endif
