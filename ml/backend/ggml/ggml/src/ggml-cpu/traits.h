#pragma once
#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
#    include <vector>
extern "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
