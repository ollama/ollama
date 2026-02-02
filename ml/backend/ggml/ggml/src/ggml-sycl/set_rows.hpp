#ifndef GGML_SYCL_SET_ROWS_HPP
#define GGML_SYCL_SET_ROWS_HPP

#include "common.hpp"

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_SET_ROWS_HPP
