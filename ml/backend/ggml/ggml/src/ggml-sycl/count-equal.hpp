#ifndef GGML_SYCL_COUNT_EQUAL_HPP
#define GGML_SYCL_COUNT_EQUAL_HPP
#include "common.hpp"

#define SYCL_COUNT_EQUAL_CHUNK_SIZE 128

void ggml_sycl_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif //GGML_SYCL_COUNT_EQUAL_HPP
