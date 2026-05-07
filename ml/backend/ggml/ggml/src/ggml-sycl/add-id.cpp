#include <sycl/sycl.hpp>
#include "common.hpp"
#include "add-id.hpp"

static void add_id_kernel(
    const float* src0,
    const float* src1,
    const int32_t* src2,
    float* dst,
    int64_t ne0,
    int64_t ne1,
    size_t nb01,
    size_t nb02,
    size_t nb11,
    size_t nb21,
    sycl::nd_item<3> item_ct1) {
  const int64_t i1 = item_ct1.get_group(2);
  const int64_t i2 = item_ct1.get_group(1);

  const int i11 =
      *(const int32_t*)((const char*)src2 + i1 * sizeof(int32_t) + i2 * nb21);

  const size_t nb1 = ne0 * sizeof(float);
  const size_t nb2 = ne1 * nb1;

  float* dst_row = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
  const float* src0_row =
      (const float*)((const char*)src0 + i1 * nb01 + i2 * nb02);
  const float* src1_row = (const float*)((const char*)src1 + i11 * nb11);

  for (int64_t i0 = item_ct1.get_local_id(2); i0 < ne0;
       i0 += item_ct1.get_local_range(2)) {
    dst_row[i0] = src0_row[i0] + src1_row[i0];
  }
}

void ggml_sycl_add_id(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
  const ggml_tensor* src0 = dst->src[0];
  const ggml_tensor* src1 = dst->src[1];
  const ggml_tensor* src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));
  GGML_ASSERT(nb20 == sizeof(int32_t));

  const float* src0_d = (const float*)src0->data;
  const float* src1_d = (const float*)src1->data;
  const int32_t* src2_d = (const int32_t*)src2->data;
  float* dst_d = (float*)dst->data;

  int threads = std::min((int)ne00, 768);  // cols
  ctx.stream()->parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, ne02, ne01) * sycl::range<3>(1, 1, threads),
          sycl::range<3>(1, 1, threads)),
      [=](sycl::nd_item<3> item_ct1) {
        add_id_kernel(
            src0_d,
            src1_d,
            src2_d,
            dst_d,
            ne0,
            ne1,
            nb01,
            nb02,
            nb11,
            nb21,
            item_ct1);
      });
}
