#include "mmq.cuh"

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t nb01 = src0->nb[1];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride00 = nb01 / ggml_type_size(src0->type);

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const mmq_args args = {src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stride00, src1_padded_row_size, src1_ncols, nrows_dst};

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(args, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_supports_mmq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}
