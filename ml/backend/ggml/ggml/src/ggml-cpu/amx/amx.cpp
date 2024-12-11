#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)(buffer->context);
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (qtype_has_amx_kernels(tensor->type)) {
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *)tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_amx_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_amx_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "AMX";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = aligned_alloc(TENSOR_ALIGNMENT, size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    return ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

static bool ggml_backend_amx_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__gnu_linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#endif
}
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
            /* .is_host          = */ ggml_backend_amx_buffer_type_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    if (!ggml_amx_init()) {
        return NULL;
    }

    return &ggml_backend_buffer_type_amx;
}

bool ggml_backend_amx_buft_is_amx(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_amx_buffer_type_get_name;
}

bool ggml_backend_amx_device_supports_op(const struct ggml_tensor * op) {
    // handle only 2d gemm for now
    auto is_contiguous_2d = [](const struct ggml_tensor * t) {
        return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            const enum ggml_type type = src0->type;
            const int64_t ne0 = op->ne[0];

            // amx kernels enables for Q4_0, Q4_1, Q8_0, F16
            // Q4_K, Q5_K, Q6_K, IQ4_XS enabled for QK_K = 256
            bool has_amx_kernels = qtype_has_amx_kernels(type) || (type == GGML_TYPE_F16);

            bool can_use_amx =
                is_contiguous_2d(src0) &&       // src0 must be contiguous
                is_contiguous_2d(src1) &&       // src1 must be contiguous
                src1->type == GGML_TYPE_F32 &&  // src1 must be float32
                has_amx_kernels &&              // with amx kernel impls
                ne0 % (TILE_N * 2) == 0;        // out_features is 32x

            return can_use_amx;
        }
        default:
            return false;
    }
}

#endif // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
