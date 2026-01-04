#include <string.h>
#include <inttypes.h>

#include "ollama-debug.h"

static int mul(int64_t *dims, int ndims) {
    int result = 1;
    for (int i = 0; i < ndims; i++) {
        result *= dims[i];
    }

    return result;
}

static void repeat(char c, int n) {
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "%c", c);
    }
}

static void print_tensor(const void *tensor, void (*cb)(const void *, int),
                         int shape,
                         int64_t *dims, int ndims, int stride,
                         int nitems, int pad) {
    fprintf(stderr, "[");
    for (int i = 0; i < dims[0]; i++) {
        if (i >= nitems && i < dims[0] - nitems) {
            fprintf(stderr, "... (%" PRIi64 " more), ", dims[0] - 2 * nitems);
            int skip = dims[0] - 2 * nitems;
            if (ndims > 1) {
                stride += mul(dims + 1, ndims - 1) * skip;
                repeat('\n', ndims - 1);
                repeat(' ', shape - ndims + 1 + pad);
            }
            i += skip - 1;
        } else if (ndims > 1) {
            print_tensor(tensor, cb, shape, dims + 1, ndims - 1, stride,
                         nitems, pad);
            stride += mul(dims + 1, ndims - 1);
            if (i < dims[0] - 1) {
                fprintf(stderr, ", ");
                repeat('\n', ndims - 1);
                repeat(' ', shape - ndims + 1 + pad);
            }
        } else {
            cb(tensor, stride + i);
            if (i < dims[0] - 1) {
                fprintf(stderr, ", ");
            }
        }
    }
    fprintf(stderr, "]");
}

static void print_tensor_f16(const void *tensor, int i) {
    float value = ggml_fp16_to_fp32(((const ggml_fp16_t *)tensor)[i]);
    fprintf(stderr, "%s%f", value < 0 ? "" : " ", value);
}

static void print_tensor_f32(const void *tensor, int i) {
    float value = ((const float *)tensor)[i];
    fprintf(stderr, "%s%f", value < 0 ? "" : " ", value);
}

static void print_tensor_i32(const void *tensor, int i) {
    int32_t value = ((const int32_t *)tensor)[i];
    fprintf(stderr, "%s%d", value < 0 ? "" : " ", value);
}

static void ollama_debug_tensor(const struct ggml_tensor *tensor, bool verbose, const char *prefix, int indent) {
    fprintf(stderr, "%s%s %s (%s): [%" PRIi64 " %" PRIi64 " %" PRIi64 " %" PRIi64 "]\n", prefix, tensor->name,
            ggml_op_name(tensor->op), ggml_type_name(tensor->type), tensor->ne[0],
            tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    if (!verbose) {
        return;
    }

    for (int i = 0; i < indent; i++) {
        fprintf(stderr, " ");
    }

    switch (tensor->type) {
    case GGML_TYPE_F16:
        print_tensor(ggml_get_data(tensor), print_tensor_f16, ggml_n_dims(tensor),
                     (int64_t *)tensor->ne, ggml_n_dims(tensor), 0, 3, indent);
        break;
    case GGML_TYPE_F32:
        print_tensor(ggml_get_data(tensor), print_tensor_f32, ggml_n_dims(tensor),
                     (int64_t *)tensor->ne, ggml_n_dims(tensor), 0, 3, indent);
        break;
    case GGML_TYPE_I32:
        print_tensor(ggml_get_data(tensor), print_tensor_i32, ggml_n_dims(tensor),
                     (int64_t *)tensor->ne, ggml_n_dims(tensor), 0, 3, indent);
        break;
    default:
        fprintf(stderr, "<unsupported type>\n");
        return;
    }

    fprintf(stderr, "\n");
}

void ollama_debug(const struct ggml_tensor *tensor, bool verbose) {
    ollama_debug_tensor(tensor, verbose, ">>> ", 4);

    for (int i = 0; i < GGML_MAX_SRC && tensor->src[i] != NULL; ++i) {
        char src[8];
        const int n = snprintf(src, sizeof(src), " src%d ", i);
        if (n >= sizeof(src)) {
            src[sizeof(src) - 1] = '\0';
        }

        ollama_debug_tensor(tensor->src[i], verbose, src, 4);
    }
}
