#include <mutex>
#include <mudnn.h>

#include "mudnn.cuh"

namespace mudnn = musa::dnn;

// Returns a human-readable error string for mudnn::Status
const char* mudnnGetErrorString(mudnn::Status err) {
    switch (err) {
        case mudnn::Status::SUCCESS:
            return "Success";
        case mudnn::Status::INVALID_PARAMETER:
            return "Invalid parameter";
        case mudnn::Status::NOT_INITIALIZED:
            return "Not initialized";
        case mudnn::Status::ALLOC_FAILED:
            return "Allocation failed";
        case mudnn::Status::NOT_SUPPORTED:
            return "Not supported";
        case mudnn::Status::INTERNAL_ERROR:
            return "Internal error";
        case mudnn::Status::ARCH_MISMATCH:
            return "Architecture mismatch";
        case mudnn::Status::EXECUTION_FAILED:
            return "Execution failed";
        default:
            return "Unknown mudnn status";
    }
}

// Error checking macro for MUDNN calls
#define MUDNN_CHECK(err) CUDA_CHECK_GEN(err, mudnn::Status::SUCCESS, mudnnGetErrorString)

namespace {
    // Thread-safe cache for mudnn::Handle objects per device
    std::unordered_map<int, std::unique_ptr<mudnn::Handle>> handle_cache;
    std::mutex handle_cache_mutex;

    mudnn::Handle* get_cached_handle(int device_id) {
        std::lock_guard<std::mutex> lock(handle_cache_mutex);
        auto it = handle_cache.find(device_id);
        if (it != handle_cache.end()) {
            return it->second.get();
        }
        auto handle = std::make_unique<mudnn::Handle>(device_id);
        mudnn::Handle* handle_ptr = handle.get();
        handle_cache[device_id] = std::move(handle);
        return handle_ptr;
    }
}

// Extracts dimensions and strides from a ggml_tensor
int get_ggml_dims_and_strides(const ggml_tensor* tensor,
                              std::vector<int64_t>& dims,
                              std::vector<int64_t>& strides) {
    const int ndims = ggml_n_dims(tensor);
    const size_t element_size = ggml_element_size(tensor);

    dims.resize(ndims);
    strides.resize(ndims);

    for (int i = 0; i < ndims; ++i) {
        dims[i] = tensor->ne[i];
        strides[i] = tensor->nb[i] / static_cast<int64_t>(element_size);
    }
    return ndims;
}

// Converts ggml_type to mudnn::Tensor::Type
mudnn::Tensor::Type ggml_type_to_mudnn_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return mudnn::Tensor::Type::FLOAT;
        case GGML_TYPE_F16:
            return mudnn::Tensor::Type::HALF;

        // TODO: Add support for other types

        default:
            MUDNN_CHECK(mudnn::Status::NOT_SUPPORTED);
    }

    return mudnn::Tensor::Type::FLOAT; // Default fallback
}

// Asynchronous memory copy using mudnn::Unary::IDENTITY
musaError_t mudnnMemcpyAsync(ggml_backend_cuda_context& ctx, const ggml_tensor* dst, const ggml_tensor* src) {
    mudnn::Tensor tensor_dst, tensor_src;

    MUDNN_CHECK(tensor_dst.SetType(ggml_type_to_mudnn_type(dst->type)));
    MUDNN_CHECK(tensor_src.SetType(ggml_type_to_mudnn_type(src->type)));

    std::vector<int64_t> dims, strides;
    const int ndims = get_ggml_dims_and_strides(src, dims, strides);

    MUDNN_CHECK(tensor_dst.SetNdInfo(ndims, dims.data(), strides.data()));
    MUDNN_CHECK(tensor_src.SetNdInfo(ndims, dims.data(), strides.data()));
    MUDNN_CHECK(tensor_dst.SetAddr(dst->data));
    MUDNN_CHECK(tensor_src.SetAddr(src->data));

    mudnn::Unary op;
    MUDNN_CHECK(op.SetMode(mudnn::Unary::Mode::IDENTITY));
    MUDNN_CHECK(op.SetAlpha(0.0f));
    MUDNN_CHECK(op.SetBeta(0.0f));

    mudnn::Handle* handle = get_cached_handle(ctx.device);
    MUDNN_CHECK(handle->SetStream(ctx.stream()));
    MUDNN_CHECK(op.Run(*handle, tensor_dst, tensor_src));

    return musaSuccess;
}
