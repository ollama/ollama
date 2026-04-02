#include "ggml-openvino-extra.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <cstring>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <optional>

ov::Core & ov_singleton_core() {
    static ov::Core core;
    return core;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

void ggml_openvino_device_config::init() {
    if (initialized) {
        return;
    }
    device_name = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "CPU";
    auto available_devices = ov_singleton_core().get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");

    auto * cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (device_name == "NPU") {
        compile_config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
        }
    } else if (cache_dir) {
        ov_singleton_core().set_property(ov::cache_dir(cache_dir));
    }

    // Initialize remote context with queue sharing for GPU
    if (device_name == "GPU") {
        // Create OpenCL context and queue
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\n", err);
            return;
        }

        cl_device_id cl_device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL device: %d\n", err);
            return;
        }

        cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL context: %d\n", err);
            return;
        }

        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL command queue: %d\n", err);
            clReleaseContext(cl_ctx);
            return;
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_queue);

        // Release the context (queue keeps a reference)
        clReleaseContext(cl_ctx);
    } else if (device_name == "NPU") {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }

    initialized = true;
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    if (cl_queue != nullptr) {
        clReleaseCommandQueue(cl_queue);
        cl_queue = nullptr;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
const std::string & ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

// Check if running on NPU
bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    return ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config() {
    return ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_queue ggml_openvino_get_cl_queue() {
    return ggml_openvino_get_device_config().cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer (lazy load)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    static clEnqueueMemFillINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
        }
    }
    return fn;
}

// Get the clEnqueueMemcpyINTEL function pointer (lazy load)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    static clEnqueueMemcpyINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");
        }
    }
    return fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant) {
    if (no_requant) {
        return std::nullopt;
    }
    if (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        return ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 : ExtraQuantType::Q8_0_C);
    }
    if (strncmp(tensor->name, "output.weight", 13) == 0) {
        return ExtraQuantType::Q8_0_C;
    }
    if (ggml_openvino_is_npu()) {
        return ExtraQuantType::Q4_0_128;
    }
    switch (tensor->type) {
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q5_K:
        return ExtraQuantType::Q8_0_C;
    default:
        return std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = false;

    if (!ggml_is_quantized(tensor->type)) {
        return layout;
    }

    // Only handle 2D weight tensors
    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return layout;
    }

    int64_t n_elements = ggml_nelements(tensor);
    const size_t alignment = 64;  // Good for SIMD

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    if (requant_type.has_value()) {
        layout.is_requant = true;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        if (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            return layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        switch (requant_type.value()) {
        case ExtraQuantType::Q4_0_128:
            layout.is_u4 = true;
            layout.weights_per_block = 128;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q4_0_C:
            layout.is_u4 = true;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_32:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_1_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            break;
        default:
            layout.weights_per_block = -1;
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
            break;
        }

        if (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * sizeof(uint16_t);
            // For symmetric quantization, we only need one zp value (not one per block)
            // Zero points are stored in U4 or U8 format matching the weight type
            size_t n_zp_elements = layout.is_symmetric ? 1 : n_blocks;
            layout.zp_size = layout.is_u4 ? ((n_zp_elements + 1) / 2) : n_zp_elements;

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            return layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = false;
    layout.weights_per_block = 32;
    layout.is_symmetric = false;

    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        layout.is_u4 = true;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        layout.is_u4 = true;
        break;

    case GGML_TYPE_Q8_0:
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_K:
        break;

    default:
        // Unsupported quantization type
        return layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * sizeof(uint16_t);  // F16 = 2 bytes
    // Zero points: U4 or U8 matching weight type
    // For symmetric quantization, we only need one zp value (not one per block)
    size_t n_zp_elements = layout.is_symmetric ? 1 : n_blocks;
    layout.zp_size = layout.is_u4 ? ((n_zp_elements + 1) / 2) : n_zp_elements;

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    return layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(const ggml_tensor * tensor, bool is_remote) {
    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    switch (tensor->type) {
    case GGML_TYPE_F32:
        element_type = ov::element::f32;
        break;
    case GGML_TYPE_F16:
        element_type = ov::element::f16;
        break;
    case GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        break;
    case GGML_TYPE_I32:
        element_type = ov::element::i32;
        break;
    case GGML_TYPE_I64:
        element_type = ov::element::i64;
        break;
    default:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        return nullptr;
    }

    const auto & device_name = ggml_openvino_get_device_name();
    auto remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    if (is_remote) {
        GGML_ASSERT(device_name == "GPU");
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } else {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    return new ggml_openvino_tensor_extra(ov_tensor);
}
