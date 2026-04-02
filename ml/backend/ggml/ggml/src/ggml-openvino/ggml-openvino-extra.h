#pragma once

#include "ggml.h"
#include "openvino/runtime/core.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cstdlib>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <string>

// ExtraQuantType enum - defines requantization target formats
enum class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q8_0_C, Q8_0_32 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_queue ggml_openvino_get_cl_queue();

// Intel USM extension function type
typedef cl_int(CL_API_CALL * clEnqueueMemFillINTEL_fn)(cl_command_queue queue,
                                                       void * dst_ptr,
                                                       const void * pattern,
                                                       size_t pattern_size,
                                                       size_t size,
                                                       cl_uint num_events_in_wait_list,
                                                       const cl_event * event_wait_list,
                                                       cl_event * event);

typedef cl_int(CL_API_CALL * clEnqueueMemcpyINTEL_fn)(cl_command_queue queue,
                                                      cl_bool blocking,
                                                      void * dst_ptr,
                                                      const void * src_ptr,
                                                      size_t size,
                                                      cl_uint num_events_in_wait_list,
                                                      const cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

struct ggml_openvino_device_config {
    std::string device_name = "CPU";
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    ov::AnyMap compile_config;
    cl_command_queue cl_queue = nullptr;

    void init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config();

// Get the device name
const std::string & ggml_openvino_get_device_name();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
struct ggml_openvino_extra_base {
    enum class Type { WEIGHT, QUANTIZED_WEIGHT, TENSOR };
    Type type;
    virtual ~ggml_openvino_extra_base() = default;
protected:
    explicit ggml_openvino_extra_base(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
struct ggml_openvino_weight_extra : public ggml_openvino_extra_base {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::Node> weight_node;  // Pre-built OpenVINO weight node

    ggml_openvino_weight_extra(ov::Tensor w, std::shared_ptr<ov::Node> n) :
        ggml_openvino_extra_base(Type::WEIGHT),
        weights(std::move(w)),
        weight_node(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
struct ggml_openvino_quantized_weight_extra : public ggml_openvino_extra_base {
    ov::Tensor weights;   // U4 or U8 extracted weights
    ov::Tensor scales;    // F16 scales
    ov::Tensor zp;        // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::Node> weight_node;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_weight_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::Node> n) :
        ggml_openvino_extra_base(Type::QUANTIZED_WEIGHT),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        weight_node(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
struct ggml_openvino_tensor_extra : public ggml_openvino_extra_base {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t)
        : ggml_openvino_extra_base(Type::TENSOR), tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

struct ggml_openvino_extracted_layout {
    size_t total_size = 0;      // Total bytes needed
    size_t weights_offset = 0;  // Offset to weights in buffer
    size_t weights_size = 0;    // Size of weights in bytes
    size_t scales_offset = 0;   // Offset to scales in buffer
    size_t scales_size = 0;     // Size of scales in bytes
    size_t zp_offset = 0;       // Offset to zero points in buffer
    size_t zp_size = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_per_block;  // weights per scale/zp block
    bool is_symmetric;        // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                      // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;   // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(const ggml_tensor * tensor, bool is_remote);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
void ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
struct ggml_backend_openvino_context {
    int device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<void> runtime_context = nullptr;

    ggml_backend_openvino_context() = default;
};
