#include "ggml-vulkan.h"
#include <vulkan/vulkan_core.h>
#if defined(GGML_VULKAN_RUN_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#include <chrono>
#include "ggml-cpu.h"
#endif

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <sstream>
#include <utility>
#include <memory>
#include <limits>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <future>
#include <thread>

#if defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#elif defined(__clang__) || defined(__GNUC__)
# if defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  if defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  else
#   define YIELD() asm volatile("yield")
#  endif
# endif
#endif

#if !defined(YIELD)
#define YIELD()
#endif

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-vulkan-shaders.hpp"

// remove this once it's more widely available in the SDK
#if !defined(VK_KHR_shader_bfloat16)

#define VK_KHR_shader_bfloat16 1
#define VK_KHR_SHADER_BFLOAT16_SPEC_VERSION                          1
#define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#define VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shaderBFloat16DotProduct;
    VkBool32                              shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#endif

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
static bool is_pow2(uint32_t x) { return x > 1 && (x & (x-1)) == 0; }

#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_APPLE 0x106b
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_DEVICE_DESCRIPTOR_POOL_SIZE 256

#define GGML_VK_MAX_NODES 8192

#define MAX_VK_BUFFERS 256

#define VK_CHECK(err, msg)                                          \
    do {                                                            \
        vk::Result err_ = (err);                                    \
        if (err_ != vk::Result::eSuccess) {                         \
            fprintf(stderr, "ggml_vulkan: %s error %s at %s:%d\n",  \
                #err, to_string(err_).c_str(), __FILE__, __LINE__); \
            exit(1);                                                \
        }                                                           \
    } while (0)

#ifdef GGML_VULKAN_DEBUG
#define VK_LOG_DEBUG(msg) std::cerr << msg << std::endl
#else
#define VK_LOG_DEBUG(msg) ((void) 0)
#endif // GGML_VULKAN_DEBUG

struct ggml_backend_vk_context;

#define MAX_PARAMETER_COUNT 8

struct vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // set to true to request the pipeline is compiled after the dryrun
    bool needed {};
    // set to true when the shader has been compiled
    bool compiled {};
};

typedef std::shared_ptr<vk_pipeline_struct> vk_pipeline;
typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);

struct vk_matmul_pipeline_struct {
    vk_pipeline l, m, s;
    vk_pipeline a_l, a_m, a_s;
};

typedef std::shared_ptr<vk_matmul_pipeline_struct> vk_matmul_pipeline;

struct vk_matmul_pipeline2 {
    vk_matmul_pipeline2() {
        f16acc = std::make_shared<vk_matmul_pipeline_struct>();
        f32acc = std::make_shared<vk_matmul_pipeline_struct>();
    }
    vk_matmul_pipeline f32acc;
    vk_matmul_pipeline f16acc;
};

struct vk_device_struct;
typedef std::shared_ptr<vk_device_struct> vk_device;
typedef std::weak_ptr<vk_device_struct> vk_device_ref;

struct vk_buffer_struct;
typedef std::shared_ptr<vk_buffer_struct> vk_buffer;
typedef std::weak_ptr<vk_buffer_struct> vk_buffer_ref;

struct ggml_backend_vk_buffer_type_context {
    std::string name;
    vk_device device;
};

struct vk_queue;

// Stores command pool/buffers. There's an instance of this
// for each (context,queue) pair and for each (device,queue) pair.
struct vk_command_pool {
    void init(vk_device& device, vk_queue *q_);
    void destroy(vk::Device& device);

    vk::CommandPool pool;
    uint32_t cmd_buffer_idx;
    std::vector<vk::CommandBuffer> cmd_buffers;

    vk_queue *q;
};

// Prevent simultaneous submissions to the same queue.
// This could be per vk_queue if we stopped having two vk_queue structures
// sharing the same vk::Queue.
static std::mutex queue_mutex;

struct vk_queue {
    uint32_t queue_family_index;
    vk::Queue queue;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;

    // copy everything except the cmd_pool
    void copyFrom(vk_queue &other) {
        queue_family_index = other.queue_family_index;
        queue = other.queue;
        stage_flags = other.stage_flags;
        transfer_only = other.transfer_only;
    }
};

static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft);
static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
static ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

#ifdef GGML_VULKAN_MEMORY_DEBUG
class vk_memory_logger;
#endif
class vk_perf_logger;
static void ggml_vk_destroy_buffer(vk_buffer& buf);

static constexpr uint32_t mul_mat_vec_max_cols = 8;
static constexpr uint32_t p021_max_gqa_ratio = 8;

enum vk_device_architecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    INTEL_XE2,
    NVIDIA_PRE_TURING,
};

// HSK x HSV
enum FaHeadSizes {
    FA_HEAD_SIZE_64,
    FA_HEAD_SIZE_80,
    FA_HEAD_SIZE_96,
    FA_HEAD_SIZE_112,
    FA_HEAD_SIZE_128,
    FA_HEAD_SIZE_192,
    FA_HEAD_SIZE_192_128,
    FA_HEAD_SIZE_256,
    FA_HEAD_SIZE_576_512,
    FA_HEAD_SIZE_UNSUPPORTED,
    FA_HEAD_SIZE_COUNT = FA_HEAD_SIZE_UNSUPPORTED,
};

static vk_device_architecture get_device_architecture(const vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    if (props.vendorID == VK_VENDOR_ID_AMD) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool amd_shader_core_properties = false;
        bool integer_dot_product = false;
        bool subgroup_size_control = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = true;
            } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_product = true;
            } else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = true;
            }
        }

        if (!amd_shader_core_properties || !integer_dot_product || !subgroup_size_control) {
            return vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

        props2.pNext = &shader_core_props_amd;
        shader_core_props_amd.pNext = &integer_dot_props;
        integer_dot_props.pNext = &subgroup_size_control_props;

        device.getProperties2(&props2);

        if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 64) {
            return vk_device_architecture::AMD_GCN;
        }
        if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 32) {
            // RDNA
            if (shader_core_props_amd.wavefrontsPerSimd == 20) {
                return vk_device_architecture::AMD_RDNA1;
            }
            if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
                return vk_device_architecture::AMD_RDNA3;
            }
            return vk_device_architecture::AMD_RDNA2;
        }
    } else if (props.vendorID == VK_VENDOR_ID_INTEL) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool subgroup_size_control = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_size_control = true;
            }
        }

        if (!subgroup_size_control) {
            return vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

        props2.pNext = &subgroup_size_control_props;
        device.getProperties2(&props2);

        if (subgroup_size_control_props.minSubgroupSize == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            return vk_device_architecture::INTEL_XE2;
        }
    } else if (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        bool cooperative_matrix = false;

        // Detect "pre-turing" based on lack of coopmat support.
        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_matrix = true;
                break;
            }
        }

        if (!cooperative_matrix) {
            return vk_device_architecture::NVIDIA_PRE_TURING;
        }
    }
    return vk_device_architecture::OTHER;
}

enum vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_COUNT,
};

struct vk_device_struct {
    std::recursive_mutex mutex;

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_memory_allocation_size;
    uint64_t suballocation_block_size;
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    vk_queue compute_queue;
    vk_queue transfer_queue;
    bool single_queue;
    uint32_t subgroup_size;
    uint32_t shader_core_count;
    bool uma;
    bool prefer_host_memory;
    bool float_controls_rte_fp16;
    bool subgroup_add;
    bool subgroup_shuffle;

    bool integer_dot_product;

    bool subgroup_size_control;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    bool subgroup_require_full_support;

    bool coopmat_support;
    bool coopmat_acc_f32_support {};
    bool coopmat_acc_f16_support {};
    bool coopmat_bf16_support {};
    bool coopmat_support_16x16x16_f16acc {};
    bool coopmat_support_16x16x16_f32acc {};
    bool coopmat1_fa_support {};
    uint32_t coopmat_m;
    uint32_t coopmat_n;
    uint32_t coopmat_k;

    bool coopmat_int_support;
    uint32_t coopmat_int_m;
    uint32_t coopmat_int_n;
    uint32_t coopmat_int_k;

    bool coopmat2;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_COUNT];
    bool mul_mat_m[GGML_TYPE_COUNT];
    bool mul_mat_s[GGML_TYPE_COUNT];
    bool mul_mat_id_l[GGML_TYPE_COUNT];
    bool mul_mat_id_m[GGML_TYPE_COUNT];
    bool mul_mat_id_s[GGML_TYPE_COUNT];

    // set to true to indicate that some shaders need to be compiled after the dryrun
    bool need_compiles {};

    vk::DescriptorSetLayout dsl;

    vk_matmul_pipeline pipeline_matmul_f32 {};
    vk_matmul_pipeline pipeline_matmul_f32_f16 {};
    vk_matmul_pipeline pipeline_matmul_bf16 {};
    vk_matmul_pipeline2 pipeline_matmul_f16;
    vk_matmul_pipeline2 pipeline_matmul_f16_f32;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_COUNT];

    vk_matmul_pipeline pipeline_matmul_id_f32 {};
    vk_matmul_pipeline pipeline_matmul_id_bf16 {};
    vk_matmul_pipeline2 pipeline_matmul_id_f16;
    vk_matmul_pipeline2 pipeline_matmul_id_f16_f32;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_id[GGML_TYPE_COUNT];

    vk_pipeline pipeline_matmul_split_k_reduce;
    vk_pipeline pipeline_quantize_q8_1;

    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_COUNT];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_ratio];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_get_rows_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_acc_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_add[2][2][2];
    vk_pipeline pipeline_add_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];

    vk_pipeline pipeline_add_id_f32;

    vk_pipeline pipeline_concat_f32, pipeline_concat_f16, pipeline_concat_i32;
    vk_pipeline pipeline_upscale_nearest_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bilinear_ac_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_sqr_f32;
    vk_pipeline pipeline_sin_f32;
    vk_pipeline pipeline_cos_f32;
    vk_pipeline pipeline_clamp_f32;
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_f32, pipeline_repeat_back_f32;
    vk_pipeline pipeline_cpy_f32_f32, pipeline_cpy_f32_f16, pipeline_cpy_f16_f16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16;
    vk_pipeline pipeline_contig_cpy_f32_f32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_f16_f16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_set_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_group_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_back_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_quick[2];
    vk_pipeline pipeline_silu[2];
    vk_pipeline pipeline_relu[2];
    vk_pipeline pipeline_tanh[2];
    vk_pipeline pipeline_sigmoid[2];

    vk_pipeline pipeline_geglu[2];
    vk_pipeline pipeline_reglu[2];
    vk_pipeline pipeline_swiglu[2];
    vk_pipeline pipeline_swiglu_oai[2];
    vk_pipeline pipeline_geglu_erf[2];
    vk_pipeline pipeline_geglu_quick[2];

    vk_pipeline pipeline_leaky_relu_f32;
    vk_pipeline pipeline_silu_back_f32;
    vk_pipeline pipeline_diag_mask_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_back_f32;
    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32;
    vk_pipeline pipeline_sum_rows_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_count_equal_i32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;
    vk_pipeline pipeline_conv_transpose_1d_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_opt_step_adamw_f32;
    vk_pipeline pipeline_conv2d_f32[CONV_SHAPE_COUNT];
    vk_pipeline pipeline_conv2d_f16_f32[CONV_SHAPE_COUNT];
    vk_pipeline pipeline_conv2d_dw_whcn_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32;

    // [2][2][2] is for {f16acc,f32acc}x{large,small_rows}x{unaligned, aligned}
    vk_pipeline pipeline_flash_attn_f32_f16_cm2[GGML_TYPE_COUNT][FA_HEAD_SIZE_COUNT][2][2][2];

    vk_pipeline pipeline_flash_attn_f32_f16_cm1[GGML_TYPE_COUNT][FA_HEAD_SIZE_COUNT][2][2][2];

    vk_pipeline pipeline_flash_attn_f32_f16[GGML_TYPE_COUNT][FA_HEAD_SIZE_COUNT][2][2][2];

    vk_pipeline pipeline_flash_attn_split_k_reduce;

    std::unordered_map<std::string, vk_pipeline_ref> pipelines;

    std::vector<std::tuple<void*, size_t, vk_buffer>> pinned_memory;

    vk::Fence fence;
    vk_buffer sync_staging;

    ggml_backend_buffer_type buffer_type;

    bool disable_fusion;
    bool disable_host_visible_vidmem;

#ifdef GGML_VULKAN_MEMORY_DEBUG
    std::unique_ptr<vk_memory_logger> memory_logger;
#endif

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    int32_t num_queries;

    ~vk_device_struct() {
        VK_LOG_DEBUG("destroy device " << name);

        device.destroyFence(fence);

        ggml_vk_destroy_buffer(sync_staging);

        compute_queue.cmd_pool.destroy(device);
        transfer_queue.cmd_pool.destroy(device);

        for (auto& pipeline : pipelines) {
            if (pipeline.second.expired()) {
                continue;
            }

            vk_pipeline pl = pipeline.second.lock();
            ggml_vk_destroy_pipeline(device, pl);
        }
        pipelines.clear();

        device.destroyDescriptorSetLayout(dsl);

        device.destroy();
    }
};

void vk_command_pool::init(vk_device& device, vk_queue *q_) {
    cmd_buffer_idx = 0;
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT), q->queue_family_index);
    pool = device->device.createCommandPool(command_pool_create_info);
}

void vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

struct vk_buffer_struct {
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceMemory device_memory = VK_NULL_HANDLE;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;

    vk_device device;

    ~vk_buffer_struct() {
        if (size == 0) {
            return;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << buffer << ", " << size << ")");

        device->device.freeMemory(device_memory);
        device->device.destroyBuffer(buffer);
    }
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;

    operator vk::DescriptorBufferInfo() const {
        return { buffer->buffer, offset, size };
    }
};

struct vk_semaphore {
    vk::Semaphore s;
    uint64_t value;
};

struct vk_submission {
    vk::CommandBuffer buffer;
    std::vector<vk_semaphore> wait_semaphores;
    std::vector<vk_semaphore> signal_semaphores;
};

typedef std::vector<vk_submission> vk_sequence;

struct vk_mat_mat_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t k_split;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};
struct vk_mat_vec_push_constants {
    uint32_t ncols; uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
};

struct vk_mat_mat_id_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t padded_N;
};
struct vk_mat_vec_id_push_constants {
    uint32_t ncols; uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t ne11;
};

struct vk_flash_attn_push_constants {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
};
static_assert(sizeof(vk_flash_attn_push_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");

struct vk_op_push_constants {
    uint32_t KX;
    uint32_t KY;
    float param1;
    float param2;
};

struct vk_op_glu_push_constants {
    uint32_t N;
    uint32_t ne00;
    uint32_t ne20;
    uint32_t mode;  // 0: default, 1: swapped, 2: split
    float alpha; // for swiglu_oai
    float limit;
};

struct vk_op_unary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    float param1; float param2;
    uint32_t ne0_012mp; uint32_t ne0_012L;
    uint32_t ne0_01mp;  uint32_t ne0_01L;
    uint32_t ne0_0mp;   uint32_t ne0_0L;
    uint32_t ne1_012mp; uint32_t ne1_012L;
    uint32_t ne1_01mp;  uint32_t ne1_01L;
    uint32_t ne1_0mp;   uint32_t ne1_0L;
};
static_assert(sizeof(vk_op_unary_push_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

static vk_op_unary_push_constants vk_op_unary_push_constants_init(const ggml_tensor * src0, const ggml_tensor * dst, int64_t ne = 0) {
    GGML_ASSERT(ne != 0 || (ggml_nelements(src0) == ggml_nelements(dst)));
    ne = ne != 0 ? ne : ggml_nelements(dst);
    GGML_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

    vk_op_unary_push_constants p{};
    p.ne = (uint32_t)ne;

    size_t src0_tsize = ggml_type_size(src0->type);
    p.ne00 = (uint32_t)src0->ne[0];
    p.ne01 = (uint32_t)src0->ne[1];
    p.ne02 = (uint32_t)src0->ne[2];
    p.ne03 = (uint32_t)src0->ne[3];
    p.nb00 = (uint32_t)(src0->nb[0] / src0_tsize);
    p.nb01 = (uint32_t)(src0->nb[1] / src0_tsize);
    p.nb02 = (uint32_t)(src0->nb[2] / src0_tsize);
    p.nb03 = (uint32_t)(src0->nb[3] / src0_tsize);

    size_t dst_tsize = ggml_type_size(dst->type);
    p.ne10 = (uint32_t)dst->ne[0];
    p.ne11 = (uint32_t)dst->ne[1];
    p.ne12 = (uint32_t)dst->ne[2];
    p.ne13 = (uint32_t)dst->ne[3];
    p.nb10 = (uint32_t)(dst->nb[0] / dst_tsize);
    p.nb11 = (uint32_t)(dst->nb[1] / dst_tsize);
    p.nb12 = (uint32_t)(dst->nb[2] / dst_tsize);
    p.nb13 = (uint32_t)(dst->nb[3] / dst_tsize);

    return p; // fastdiv values and offsets are initialized later in ggml_vk_op
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static void init_fastdiv_values(uint32_t d, uint32_t &mp, uint32_t &L)
{
    // compute L = ceil(log2(d));
    L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) {
        L++;
    }

    mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

template <typename T> void init_pushconst_fastdiv(T &p) {
    GGML_UNUSED(p);
    static_assert(!std::is_const<T>::value, "unexpected type");
}

template <> void init_pushconst_fastdiv(vk_op_unary_push_constants &p) {
    // Compute magic values to divide by these six numbers.
    init_fastdiv_values(p.ne02*p.ne01*p.ne00,  p.ne0_012mp,    p.ne0_012L);
    init_fastdiv_values(p.ne01*p.ne00,         p.ne0_01mp,     p.ne0_01L);
    init_fastdiv_values(p.ne00,                p.ne0_0mp,      p.ne0_0L);
    init_fastdiv_values(p.ne12*p.ne11*p.ne10,  p.ne1_012mp,    p.ne1_012L);
    init_fastdiv_values(p.ne11*p.ne10,         p.ne1_01mp,     p.ne1_01L);
    init_fastdiv_values(p.ne10,                p.ne1_0mp,      p.ne1_0L);
}

struct vk_op_binary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    float param1; float param2; int32_t param3;
};

struct vk_op_add_id_push_constants {
    uint32_t ne0;
    uint32_t ne1;
    uint32_t s01;
    uint32_t s02;
    uint32_t s11;
    uint32_t s21;
};

struct vk_op_diag_mask_push_constants {
    uint32_t ncols;
    uint32_t rows_per_channel;
    int32_t n_past;
};

struct vk_op_rope_push_constants {
    uint32_t ncols;
    uint32_t n_dims;
    float freq_scale;
    uint32_t p_delta_rows;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    uint32_t has_ff;
    uint32_t ne02;
    uint32_t s1;
    uint32_t s2;
    int32_t sections[4];
    uint32_t is_back;
};

struct vk_op_soft_max_push_constants {
    uint32_t KX;
    uint32_t KY;
    uint32_t ne00;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t ne12;
    uint32_t ne13;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint32_t n_head_log2;
    uint32_t nrows_x;
    uint32_t has_sinks;
};

struct vk_op_argsort_push_constants {
    uint32_t ncols;
    uint32_t ncols_pad;
    int32_t order;
};

struct vk_op_im2col_push_constants {
    uint32_t batch_offset; uint32_t offset_delta;
    uint32_t IC;
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t KW; uint32_t KH;
    uint32_t pelements;
    uint32_t CHW;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
    int32_t d0; int32_t d1;
};

struct vk_op_timestep_embedding_push_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

struct vk_op_conv_transpose_1d_push_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t K;
    uint32_t L;
    uint32_t KL;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb11;
    uint32_t nb1;

    int32_t s0;
};

struct vk_op_pool2d_push_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

struct vk_op_rwkv_wkv6_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

struct vk_op_rwkv_wkv7_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

struct vk_op_conv2d_push_constants {
    uint32_t Cout;
    uint32_t Cin;
    uint32_t N;

    uint32_t KW;
    uint32_t KH;
    uint32_t W;
    uint32_t H;
    uint32_t OW;
    uint32_t OH;

    uint32_t s0;
    uint32_t s1;
    uint32_t p0;
    uint32_t p1;
    uint32_t d0;
    uint32_t d1;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;

    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;

    uint32_t nb1;
    uint32_t nb2;
    uint32_t nb3;

    // init_fastdiv_values constants for dividing by KW, KW*KH, OW, OW*OH
    uint32_t KWmp;   uint32_t KWL;
    uint32_t KWKHmp; uint32_t KWKHL;
    uint32_t OWmp;   uint32_t OWL;
    uint32_t OWOHmp; uint32_t OWOHL;
};

template <> void init_pushconst_fastdiv(vk_op_conv2d_push_constants &p) {
    // Compute magic values to divide by KW, KW*KH, OW, OW*OH
    init_fastdiv_values(p.KW,       p.KWmp,    p.KWL);
    init_fastdiv_values(p.KW*p.KH,  p.KWKHmp,  p.KWKHL);
    init_fastdiv_values(p.OW,       p.OWmp,    p.OWL);
    init_fastdiv_values(p.OW*p.OH,  p.OWOHmp,  p.OWOHL);
}

struct vk_op_conv2d_dw_push_constants {
    uint32_t ne;
    uint32_t batches;
    uint32_t channels;
    uint32_t dst_w;
    uint32_t dst_h;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t knl_w;
    uint32_t knl_h;
    int32_t stride_x;
    int32_t stride_y;
    int32_t pad_x;
    int32_t pad_y;
    int32_t dilation_x;
    int32_t dilation_y;
};

struct vk_op_upscale_push_constants {
    uint32_t ne; uint32_t a_offset; uint32_t d_offset;
    uint32_t ne00; uint32_t ne01;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    float sf0; float sf1; float sf2; float sf3;
};

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct vk_context_struct {
    vk_submission * s;
    std::vector<vk_sequence> seqs;

    int exit_tensor_idx;

    std::vector<vk_staging_memcpy> in_memcpys;
    std::vector<vk_staging_memcpy> out_memcpys;

    vk_command_pool * p {};
};
typedef std::shared_ptr<vk_context_struct> vk_context;
typedef std::weak_ptr<vk_context_struct> vk_context_ref;

struct ggml_vk_garbage_collector {
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
    std::vector<vk::Event> events;
    std::vector<vk_buffer> temp_buffers;
    std::vector<vk_context> contexts;
};

#if defined(GGML_VULKAN_MEMORY_DEBUG) || defined(GGML_VULKAN_DEBUG)
#define VK_LOG_MEMORY(msg) std::cerr << "ggml_vulkan memory: " << msg << std::endl

static std::string format_size(size_t size) {
    const size_t kib = 1024;
    const size_t mib = kib * 1024;
    const size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (size >= gib) {
        oss << static_cast<double>(size) / gib << " GiB";
    } else if (size >= mib) {
        oss << static_cast<double>(size) / mib << " MiB";
    } else if (size >= kib) {
        oss << static_cast<double>(size) / kib << " KiB";
    } else {
        oss << size << " B";
    }

    return oss.str();
}

static std::mutex log_mutex;

class vk_memory_logger {
public:
    vk_memory_logger(): total_device(0), total_host(0) {}
    void log_allocation(vk_buffer_ref buf_ref, size_t size);
    void log_deallocation(vk_buffer_ref buf_ref);

private:
    std::map<vk::Buffer, size_t> allocations; // Track allocations
    size_t total_device;
    size_t total_host;
};
#else
#define VK_LOG_MEMORY(msg) ((void) 0)
#endif // GGML_VULKAN_MEMORY_DEBUG

class vk_perf_logger {
  public:
    void print_timings() {
        if (timings.empty()) {
            return;
        }
        uint64_t total_all_op_times = 0;
        std::cerr << "----------------\nVulkan Timings:" << std::endl;
        for (const auto & t : timings) {
            uint64_t total_op_times = 0;
            for (const auto & time : t.second) {
                total_op_times += time;
            }
            std::cerr << t.first << ": " << t.second.size() << " x " << (total_op_times / t.second.size() / 1000.0)
                      << " us";

            // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
            auto it = flops.find(t.first);
            if (it != flops.end() && (it->second).size() == t.second.size()) {
                uint64_t total_op_flops = 0;
                for (const auto & elem : it->second) {
                    total_op_flops += elem;
                }
                std::cerr << " ("
                          << (double(total_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                                 (double(total_op_times) / (1000.0 * 1000.0 * 1000.0))
                          << " GFLOPS/s)";
            }

            total_all_op_times += total_op_times;

            std::cerr << std::endl;
        }

        if (timings.size() > 0) {
            std::cerr << "Total time: " << total_all_op_times / 1000.0 << " us." << std::endl;
        }

        timings.clear();
        flops.clear();
    }

    void log_timing(const ggml_tensor * node, uint64_t time) {
        if (node->op == GGML_OP_UNARY) {
            timings[ggml_unary_op_name(ggml_get_unary_op(node))].push_back(time);
            return;
        }
        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
            const uint64_t m    = node->src[0]->ne[1];
            const uint64_t n    = node->src[1]->ne[1];
            const uint64_t k    = node->src[1]->ne[0];
            std::string    name = ggml_op_name(node->op);
            if (n == 1) {
                name += "_VEC m=" + std::to_string(m) + " k=" + std::to_string(k);
            } else {
                name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
            }
            timings[name].push_back(time);
            flops[name].push_back(m * n * (k + (k - 1)));
            return;
        }
        if (node->op == GGML_OP_CONV_2D) {
            std::string   name    = ggml_op_name(node->op);
            ggml_tensor * knl     = node->src[0];
            uint64_t      OW      = node->ne[0];
            uint64_t      OH      = node->ne[1];
            uint64_t      N       = node->ne[3];
            uint64_t      Cout    = node->ne[2];
            uint64_t      KW      = knl->ne[0];
            uint64_t      KH      = knl->ne[1];
            uint64_t      Cin     = knl->ne[2];
            // KxCRS @ CRSxNPQ = KxNPQ -> M=K, K=CRS, N=NPQ
            uint64_t      size_M  = Cout;
            uint64_t      size_K  = Cin * KW * KH;
            uint64_t      size_N  = N * OW * OH;
            uint64_t      n_flops = size_M * size_N * (size_K + (size_K - 1));
            name += " M=Cout=" + std::to_string(size_M) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
                    ", N=N*OW*OH=" + std::to_string(size_N);
            flops[name].push_back(n_flops);
            timings[name].push_back(time);
            return;
        }
        timings[ggml_op_name(node->op)].push_back(time);
    }
  private:
    std::map<std::string, std::vector<uint64_t>> timings;
    std::map<std::string, std::vector<uint64_t>> flops;
};

struct ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k;
    vk_buffer prealloc_x, prealloc_y, prealloc_split_k;
    vk::Fence fence, almost_ready_fence;
    bool almost_ready_fence_pending {};

    vk_buffer buffer_pool[MAX_VK_BUFFERS];

    vk_context_ref compute_ctx;
    vk_context_ref transfer_ctx;

    std::vector<vk_context_ref> tensor_ctxs;

    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::DescriptorSet> descriptor_sets;
    uint32_t descriptor_set_idx {};
    uint32_t pipeline_descriptor_set_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    int num_additional_fused_ops {};
};

static void * const vk_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

static uint64_t vk_tensor_offset(const ggml_tensor * tensor) {
    if (tensor->view_src) {
        return (uint8_t *) tensor->view_src->data - (uint8_t *) vk_ptr_base;
    }
    return (uint8_t *) tensor->data - (uint8_t *) vk_ptr_base;
}

struct ggml_backend_vk_buffer_context {
    vk_device_ref device;
    vk_buffer dev_buffer;
    std::string name;

    ggml_backend_vk_buffer_context(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name) :
        device(device),
        dev_buffer(dev_buffer),
        name(name) {
    }

    ~ggml_backend_vk_buffer_context() {
        ggml_vk_destroy_buffer(dev_buffer);
    }
};

#ifdef GGML_VULKAN_MEMORY_DEBUG
void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    const std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}

void vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
    if (buf_ref.expired() || buf_ref.lock()->size == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->buffer);
    total_device -= device ? it->second : 0;
    total_host -= device ? 0 : it->second;
    if (it != allocations.end()) {
        VK_LOG_MEMORY(buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
        allocations.erase(it);
    } else {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
    }
}
#endif // GGML_VULKAN_MEMORY_DEBUG

struct vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT = {};
    PFN_vkQueueBeginDebugUtilsLabelEXT pfn_vkQueueBeginDebugUtilsLabelEXT = {};
    PFN_vkQueueEndDebugUtilsLabelEXT   pfn_vkQueueEndDebugUtilsLabelEXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vkCmdEndDebugUtilsLabelEXT pfn_vkCmdEndDebugUtilsLabelEXT = {};
    PFN_vkCmdInsertDebugUtilsLabelEXT  pfn_vkCmdInsertDebugUtilsLabelEXT  = {};

    std::vector<size_t> device_indices;
    vk_device devices[GGML_VK_MAX_DEVICES];
};

static bool vk_instance_initialized = false;
static vk_instance_t vk_instance;

static bool vk_perf_logger_enabled = false;

#ifdef GGML_VULKAN_CHECK_RESULTS
static size_t vk_skip_checks;
static size_t vk_output_tensor;

static void ggml_vk_print_tensor(const ggml_tensor * tensor, const char * name);
static void ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int tensor_idx);
static void ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int tensor_idx);
#endif

typedef void (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static void ggml_backend_vk_free(ggml_backend_t backend);

// Wait for ctx->fence to be signaled.
static void ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    if (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, true, UINT64_MAX), "almost_ready_fence");
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = false;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::Result result;
    while ((result = ctx->device->device.getFenceStatus(ctx->fence)) != vk::Result::eSuccess) {
        if (result != vk::Result::eNotReady) {
            fprintf(stderr, "ggml_vulkan: error %s at %s:%d\n", to_string(result).c_str(), __FILE__, __LINE__);
            exit(1);
        }
        for (uint32_t i = 0; i < 100; ++i) {
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
            YIELD();
        }
    }
    ctx->device->device.resetFences({ ctx->fence });
}

// variables to track number of compiles in progress
static uint32_t compile_count = 0;
static std::mutex compile_count_mutex;
static std::condition_variable compile_count_cond;

static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));
    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(uint32_t);
        specialization_entries[i].size = sizeof(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    if (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
    pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
    if (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    if (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

    try {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
    pipeline->compiled = true;

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    {
        std::lock_guard<std::recursive_mutex> guard(device->mutex);
        device->pipelines.insert({ pipeline->name, pipeline });
    }

    {
        std::lock_guard<std::mutex> guard(compile_count_mutex);
        assert(compile_count > 0);
        compile_count--;
    }
    compile_count_cond.notify_all();
}

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

static void ggml_pipeline_request_descriptor_sets(ggml_backend_vk_context *ctx, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
    ctx->pipeline_descriptor_set_requirements += n;
    if (!pipeline->compiled) {
        pipeline->needed = true;
        ctx->device->need_compiles = true;
    }
}

static void ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
        // Enough descriptors are available
        return;
    }

    vk_device& device = ctx->device;

    uint32_t to_alloc = ctx->pipeline_descriptor_set_requirements - ctx->descriptor_sets.size();
    uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_SIZE;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    while (to_alloc > 0) {
        const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
        to_alloc -= alloc_count;
        pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

        if (pool_idx >= ctx->descriptor_pools.size()) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = device->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
}

static vk::CommandBuffer ggml_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");

    if (p.cmd_buffers.size() > p.cmd_buffer_idx) {
        // Reuse command buffer
        return p.cmd_buffers[p.cmd_buffer_idx++];
    }

    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        p.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
    auto buf = cmd_buffers.front();

    p.cmd_buffers.push_back(buf);
    p.cmd_buffer_idx++;

    return buf;
}

static void ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    if (ctx->seqs.empty()) {
        if (fence) {
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->p->q->queue.submit({}, fence);
        }
        return;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    for (const auto& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    for (const auto& sequence : ctx->seqs) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    std::lock_guard<std::mutex> guard(queue_mutex);
    ctx->p->q->queue.submit(submit_infos, fence);

    ctx->seqs.clear();
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to ignoring min_num_queries
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    if (compute_index >= 0) {
        return compute_index;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static void ggml_vk_create_queue(vk_device& device, vk_queue& q, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags, bool transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    q.queue_family_index = queue_family_index;
    q.transfer_only = transfer_only;

    q.cmd_pool.init(device, &q);

    q.queue = device->device.getQueue(queue_family_index, queue_index);

    q.stage_flags = stage_flags;
}

static vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << result << ")");
    ctx->gc.contexts.emplace_back(result);
    result->p = &p;
    return result;
}

static vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << result << ")");
    result->p = &p;
    return result;
}

static vk_semaphore * ggml_vk_create_binary_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eBinary, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

static vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    if (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    return &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

static vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    if (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    return ctx->gc.events[ctx->event_idx++];
}

static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    p.cmd_buffer_idx = 0;
}

static void ggml_vk_queue_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    static constexpr uint32_t cleanup_frequency = 10;

    if (device->compute_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_queue.cmd_pool);
    }
    if (device->transfer_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_queue.cmd_pool);
    }
}


static uint32_t find_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            return static_cast<int32_t>(i);
        }
    }
    return UINT32_MAX;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0)) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags) << ", " << to_string(fallback_flags) << ")");
    if (size > device->max_memory_allocation_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device memory allocation limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    uint32_t memory_type_index = UINT32_MAX;

    memory_type_index = find_properties(&mem_props, &mem_req, req_flags);
    buf->memory_property_flags = req_flags;

    if (memory_type_index == UINT32_MAX && fallback_flags) {
        memory_type_index = find_properties(&mem_props, &mem_req, fallback_flags);
        buf->memory_property_flags = fallback_flags;
    }

    if (memory_type_index == UINT32_MAX) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    try {
        buf->device_memory = device->device.allocateMemory({ mem_req.size, memory_type_index });
    } catch (const vk::SystemError& e) {
        if (buf->memory_property_flags != fallback_flags) {
            // Try again with fallback flags
            memory_type_index = find_properties(&mem_props, &mem_req, fallback_flags);
            buf->memory_property_flags = fallback_flags;

            try {
                buf->device_memory = device->device.allocateMemory({ mem_req.size, memory_type_index });
            }
            catch (const vk::SystemError& e) {
                device->device.destroyBuffer(buf->buffer);
                throw e;
            }
        } else {
            // Out of Host/Device memory, clean up buffer
            device->device.destroyBuffer(buf->buffer);
            throw e;
        }
    }
    buf->ptr = nullptr;

    if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

#ifdef GGML_VULKAN_MEMORY_DEBUG
    device->memory_logger->log_allocation(buf, size);
#endif

    return buf;
}

static vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0)) {
    try {
        return ggml_vk_create_buffer(device, size, req_flags, fallback_flags);
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, vk::MemoryPropertyFlagBits::eDeviceLocal);
        } else if (device->uma) {
            // Fall back to host memory type
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        } else if (device->disable_host_visible_vidmem) {
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::MemoryPropertyFlagBits::eDeviceLocal);
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

#ifdef GGML_VULKAN_MEMORY_DEBUG
    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }
#endif

    buf.reset();
}

static vk_subbuffer ggml_vk_subbuffer(vk_buffer& buf) {
    return { buf, 0, VK_WHOLE_SIZE };
}

static void ggml_vk_sync_buffers(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    const bool transfer_queue = ctx->p->q->transfer_only;

    ctx->s->buffer.pipelineBarrier(
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        { {
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) },
          { !transfer_queue ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite) }
        } },
        {},
        {}
    );
}

static void ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    if (events.empty()) {
        return;
    }

    ctx->s->buffer.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

enum FaCodePath {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

static FaHeadSizes fa_get_head_sizes(uint32_t hsk, uint32_t hsv) {
    if (hsk != 192 && hsk != 576 && hsk != hsv) {
        return FA_HEAD_SIZE_UNSUPPORTED;
    }
    switch (hsk) {
    case 64: return FA_HEAD_SIZE_64;
    case 80: return FA_HEAD_SIZE_80;
    case 96: return FA_HEAD_SIZE_96;
    case 112: return FA_HEAD_SIZE_112;
    case 128: return FA_HEAD_SIZE_128;
    case 192:
        if (hsv == 192) {
            return FA_HEAD_SIZE_192;
        } else if (hsv == 128) {
            return FA_HEAD_SIZE_192_128;
        } else {
            return FA_HEAD_SIZE_UNSUPPORTED;
        }
    case 256: return FA_HEAD_SIZE_256;
    case 576:
        if (hsv == 512) {
            return FA_HEAD_SIZE_576_512;
        } else {
            return FA_HEAD_SIZE_UNSUPPORTED;
        }
    default: return FA_HEAD_SIZE_UNSUPPORTED;
    }
}

// number of rows/cols for flash attention shader
static constexpr uint32_t flash_attention_num_small_rows = 32;
static constexpr uint32_t scalar_flash_attention_num_small_rows = 1;

static uint32_t get_fa_scalar_num_large_rows(uint32_t hsv) {
    if (hsv >= 512) {
        return 2;
    } else {
        return 8;
    }
}

// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
// 128 threads split into four subgroups, each subgroup does 1/4
// of the Bc dimension.
static constexpr uint32_t coopmat1_flash_attention_num_large_rows = 16;
static constexpr uint32_t scalar_flash_attention_Bc = 64;
static constexpr uint32_t scalar_flash_attention_workgroup_size = 128;

static uint32_t get_fa_num_small_rows(FaCodePath path) {
    if (path == FA_COOPMAT2) {
        return flash_attention_num_small_rows;
    } else {
        return scalar_flash_attention_num_small_rows;
    }
}

static std::array<uint32_t, 2> fa_rows_cols(FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, ggml_type type, bool small_rows) {
    GGML_UNUSED(clamp);
    GGML_UNUSED(hsv);

    if (path == FA_SCALAR) {
        if (small_rows) {
            return {scalar_flash_attention_num_small_rows, 64};
        } else {
            return {get_fa_scalar_num_large_rows(hsv), 32};
        }
    }

    if (path == FA_COOPMAT1) {
        if (small_rows) {
            return {scalar_flash_attention_num_small_rows, scalar_flash_attention_Bc};
        } else {
            return {coopmat1_flash_attention_num_large_rows, scalar_flash_attention_Bc};
        }
    }

    // small rows, large cols
    if (small_rows) {
        return {get_fa_num_small_rows(FA_COOPMAT2), 32};
    }

    // small cols to reduce register count
    if (ggml_is_quantized(type) || hsk >= 256) {
        if (hsk >= 512) {
            return {32, 32};
        } else {
            return {64, 32};
        }
    }
    return {64, 64};
}

static bool ggml_vk_matmul_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    uint32_t lut_size = 0;
    switch (src0_type) {
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
        lut_size = 2*2048;
        break;
    case GGML_TYPE_IQ2_XXS:
        lut_size = 8*256;
        break;
    case GGML_TYPE_IQ2_XS:
        lut_size = 8*512;
        break;
    case GGML_TYPE_IQ2_S:
        lut_size = 8*1024;
        break;
    case GGML_TYPE_IQ3_XXS:
        lut_size = 4*256;
        break;
    case GGML_TYPE_IQ3_S:
        lut_size = 4*512;
        break;
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_MXFP4:
        lut_size = 4*16;
        break;
    default:
        break;
    }

    // Needs to be kept up to date on shader changes
    const uint32_t bank_conflict_offset = device->coopmat_support ? 8 : 1;
    const uint32_t type_size = device->fp16 ? sizeof(ggml_fp16_t) : sizeof(float);
    const uint32_t warps = warptile[0] / warptile[10];

    const uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (4096 * sizeof(uint32_t) + 4/*_ne1*/) : 0;
    const uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * sizeof(float) : 0;

    const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    return supported;
}

struct GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_size = 0;
};

// Pipeline configuration for RDNA1 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

// Pipeline configuration for RDNA2 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;

// Define configurations for different GPUs.
static std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
};

static uint32_t get_subgroup_size(const std::string &pipeline_name, const vk_device_architecture &arch) {
    for (const auto &config : gpu_pipeline_configs) {
        if (config.arch == arch) {
            auto pipIt = config.pipelines.find(pipeline_name);
            if (pipIt != config.pipelines.end()) {
                return pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
            for (const auto &entry : sorted_pipelines) {
                if (pipeline_name.find(entry.first) != std::string::npos) {
                    return entry.second;
                }
            }
            return config.default_subgroup_size;
        }
    }
    return 0; // If no matching configuration is found
}

static void ggml_vk_load_shaders(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    const uint32_t subgroup_size_8 = std::max(device->subgroup_size, 8u);
    const uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
    const uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

    // mulmat
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;
    if (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        l_warptile_mmqid = { 256, 128, 128, 16, 0 };
        m_warptile_mmqid = { 256, 128, 64, 16, 0 };
        s_warptile_mmqid = { 256, 128, 64, 16, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } else {
        // Matrix cores require different warp group sizes
        const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        const uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        l_warptile = { 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile = { 128,  64,  64, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq = { 128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile_mmq = { 128,  64,  64, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile_mmq = { subgroup_size_32, 32, 32, 32, 32, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq_int = { 128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile_mmq_int = { 128,  64,  64, 32, subgroup_size_8,     32, 2, 2, 2, 1, subgroup_size_8 };
        s_warptile_mmq_int = { subgroup_size_32, 32, 32, 32, 32,       32, 2, 2, 1, 1, subgroup_size_8 };

        // chip specific tuning
        if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
                throw std::runtime_error("Shared memory size too small for matrix multiplication.");
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
                device->mul_mat_m[i] = false;
                device->mul_mat_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) {
                device->mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, true, t)) {
                device->mul_mat_id_s[i] = false;
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, true, t)) {
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, true, t)) {
                device->mul_mat_id_l[i] = false;
            }
        }
    }

    if (!device->pipeline_matmul_f32) {
        device->pipeline_matmul_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_f32_f16) {
        device->pipeline_matmul_f32_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_f32) {
        device->pipeline_matmul_id_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_bf16) {
        device->pipeline_matmul_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_bf16) {
        device->pipeline_matmul_id_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }

    std::vector<std::future<void>> compiles;
    auto const &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& pipeline, const std::string &name, size_t spv_size, const void* spv_data, const std::string &entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {

        if (!require_full_subgroups && required_subgroup_size == 0) {
            required_subgroup_size = get_subgroup_size(name, device->architecture);
        }

        if (!pipeline) {
            pipeline = std::make_shared<vk_pipeline_struct>();
            pipeline->name = name;
            pipeline->parameter_count = parameter_count;
            pipeline->push_constant_size = push_constant_size;
            pipeline->wg_denoms = wg_denoms;
            pipeline->align = align;
        }

        if (!pipeline->needed || pipeline->compiled) {
            return;
        }
        {
            // wait until fewer than N compiles are in progress
            uint32_t N = std::max(1u, std::thread::hardware_concurrency());
            std::unique_lock<std::mutex> guard(compile_count_mutex);
            while (compile_count >= N) {
                compile_count_cond.wait(guard);
            }
            compile_count++;
        }

        compiles.push_back(std::async(ggml_vk_create_pipeline_func, std::ref(device), std::ref(pipeline), spv_size, spv_data, entrypoint,
                                      parameter_count, wg_denoms, specialization_constants, disable_robustness, require_full_subgroups, required_subgroup_size));
    };

    auto const &fa_wg_denoms = [&](FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, ggml_type type, bool small_rows) -> std::array<uint32_t, 3> {
        return {fa_rows_cols(path, hsk, hsv, clamp, type, small_rows)[0], 1, 1};
    };

    auto const &fa_spec_constants = [&](FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, ggml_type type, bool small_rows) -> std::vector<uint32_t> {
        // For large number of rows, 128 invocations seems to work best.
        // For small number of rows (e.g. N==1), 256 works better. But matrix granularity for 256 is 32, so we
        // can't use 256 for D==80.
        // For scalar, use 128 (arbitrary)
        // The same D_split value is used for both HSK and HSV, so just base it on the union of the LSBs.
        const uint32_t D = (hsk|hsv);
        uint32_t wg_size = (path == FA_SCALAR || path == FA_COOPMAT1)
                            ? scalar_flash_attention_workgroup_size
                            : ((small_rows && (D % 32) == 0) ? 256 : 128);
        auto rows_cols = fa_rows_cols(path, hsk, hsv, clamp, type, small_rows);

        // D_split can't be larger than a subgroup because we use subgroupShuffle to reduce it.
        // D_split can't be larger than the LSB of D divided by 4 due to vectorization in the shader.
        const uint32_t D_lsb = D ^ (D & (D-1));
        uint32_t D_split = std::min(std::min(device->subgroup_size, 8u), D_lsb / 4);

        // mask dim1 is padded to 64, we rely on this to avoid clamping mask loads
        GGML_ASSERT((GGML_KQ_MASK_PAD % rows_cols[0]) == 0);
        return {wg_size, rows_cols[0], rows_cols[1], hsk, hsv, clamp, D_split};
    };

#define CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, HSK, HSV, HEAD_SIZES) \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][0][0][0], "flash_attn_f32_f16_" #HEAD_SIZES "_f16acc"         #NAMELC #SUFFIX,           flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,false), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,false), 1,                                            true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][0][0][1], "flash_attn_f32_f16_" #HEAD_SIZES "_aligned_f16acc" #NAMELC #SUFFIX,           flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,false), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,false), fa_rows_cols(FAPATH,HSK,HSV,0,TYPE,false)[1], true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][1][0][0], "flash_attn_f32_f16_" #HEAD_SIZES "_f32acc"         #NAMELC #SUFFIX,           flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _len,         flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _data,         "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,false), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,false), 1,                                            true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][1][0][1], "flash_attn_f32_f16_" #HEAD_SIZES "_aligned_f32acc" #NAMELC #SUFFIX,           flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _len,         flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _data,         "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,false), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,false), fa_rows_cols(FAPATH,HSK,HSV,0,TYPE,false)[1], true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][0][1][0], "flash_attn_f32_f16_" #HEAD_SIZES "_f16acc_smallrows"         #NAMELC #SUFFIX, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,true), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,true),   1,                                            true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][0][1][1], "flash_attn_f32_f16_" #HEAD_SIZES "_aligned_f16acc_smallrows" #NAMELC #SUFFIX, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,true), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,true),   fa_rows_cols(FAPATH,HSK,HSV,0,TYPE,true)[1],  true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][1][1][0], "flash_attn_f32_f16_" #HEAD_SIZES "_f32acc_smallrows"         #NAMELC #SUFFIX, flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _len,         flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _data,         "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,true), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,true),   1,                                            true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
        ggml_vk_create_pipeline(device, device->pipeline_flash_attn_f32_f16 ## SUFFIX[TYPE][FA_HEAD_SIZE_##HEAD_SIZES][1][1][1], "flash_attn_f32_f16_" #HEAD_SIZES "_aligned_f32acc_smallrows" #NAMELC #SUFFIX, flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _len,         flash_attn_f32_f16_ ## NAMELC ##     SUFFIX ## _data,         "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,true), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,true),   fa_rows_cols(FAPATH,HSK,HSV,0,TYPE,true)[1],  true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \

#define CREATE_FA(TYPE, NAMELC, FAPATH, SUFFIX) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 64, 64, 64) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 80, 80, 80) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 96, 96, 96) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 112, 112, 112) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 128, 128, 128) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 192, 192, 192) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 192, 128, 192_128) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 256, 256, 256) \
        CREATE_FA2(TYPE, NAMELC, FAPATH, SUFFIX, 576, 512, 576_512)

    CREATE_FA(GGML_TYPE_F16, f16, FA_SCALAR, )
    CREATE_FA(GGML_TYPE_Q4_0, q4_0, FA_SCALAR, )
    CREATE_FA(GGML_TYPE_Q8_0, q8_0, FA_SCALAR, )
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat1_fa_support) {
        CREATE_FA(GGML_TYPE_F16, f16, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q4_0, q4_0, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q8_0, q8_0, FA_COOPMAT1, _cm1)
    }
#endif
#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {
        CREATE_FA(GGML_TYPE_F16, f16, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_Q4_0, q4_0, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_Q4_1, q4_1, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_Q5_0, q5_0, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_Q5_1, q5_1, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_Q8_0, q8_0, FA_COOPMAT2, _cm2)
        CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_COOPMAT2, _cm2)
    }
#endif
#undef CREATE_FA2
#undef CREATE_FA

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {

        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        CREATE_MM(PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \
        CREATE_MM(PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \

        CREATE_MM2(pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3)
        }
#endif
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_0], matmul_q4_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_1], matmul_q4_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_0], matmul_q5_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_1], matmul_q5_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q8_0], matmul_q8_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q2_K], matmul_q2_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q3_K], matmul_q3_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_K], matmul_q4_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_K], matmul_q5_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q6_K], matmul_q6_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_S],   matmul_iq1_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_M],   matmul_iq1_m_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_S],   matmul_iq2_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_S],   matmul_iq3_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_MXFP4],   matmul_mxfp4_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)

        CREATE_MM2(pipeline_matmul_id_f16, matmul_id_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, 4)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 4)
        }
#endif
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f16acc, matmul_id_q4_0_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f16acc, matmul_id_q4_1_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f16acc, matmul_id_q5_0_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f16acc, matmul_id_q5_1_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f16acc, matmul_id_q8_0_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f16acc, matmul_id_q2_k_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f16acc, matmul_id_q3_k_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f16acc, matmul_id_q4_k_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f16acc, matmul_id_q5_k_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f16acc, matmul_id_q6_k_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f16acc,   matmul_id_iq1_s_f16,   , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f16acc,   matmul_id_iq1_m_f16,   , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f16acc, matmul_id_iq2_xxs_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f16acc,  matmul_id_iq2_xs_f16,  , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f16acc,   matmul_id_iq2_s_f16,   , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f16acc, matmul_id_iq3_xxs_f16, , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f16acc,   matmul_id_iq3_s_f16,   , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f16acc,  matmul_id_iq4_xs_f16,  , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f16acc,  matmul_id_iq4_nl_f16,  , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
        CREATE_MM(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f16acc,   matmul_id_mxfp4_f16,   , mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 4)
#undef CREATE_MM
#undef CREATE_MM2
    } else
#endif  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat_support) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->coopmat_acc_f16_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \
        if (device->coopmat_acc_f32_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, )
        }
#endif

        if (device->coopmat_acc_f16_support) {
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        } else {
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        }

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        }
#endif

        if (device->coopmat_acc_f16_support) {
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f16acc, matmul_id_q4_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f16acc, matmul_id_q4_1_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f16acc, matmul_id_q5_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f16acc, matmul_id_q5_1_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f16acc, matmul_id_q8_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);

            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f16acc, matmul_id_q2_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f16acc, matmul_id_q3_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f16acc, matmul_id_q4_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f16acc, matmul_id_q5_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f16acc, matmul_id_q6_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f16acc,   matmul_id_iq1_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f16acc,   matmul_id_iq1_m_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f16acc, matmul_id_iq2_xxs_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f16acc,  matmul_id_iq2_xs_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f16acc,   matmul_id_iq2_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f16acc, matmul_id_iq3_xxs_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f16acc,   matmul_id_iq3_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f16acc,  matmul_id_iq4_xs_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f16acc,  matmul_id_iq4_nl_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f16acc,   matmul_id_mxfp4_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        } else {
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f16acc, matmul_id_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f16acc, matmul_id_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f16acc, matmul_id_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f16acc, matmul_id_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f16acc, matmul_id_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);

            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f16acc, matmul_id_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f16acc, matmul_id_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f16acc, matmul_id_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f16acc, matmul_id_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f16acc, matmul_id_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f16acc,   matmul_id_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f16acc,   matmul_id_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f16acc, matmul_id_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f16acc,  matmul_id_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f16acc,   matmul_id_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f16acc, matmul_id_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f16acc,   matmul_id_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f16acc,  matmul_id_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f16acc,  matmul_id_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f16acc,   matmul_id_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        }
#undef CREATE_MM2
#undef CREATE_MM
    } else
#endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->fp16) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->l, #NAMELC "_f16acc_l", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->l, #NAMELC        "_l", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        } \
        if (device->mul_mat ## ID ## _m[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->m, #NAMELC "_f16acc_m", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->m, #NAMELC        "_m", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        } \
        if (device->mul_mat ## ID ## _s[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->s, #NAMELC "_f16acc_s", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->s, #NAMELC        "_s", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        } \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );

        CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

        CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0], matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1], matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0], matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1], matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0], matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
        }
#endif

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f16acc, matmul_id_q4_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f16acc, matmul_id_q4_1_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f16acc, matmul_id_q5_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f16acc, matmul_id_q5_1_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f16acc, matmul_id_q8_0_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f16acc, matmul_id_q2_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f16acc, matmul_id_q3_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f16acc, matmul_id_q4_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f16acc, matmul_id_q5_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f16acc, matmul_id_q6_k_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f16acc,   matmul_id_iq1_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f16acc,   matmul_id_iq1_m_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f16acc, matmul_id_iq2_xxs_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f16acc,  matmul_id_iq2_xs_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f16acc,   matmul_id_iq2_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f16acc, matmul_id_iq3_xxs_f32, _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f16acc,   matmul_id_iq3_s_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f16acc,  matmul_id_iq4_xs_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f16acc,  matmul_id_iq4_nl_f32,  _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f16acc,   matmul_id_mxfp4_f32,   _f16acc, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
#undef CREATE_MM2
#undef CREATE_MMQ
#undef CREATE_MM
    } else {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC "_l", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC "_m", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC "_s", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16.f32acc, matmul_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16_f32.f32acc, matmul_f16_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );

        CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

        CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
        }
#endif

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16.f32acc, matmul_id_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16_f32.f32acc, matmul_id_f16_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f32acc, matmul_id_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f32acc, matmul_id_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f32acc, matmul_id_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f32acc, matmul_id_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f32acc, matmul_id_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);

        CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f32acc, matmul_id_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f32acc, matmul_id_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f32acc, matmul_id_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f32acc, matmul_id_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f32acc, matmul_id_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f32acc,   matmul_id_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f32acc,   matmul_id_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f32acc, matmul_id_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f32acc,  matmul_id_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f32acc,   matmul_id_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f32acc, matmul_id_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f32acc,   matmul_id_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f32acc,  matmul_id_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f32acc,  matmul_id_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
        CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f32acc,   matmul_id_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, 4, _id);
    }
    // reusing CREATE_MM from the fp32 path
    if ((device->coopmat2 || device->coopmat_support)
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#endif
        ) {
        // use scalar tile sizes
        l_warptile = { 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile = { 128,  64,  64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8 };
        s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, 2, 2, 1, subgroup_size_8 };

        l_wg_denoms = {128, 128, 1 };
        m_wg_denoms = { 64,  64, 1 };
        s_wg_denoms = { 32,  32, 1 };

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 4, _id);
    }
#undef CREATE_MM

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    if (device->vendor_id == VK_VENDOR_ID_AMD) {
        if (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_INTEL)
        rm_stdq = 2;
    uint32_t rm_iq = 2 * rm_kq;

    for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32_"+std::to_string(i+1),  mul_mat_vec_f32_f32_f32_len,  mul_mat_vec_f32_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32_"+std::to_string(i+1),  mul_mat_vec_f16_f32_f32_len,  mul_mat_vec_f16_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32_"+std::to_string(i+1), mul_mat_vec_bf16_f32_f32_len, mul_mat_vec_bf16_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32_"+std::to_string(i+1), mul_mat_vec_q4_0_f32_f32_len, mul_mat_vec_q4_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32_"+std::to_string(i+1), mul_mat_vec_q4_1_f32_f32_len, mul_mat_vec_q4_1_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32_"+std::to_string(i+1), mul_mat_vec_q5_0_f32_f32_len, mul_mat_vec_q5_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32_"+std::to_string(i+1), mul_mat_vec_q5_1_f32_f32_len, mul_mat_vec_q5_1_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32_"+std::to_string(i+1), mul_mat_vec_q8_0_f32_f32_len, mul_mat_vec_q8_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {device->subgroup_size, 1*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32_"+std::to_string(i+1), mul_mat_vec_q2_k_f32_f32_len, mul_mat_vec_q2_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32_"+std::to_string(i+1), mul_mat_vec_q3_k_f32_f32_len, mul_mat_vec_q3_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32_"+std::to_string(i+1), mul_mat_vec_q4_k_f32_f32_len, mul_mat_vec_q4_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32_"+std::to_string(i+1), mul_mat_vec_q5_k_f32_f32_len, mul_mat_vec_q5_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32_"+std::to_string(i+1), mul_mat_vec_q6_k_f32_f32_len, mul_mat_vec_q6_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32_"+std::to_string(i+1),   mul_mat_vec_iq1_s_f32_f32_len,   mul_mat_vec_iq1_s_f32_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32_"+std::to_string(i+1),   mul_mat_vec_iq1_m_f32_f32_len,   mul_mat_vec_iq1_m_f32_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32_"+std::to_string(i+1), mul_mat_vec_iq2_xxs_f32_f32_len, mul_mat_vec_iq2_xxs_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32_"+std::to_string(i+1),  mul_mat_vec_iq2_xs_f32_f32_len,  mul_mat_vec_iq2_xs_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32_"+std::to_string(i+1),   mul_mat_vec_iq2_s_f32_f32_len,   mul_mat_vec_iq2_s_f32_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32_"+std::to_string(i+1), mul_mat_vec_iq3_xxs_f32_f32_len, mul_mat_vec_iq3_xxs_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32_"+std::to_string(i+1),   mul_mat_vec_iq3_s_f32_f32_len,   mul_mat_vec_iq3_s_f32_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32_"+std::to_string(i+1),  mul_mat_vec_iq4_xs_f32_f32_len,  mul_mat_vec_iq4_xs_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32_"+std::to_string(i+1),  mul_mat_vec_iq4_nl_f32_f32_len,  mul_mat_vec_iq4_nl_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32_"+std::to_string(i+1),   mul_mat_vec_mxfp4_f32_f32_len,   mul_mat_vec_mxfp4_f32_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32_"+std::to_string(i+1),  mul_mat_vec_f32_f16_f32_len,  mul_mat_vec_f32_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32_"+std::to_string(i+1),  mul_mat_vec_f16_f16_f32_len,  mul_mat_vec_f16_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32_"+std::to_string(i+1), mul_mat_vec_bf16_f16_f32_len, mul_mat_vec_bf16_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {device->subgroup_size, 2, i+1}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32_"+std::to_string(i+1), mul_mat_vec_q4_0_f16_f32_len, mul_mat_vec_q4_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32_"+std::to_string(i+1), mul_mat_vec_q4_1_f16_f32_len, mul_mat_vec_q4_1_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32_"+std::to_string(i+1), mul_mat_vec_q5_0_f16_f32_len, mul_mat_vec_q5_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32_"+std::to_string(i+1), mul_mat_vec_q5_1_f16_f32_len, mul_mat_vec_q5_1_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32_"+std::to_string(i+1), mul_mat_vec_q8_0_f16_f32_len, mul_mat_vec_q8_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {device->subgroup_size, 1*rm_stdq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32_"+std::to_string(i+1), mul_mat_vec_q2_k_f16_f32_len, mul_mat_vec_q2_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32_"+std::to_string(i+1), mul_mat_vec_q3_k_f16_f32_len, mul_mat_vec_q3_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32_"+std::to_string(i+1), mul_mat_vec_q4_k_f16_f32_len, mul_mat_vec_q4_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32_"+std::to_string(i+1), mul_mat_vec_q5_k_f16_f32_len, mul_mat_vec_q5_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32_"+std::to_string(i+1), mul_mat_vec_q6_k_f16_f32_len, mul_mat_vec_q6_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32_"+std::to_string(i+1),   mul_mat_vec_iq1_s_f16_f32_len,   mul_mat_vec_iq1_s_f16_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32_"+std::to_string(i+1),   mul_mat_vec_iq1_m_f16_f32_len,   mul_mat_vec_iq1_m_f16_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32_"+std::to_string(i+1), mul_mat_vec_iq2_xxs_f16_f32_len, mul_mat_vec_iq2_xxs_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32_"+std::to_string(i+1),  mul_mat_vec_iq2_xs_f16_f32_len,  mul_mat_vec_iq2_xs_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32_"+std::to_string(i+1),   mul_mat_vec_iq2_s_f16_f32_len,   mul_mat_vec_iq2_s_f16_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32_"+std::to_string(i+1), mul_mat_vec_iq3_xxs_f16_f32_len, mul_mat_vec_iq3_xxs_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32_"+std::to_string(i+1),   mul_mat_vec_iq3_s_f16_f32_len,   mul_mat_vec_iq3_s_f16_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32_"+std::to_string(i+1),  mul_mat_vec_iq4_xs_f16_f32_len,  mul_mat_vec_iq4_xs_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32_"+std::to_string(i+1),  mul_mat_vec_iq4_nl_f16_f32_len,  mul_mat_vec_iq4_nl_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32_"+std::to_string(i+1),   mul_mat_vec_mxfp4_f16_f32_len,   mul_mat_vec_mxfp4_f16_f32_data,   "main", 3, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq, i+1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",  mul_mat_vec_id_f32_f32_len,  mul_mat_vec_id_f32_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {device->subgroup_size, 2}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",  mul_mat_vec_id_f16_f32_len,  mul_mat_vec_id_f16_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {device->subgroup_size, 2}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32", mul_mat_vec_id_bf16_f32_len, mul_mat_vec_id_bf16_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {device->subgroup_size, 2}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32", mul_mat_vec_id_q4_0_f32_len, mul_mat_vec_id_q4_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32", mul_mat_vec_id_q4_1_f32_len, mul_mat_vec_id_q4_1_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32", mul_mat_vec_id_q5_0_f32_len, mul_mat_vec_id_q5_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32", mul_mat_vec_id_q5_1_f32_len, mul_mat_vec_id_q5_1_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {device->subgroup_size, 2*rm_stdq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32", mul_mat_vec_id_q8_0_f32_len, mul_mat_vec_id_q8_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq, 1, 1}, {device->subgroup_size, 1*rm_stdq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32", mul_mat_vec_id_q2_k_f32_len, mul_mat_vec_id_q2_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32", mul_mat_vec_id_q3_k_f32_len, mul_mat_vec_id_q3_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32", mul_mat_vec_id_q4_k_f32_len, mul_mat_vec_id_q4_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32", mul_mat_vec_id_q5_k_f32_len, mul_mat_vec_id_q5_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32", mul_mat_vec_id_q6_k_f32_len, mul_mat_vec_id_q6_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {subgroup_size_16, rm_kq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   mul_mat_vec_id_iq1_s_f32_len,   mul_mat_vec_id_iq1_s_f32_data,   "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   mul_mat_vec_id_iq1_m_f32_len,   mul_mat_vec_id_iq1_m_f32_data,   "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", mul_mat_vec_id_iq2_xxs_f32_len, mul_mat_vec_id_iq2_xxs_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  mul_mat_vec_id_iq2_xs_f32_len,  mul_mat_vec_id_iq2_xs_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   mul_mat_vec_id_iq2_s_f32_len,   mul_mat_vec_id_iq2_s_f32_data,   "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", mul_mat_vec_id_iq3_xxs_f32_len, mul_mat_vec_id_iq3_xxs_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   mul_mat_vec_id_iq3_s_f32_len,   mul_mat_vec_id_iq3_s_f32_data,   "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  mul_mat_vec_id_iq4_xs_f32_len,  mul_mat_vec_id_iq4_xs_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  mul_mat_vec_id_iq4_nl_f32_len,  mul_mat_vec_id_iq4_nl_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   mul_mat_vec_id_mxfp4_f32_len,   mul_mat_vec_id_mxfp4_f32_data,   "main", 4, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {subgroup_size_16, rm_iq}, 1, true);

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, 5 * sizeof(uint32_t), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1, "quantize_q8_1", quantize_q8_1_len, quantize_q8_1_data, "main", 2, 1 * sizeof(uint32_t), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1);

    for (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        if (device->subgroup_add && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_data, "main", 3, 6 * sizeof(uint32_t), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true, true);
        } else {
            ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_data,              "main", 3, 6 * sizeof(uint32_t), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", 3, 12 * sizeof(uint32_t), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f32, "cpy_f32_f32", cpy_f32_f32_len, cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f16, "cpy_f16_f16", cpy_f16_f16_len, cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f32, "contig_cpy_f32_f32", contig_cpy_f32_f32_len, contig_cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f16, "contig_cpy_f16_f16", contig_cpy_f16_f16_len, contig_cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    if (device->float_controls_rte_fp16) {
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_rte_len, cpy_f32_q4_0_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_rte_len, cpy_f32_q4_1_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_rte_len, cpy_f32_q5_0_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_rte_len, cpy_f32_q5_1_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_rte_len, cpy_f32_q8_0_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_rte_len, cpy_f32_iq4_nl_rte_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    }

    if (device->float_controls_rte_fp16) {
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_F32],  "set_rows_f32",  set_rows_f32_rte_len,  set_rows_f32_rte_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_F16],  "set_rows_f16",  set_rows_f16_rte_len,  set_rows_f16_rte_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_BF16], "set_rows_bf16", set_rows_bf16_rte_len, set_rows_bf16_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q4_0], "set_rows_q4_0", set_rows_q4_0_rte_len, set_rows_q4_0_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q4_1], "set_rows_q4_1", set_rows_q4_1_rte_len, set_rows_q4_1_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q5_0], "set_rows_q5_0", set_rows_q5_0_rte_len, set_rows_q5_0_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q5_1], "set_rows_q5_1", set_rows_q5_1_rte_len, set_rows_q5_1_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q8_0], "set_rows_q8_0", set_rows_q8_0_rte_len, set_rows_q8_0_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_IQ4_NL], "set_rows_iq4_nl", set_rows_iq4_nl_rte_len, set_rows_iq4_nl_rte_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_F32],  "set_rows_f32",  set_rows_f32_len,  set_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_F16],  "set_rows_f16",  set_rows_f16_len,  set_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_BF16], "set_rows_bf16", set_rows_bf16_len, set_rows_bf16_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q4_0], "set_rows_q4_0", set_rows_q4_0_len, set_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q4_1], "set_rows_q4_1", set_rows_q4_1_len, set_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q5_0], "set_rows_q5_0", set_rows_q5_0_len, set_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q5_1], "set_rows_q5_1", set_rows_q5_1_len, set_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_Q8_0], "set_rows_q8_0", set_rows_q8_0_len, set_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_set_rows[GGML_TYPE_IQ4_NL], "set_rows_iq4_nl", set_rows_iq4_nl_len, set_rows_iq4_nl_data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };

    bool rte = device->float_controls_rte_fp16;
#define CREATE_BINARY(name, namemod, spec) \
    for (int s0 : {0,1}) for (int s1 : {0,1}) for (int d : {0,1}) \
        ggml_vk_create_pipeline(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d][rte], name ## _data[s0][s1][d][rte], \
                                "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, spec, 1);

    CREATE_BINARY(add, , {0})
    CREATE_BINARY(add, _norepeat, {1})
    CREATE_BINARY(sub, , {0})
    CREATE_BINARY(sub, _norepeat, {1})
    CREATE_BINARY(mul, , {0})
    CREATE_BINARY(mul, _norepeat, {1})
    CREATE_BINARY(div, , {0})
    CREATE_BINARY(div, _norepeat, {1})
#undef CREATE_BINARY

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_data, "main", 4, sizeof(vk_op_add_id_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_f32, "concat_f32", concat_f32_len, concat_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_f16, "concat_f16", concat_f16_len, concat_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_ac_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sqr_f32, "sqr_f32", sqr_f32_len, sqr_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sin_f32, "sin_f32", sin_f32_len, sin_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cos_f32, "cos_f32", cos_f32_len, cos_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_clamp_f32, "clamp_f32", clamp_f32_len, clamp_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_f32, "repeat_f32", repeat_f32_len, repeat_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
#undef CREATE_UNARY

#define CREATE_GLU(name)  \
    if (device->float_controls_rte_fp16) {  \
        ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32_rte", name ## _f32_rte_len, name ## _f32_rte_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
        ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16_rte", name ## _f16_rte_len, name ## _f16_rte_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    } else {    \
        ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
        ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    }

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#undef CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu_f32, "leaky_relu_f32", leaky_relu_f32_len, leaky_relu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, sizeof(vk_op_diag_mask_push_constants), {1, 512, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    if (device->float_controls_rte_fp16) {
        ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_rte_len, rope_norm_f16_rte_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_rte_len, rope_neox_f16_rte_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_rte_len, rope_multi_f16_rte_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_rte_len, rope_vision_f16_rte_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argsort_f32, "argsort_f32", argsort_f32_len, argsort_f32_data, "main", 2, sizeof(vk_op_argsort_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows_f32, "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_count_equal_i32, "count_equal_i32", count_equal_i32_len, count_equal_i32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32_len, im2col_f32_data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);
    if (device->float_controls_rte_fp16) {
        ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16_rte_len, im2col_f32_f16_rte_data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16_len, im2col_f32_f16_data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, sizeof(vk_op_timestep_embedding_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_data, "main", 3, sizeof(vk_op_conv_transpose_1d_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_data, "main", 2, sizeof(vk_op_pool2d_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_data, "main", 7, sizeof(vk_op_rwkv_wkv6_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_data, "main", 8, sizeof(vk_op_rwkv_wkv7_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_data, "main", 5, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    // conv2d
    for (uint32_t s = 0; s < CONV_SHAPE_COUNT; ++s) {
        uint32_t conv2d_WG_SIZE  = 256;
        uint32_t conv2d_BS_K     = 128;
        uint32_t conv2d_BS_CRS   = 16;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_BS_NPQ = 128;
        uint32_t conv2d_TS_K   = 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        bool conv2d_UNROLL = true;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#endif

        if (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } else if (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
        }

        switch (s) {
        default:
        case CONV_SHAPE_128x128:
            conv2d_BS_K = 128;
            conv2d_BS_NPQ = 128;
            conv2d_BS_CRS = 16;
            if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = false;
            }
            break;
        case CONV_SHAPE_64x32:
            conv2d_BS_K = 64;
            conv2d_BS_NPQ = 32;
            conv2d_BS_CRS = 32;
            conv2d_TS_K   = 4;
            break;
        case CONV_SHAPE_32x256:
            conv2d_BS_K = 32;
            conv2d_BS_NPQ = 256;
            conv2d_BS_CRS = 16;
            break;
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        if (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS_CRS   = std::min(
                device->subgroup_size,
                conv2d_BS_CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        uint32_t conv2d_shmem_req =
            (conv2d_BS_K * (conv2d_BS_CRS + conv2d_SHMEM_PAD) + conv2d_BS_CRS * (conv2d_BS_NPQ + conv2d_SHMEM_PAD)) * sizeof(float);
        if (device->properties.limits.maxComputeSharedMemorySize < conv2d_shmem_req) {
            conv2d_BS_CRS = 8;
            if (use_collectives) {
                conv2d_BS_CRS = std::min(device->subgroup_size, conv2d_BS_CRS);
            }
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS_K, conv2d_BS_NPQ, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_SIZE, conv2d_BS_K, conv2d_BS_CRS, conv2d_BS_NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f32[s], "conv2d_f32", conv2d_f32_cm2_len, conv2d_f32_cm2_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f16_f32[s], "conv2d_f16_f32", conv2d_f16_f32_cm2_len, conv2d_f16_f32_cm2_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
        } else
#endif
        if (conv2d_UNROLL) {
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f32[s], "conv2d_f32", conv2d_f32_unroll_len, conv2d_f32_unroll_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f16_f32[s], "conv2d_f16_f32", conv2d_f16_f32_unroll_len, conv2d_f16_f32_unroll_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
        } else {
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f32[s], "conv2d_f32", conv2d_f32_len, conv2d_f32_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
            ggml_vk_create_pipeline(
                device, device->pipeline_conv2d_f16_f32[s], "conv2d_f16_f32", conv2d_f16_f32_len, conv2d_f16_f32_data, "main", 3,
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);

    for (auto &c : compiles) {
        c.wait();
    }
    device->need_compiles = false;
}

static bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch);

static vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    if (vk_instance.devices[idx] == nullptr) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();
        vk_instance.devices[idx] = device;

#ifdef GGML_VULKAN_MEMORY_DEBUG
        device->memory_logger = std::unique_ptr<vk_memory_logger>(new vk_memory_logger());
#endif
        if (vk_perf_logger_enabled) {
            device->perf_logger = std::unique_ptr<vk_perf_logger>(new vk_perf_logger());
        }

        size_t dev_num = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        if (dev_num >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
            throw std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_num];
        const std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        const char* GGML_VK_PREFER_HOST_MEMORY = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_memory = GGML_VK_PREFER_HOST_MEMORY != nullptr;

        const char* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;

        bool fp16_storage = false;
        bool fp16_compute = false;
        bool maintenance4_support = false;
        bool sm_builtins = false;
        bool amd_shader_core_properties2 = false;
        bool pipeline_robustness = false;
        bool coopmat2_support = false;
        device->coopmat_support = false;
        device->integer_dot_product = false;
        bool bfloat16_support = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = true;
            } else if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = true;
            } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = true;
            } else if (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = true;
            } else if (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = true;
            } else if (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = true;
            } else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_size_control = true;
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = true;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#endif
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } else if (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = true;
#endif
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_product = true;
#endif
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } else if (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = true;
#endif
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceDriverProperties driver_props;
        vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
        vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
        vk::PhysicalDeviceVulkan11Properties vk11_props;
        vk::PhysicalDeviceVulkan12Properties vk12_props;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
        vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;

        props2.pNext = &props3;
        props3.pNext = &subgroup_props;
        subgroup_props.pNext = &driver_props;
        driver_props.pNext = &vk11_props;
        vk11_props.pNext = &vk12_props;

        VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&vk12_props;

        if (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&props4;
            last_struct = (VkBaseOutStructure *)&props4;
        }
        if (sm_builtins) {
            last_struct->pNext = (VkBaseOutStructure *)&sm_props;
            last_struct = (VkBaseOutStructure *)&sm_props;
        }
        if (amd_shader_core_properties2) {
            last_struct->pNext = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
            last_struct = (VkBaseOutStructure *)&amd_shader_core_properties2_props;
        }
        if (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_props;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_props;
        }

#if defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        if (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_props;
            last_struct = (VkBaseOutStructure *)&coopmat2_props;
        }
#endif

        if (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        const char* GGML_VK_FORCE_MAX_ALLOCATION_SIZE = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        if (GGML_VK_FORCE_MAX_ALLOCATION_SIZE != nullptr) {
            device->max_memory_allocation_size = std::stoul(GGML_VK_FORCE_MAX_ALLOCATION_SIZE);
        } else if (maintenance4_support) {
            device->max_memory_allocation_size = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
        } else {
            device->max_memory_allocation_size = props3.maxMemoryAllocationSize;
        }

        const char* GGML_VK_SUBALLOCATION_BLOCK_SIZE = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        if (GGML_VK_SUBALLOCATION_BLOCK_SIZE != nullptr) {
            device->suballocation_block_size = std::stoul(GGML_VK_SUBALLOCATION_BLOCK_SIZE);
        } else {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_block_size = 1024*1024*1024;
        }
        device->suballocation_block_size = std::min(device->suballocation_block_size, device->max_memory_allocation_size);

        device->subgroup_size = subgroup_props.subgroupSize;
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        if (sm_builtins) {
            device->shader_core_count = sm_props.shaderSMCount;
        } else if (amd_shader_core_properties2) {
            device->shader_core_count = amd_shader_core_properties2_props.activeComputeUnitCount;
        } else {
            device->shader_core_count = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;

        device->subgroup_add = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                               (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);

        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);

        const bool force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != nullptr;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        if (!ggml_vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = false;
        }

        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        const uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, -1, 1);
        const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics, compute_queue_family_index, 1);

        const float priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_index == transfer_queue_family_index && queue_family_props[compute_queue_family_index].queueCount == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        if (compute_queue_family_index != transfer_queue_family_index) {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, 1, priorities + 1});
        } else if(!device->single_queue) {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 2, priorities});
        } else {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
        }
        vk::DeviceCreateInfo device_create_info;
        std::vector<const char *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.pNext = nullptr;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.pNext = nullptr;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.pNext = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.pNext = nullptr;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.pNext = &vk12_features;

        last_struct = (VkBaseOutStructure *)&vk12_features;

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.pNext = nullptr;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        if (pipeline_robustness) {
            last_struct->pNext = (VkBaseOutStructure *)&pl_robustness_features;
            last_struct = (VkBaseOutStructure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features;
        subgroup_size_control_features.pNext = nullptr;
        subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        subgroup_size_control_features.computeFullSubgroups = false;
        subgroup_size_control_features.subgroupSizeControl = false;

        if (device->subgroup_size_control) {
            last_struct->pNext = (VkBaseOutStructure *)&subgroup_size_control_features;
            last_struct = (VkBaseOutStructure *)&subgroup_size_control_features;
        }

#if defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
        coopmat_features.pNext = nullptr;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopmat_features.cooperativeMatrix = VK_FALSE;

        if (device->coopmat_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
            last_struct = (VkBaseOutStructure *)&coopmat_features;
        }
#endif

#if defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.pNext = nullptr;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
        if (coopmat2_support) {
            last_struct->pNext = (VkBaseOutStructure *)&coopmat2_features;
            last_struct = (VkBaseOutStructure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#endif

#if defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.pNext = nullptr;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        if (bfloat16_support) {
            last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
            last_struct = (VkBaseOutStructure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#endif

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        if (maintenance4_support) {
            last_struct->pNext = (VkBaseOutStructure *)&maint4_features;
            last_struct = (VkBaseOutStructure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
        shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        if (device->integer_dot_product) {
            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#if defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#else
        device->bf16 = false;
#endif

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        if (device->subgroup_size_control) {
            device->subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
            device->subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_size_control = device->subgroup_size_control &&
                (subgroup_size_control_props.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute) &&
                subgroup_size_control_features.subgroupSizeControl;

        if (device->subgroup_size_control) {
            device->subgroup_require_full_support = subgroup_size_control_features.computeFullSubgroups;
        }

#if defined(VK_KHR_cooperative_matrix)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;

        // coopmat1 fa shader currently assumes 32 invocations per subgroup
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_full_support &&
                                      device->subgroup_size_control && device->subgroup_min_size <= 32 &&
                                      device->subgroup_max_size >= 32;
#endif

        if (coopmat2_support) {
#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperativeMatrixPerElementOperations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperativeMatrixBlockLoads &&
                vk12_features.bufferDeviceAddress) {

                std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
                    _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
                        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, nullptr);

                VkCooperativeMatrixFlexibleDimensionsPropertiesNV empty_prop {};
                empty_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, empty_prop);

                _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, flexible_dimensions.data());

                bool found_fp16_128 = false,
                     found_fp16_256 = false,
                     found_fp32_128 = false,
                     found_fp32_256 = false;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                for (auto &prop : flexible_dimensions) {
                    if (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR &&
                        prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                        prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                        if (prop.workgroupInvocations == 128 &&
                            prop.MGranularity <= 32 &&
                            prop.NGranularity <= 16 &&
                            prop.KGranularity <= 16) {
                            if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                found_fp16_128 = true;
                            }
                            if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                found_fp32_128 = true;
                            }
                        }
                        if (prop.workgroupInvocations == 256 &&
                            prop.MGranularity <= 32 &&
                            prop.NGranularity <= 32 &&
                            prop.KGranularity <= 16) {
                            if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                found_fp16_256 = true;
                            }
                            if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                found_fp32_256 = true;
                            }
                        }
                    }
                }
                if (found_fp16_128 && found_fp16_256 &&
                    found_fp32_128 && found_fp32_256 &&
                    coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
                    device->coopmat2 = true;
                }
            }
#endif
        }

        if (!vk11_features.storageBuffer16BitAccess) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::endl;
            throw std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ifdef GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#endif

        if (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#if defined(VK_KHR_cooperative_matrix)
        if (device->coopmat_support) {
            // Query supported shapes
            std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

            PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
                (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_num;

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, nullptr);

            cm_props.resize(cm_props_num);

            for (auto& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }

            pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, cm_props.data());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            for (auto& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ResultType) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                if ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        if (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = true;
                        }
                        if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = true;
                        }
                    } else if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        if (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = true;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = true;
                        }
                        if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = true;
                        }
                    }
                } else if ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = true;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    if (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = true;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = true;
                    }
                }
#endif
            }

            if (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = false;
            }
            if (getenv("GGML_VK_DISABLE_BFLOAT16")) {
                device->coopmat_bf16_support = false;
            }
        }

        if (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#if defined(VK_KHR_shader_bfloat16)
        if (device->coopmat_bf16_support) {
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#endif
#endif
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info = {
            vk::DeviceCreateFlags(),
            device_queue_create_infos,
            {},
            device_extensions
        };
        device_create_info.setPNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        // Queues
        ggml_vk_create_queue(device, device->compute_queue, compute_queue_family_index, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer }, false);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            switch (device->vendor_id) {
#ifndef GGML_VULKAN_RUN_TESTS
            case VK_VENDOR_ID_AMD:
            case VK_VENDOR_ID_INTEL:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            case VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = false;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = false;
                device->mul_mat_id_l[i] = false;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = false;
                break;
#endif
            default:
                device->mul_mat_l[i] = true;
                device->mul_mat_m[i] = true;
                device->mul_mat_s[i] = true;
                device->mul_mat_id_l[i] = true;
                device->mul_mat_id_m[i] = true;
                device->mul_mat_id_s[i] = true;
                break;
            }
        }


        std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        for (uint32_t i = 0; i < MAX_PARAMETER_COUNT; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
            {},
            dsl_binding);
        descriptor_set_layout_create_info.setPNext(&dslbfci);
        device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

        ggml_vk_load_shaders(device);

        if (!device->single_queue) {
            const uint32_t transfer_queue_index = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;
            ggml_vk_create_queue(device, device->transfer_queue, transfer_queue_family_index, transfer_queue_index, { vk::PipelineStageFlagBits::eTransfer }, true);
        } else {
            // TODO: Use pointer or reference to avoid copy
            device->transfer_queue.copyFrom(device->compute_queue);
            device->transfer_queue.cmd_pool.init(device, &device->transfer_queue);
        }

        device->buffer_type = {
            /* .iface    = */ ggml_backend_vk_buffer_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ new ggml_backend_vk_buffer_type_context{ device->name, device },
        };

        device->fence = device->device.createFence({});

        device->idx = idx;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != nullptr;

        return device;
    }

    return vk_instance.devices[idx];
}

static void ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_num = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_num << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    if (dev_num >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
        throw std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_num];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    bool fp16_storage = false;
    bool fp16_compute = false;
    bool coopmat_support = false;
    bool coopmat2_support = false;
    bool integer_dot_product = false;
    bool bfloat16_support = false;

    for (auto properties : ext_props) {
        if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = true;
        } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = true;
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = true;
#endif
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } else if (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = true;
#endif
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_product = true;
#endif
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } else if (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = true;
#endif
        }
    }

    const vk_device_architecture device_architecture = get_device_architecture(physical_device);

    const char* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    bool force_disable_f16 = GGML_VK_DISABLE_F16 != nullptr;

    bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;

    // Pointer to the last chain element
    VkBaseOutStructure * last_struct = (VkBaseOutStructure *)&driver_props;

    if (integer_dot_product) {
        last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_props;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext = nullptr;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.pNext = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.pNext = &vk12_features;

    // Pointer to the last chain element
    last_struct = (VkBaseOutStructure *)&vk12_features;

#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
    coopmat_features.pNext = nullptr;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat_features.cooperativeMatrix = VK_FALSE;

    if (coopmat_support) {
        last_struct->pNext = (VkBaseOutStructure *)&coopmat_features;
        last_struct = (VkBaseOutStructure *)&coopmat_features;
    }
#endif

    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};
    shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    if (integer_dot_product) {
        last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;
        last_struct = (VkBaseOutStructure *)&shader_integer_dot_product_features;
    }

#if defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    if (bfloat16_support) {
        last_struct->pNext = (VkBaseOutStructure *)&bfloat16_features;
        last_struct = (VkBaseOutStructure *)&bfloat16_features;
    }
#endif

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#if defined(VK_KHR_shader_bfloat16)
    bool bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#else
    bool bf16 = false;
#endif

    uint32_t default_subgroup_size = get_subgroup_size("", device_architecture);
    const size_t subgroup_size = (default_subgroup_size != 0) ? default_subgroup_size : subgroup_props.subgroupSize;
    const bool uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_product = integer_dot_product
                       && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated
                       && shader_integer_dot_product_features.shaderIntegerDotProduct;

    coopmat_support = coopmat_support
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperativeMatrix
#endif
                   && ggml_vk_khr_cooperative_matrix_support(props2.properties, driver_props, device_architecture);

    std::string matrix_cores = coopmat2_support ? "NV_coopmat2" : coopmat_support ? "KHR_coopmat" : "none";

    std::string device_name = props2.properties.deviceName.data();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %d | bf16: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.data(), uma, fp16, bf16, subgroup_size,
              props2.properties.limits.maxComputeSharedMemorySize, integer_dot_product, matrix_cores.c_str());

    if (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);

static bool ggml_vk_instance_debug_utils_ext_available(const std::vector<vk::ExtensionProperties> & instance_extensions);

static void ggml_vk_instance_init() {
    if (vk_instance_initialized) {
        return;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    uint32_t api_version = vk::enumerateInstanceVersion();

    if (api_version < VK_API_VERSION_1_2) {
        std::cerr << "ggml_vulkan: Error: Vulkan 1.2 required." << std::endl;
        GGML_ABORT("fatal error");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, api_version };

    const std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    const bool validation_ext = ggml_vk_instance_validation_ext_available(instance_extensions);
#ifdef __APPLE__
    const bool portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_available(instance_extensions);
#endif
    const bool debug_utils_ext = ggml_vk_instance_debug_utils_ext_available(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != nullptr;
    std::vector<const char*> layers;

    if (validation_ext) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<const char*> extensions;
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features");
    }
#ifdef __APPLE__
    if (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#endif
    if (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions);
#ifdef __APPLE__
    if (portability_enumeration_ext) {
        instance_create_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#endif

    std::vector<vk::ValidationFeatureEnableEXT> features_enable;
    vk::ValidationFeaturesEXT validation_features;

    if (validation_ext) {
        features_enable = { vk::ValidationFeatureEnableEXT::eBestPractices };
        validation_features = {
            features_enable,
            {},
        };
        validation_features.setPNext(nullptr);
        instance_create_info.setPNext(&validation_features);
        GGML_LOG_DEBUG("ggml_vulkan: Validation layers enabled\n");
    }
    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = true;

    if (debug_utils_ext) {
        vk_instance.debug_utils_support              = true;
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT =   (PFN_vkCmdEndDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdInsertDebugUtilsLabelEXT = (PFN_vkCmdInsertDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");

    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != nullptr;

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    char * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    if (devices_env != nullptr) {
        size_t num_available_devices = vk_instance.instance.enumeratePhysicalDevices().size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        while (ss >> tmp) {
            if(tmp >= num_available_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::endl;
                throw std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } else {
        std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

        // If no vulkan devices are found, return early
        if (devices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            return;
        }

        // Default to using all dedicated GPUs
        for (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.pNext = &new_driver;
            new_driver.pNext = &new_id;
            devices[i].getProperties2(&new_props);

            if (new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                // Check if there are two physical devices corresponding to the same GPU
                auto old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id](const size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.pNext = &old_id;
                        devices[k].getProperties2(&old_props);
                        return std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                    }
                );
                if (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } else {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.pNext = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, int> driver_priorities {};
                    int old_priority = std::numeric_limits<int>::max();
                    int new_priority = std::numeric_limits<int>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    switch (old_props.properties.vendorID) {
                        case VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::eAmdOpenSource] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            break;
                        case VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            break;
                        case VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#if defined(VK_API_VERSION_1_3) && VK_HEADER_VERSION >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#endif
                            break;
                    }

                    if (driver_priorities.count(old_driver.driverID)) {
                        old_priority = driver_priorities[old_driver.driverID];
                    }
                    if (driver_priorities.count(new_driver.driverID)) {
                        new_priority = driver_priorities[new_driver.driverID];
                    }

                    if (new_priority < old_priority) {
                        auto r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    else {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::endl);
                    }
                }
            }
        }

        // If no dedicated GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        if (vk_instance.device_indices.empty()) {
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    break;
                }
            }
        }

        if (vk_instance.device_indices.empty()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            return;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        ggml_vk_print_gpu_info(i);
    }
}

static void ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, &ctx->device->compute_queue);
    ctx->transfer_cmd_pool.init(ctx->device, &ctx->device->transfer_queue);

#ifdef GGML_VULKAN_CHECK_RESULTS
    const char* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == NULL ? 0 : atoi(skip_checks));
    const char* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == NULL ? 0 : atoi(output_tensor));
#endif
}

static vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    switch (type) {
        case GGML_TYPE_F32:
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
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant[type];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_pipeline(" << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ", " << prec << ")");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_f32;
    }
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
        return ctx->device->pipeline_matmul_f32_f16;
    }
    if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
        return ctx->device->pipeline_matmul_bf16;
    }
    if (prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_f16_f32.f16acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_f16.f16acc;
        }
    } else {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_f16_f32.f32acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_f16.f32acc;
        }
    }

    // MMQ
    if (src1_type == GGML_TYPE_Q8_1) {
        vk_matmul_pipeline pipelines = (ctx->device->fp16 && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f32acc;

        if (pipelines->s == nullptr && pipelines->m == nullptr && pipelines->l == nullptr) {
            return nullptr;
        }

        return pipelines;
    }

    if (src1_type != GGML_TYPE_F32 && !ctx->device->coopmat2) {
        return nullptr;
    }

    switch (src0_type) {
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
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return nullptr;
    }

    if (ctx->device->coopmat2) {
        assert(src1_type == GGML_TYPE_F16);
        return prec == GGML_PREC_DEFAULT ? ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f32acc;
    }
    if (ctx->device->coopmat_support) {
        return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
    }
    return (ctx->device->fp16 && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t num_cols) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16);
    GGML_ASSERT(num_cols >= 1 && num_cols <= mul_mat_vec_max_cols);

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
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
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return nullptr;
    }

    return b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[a_type][num_cols-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[a_type][num_cols-1];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_id_pipeline()");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_id_f32;
    }
    if (src0_type == GGML_TYPE_BF16 && src1_type == GGML_TYPE_BF16) {
        return ctx->device->pipeline_matmul_id_bf16;
    }
    if (prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support)) {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_id_f16_f32.f16acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_id_f16.f16acc;
        }
    } else {
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
            return ctx->device->pipeline_matmul_id_f16_f32.f32acc;
        }
        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
            return ctx->device->pipeline_matmul_id_f16.f32acc;
        }
    }

    GGML_ASSERT(src1_type == GGML_TYPE_F32 || (ctx->device->coopmat2 && src1_type == GGML_TYPE_F16));

    switch (src0_type) {
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
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return nullptr;
    }

    return ctx->device->fp16 ? ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f32acc;
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32);

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
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
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[a_type];
}

static vk_buffer ggml_vk_pool_malloc(ggml_backend_vk_context * ctx, size_t size) {
    VK_LOG_DEBUG("ggml_vk_pool_malloc(" << size << ")");
    VK_LOG_MEMORY("ggml_vk_pool_malloc");

    int best_i = -1;
    size_t best_size = std::numeric_limits<size_t>::max(); //smallest unused buffer that fits our needs
    int worst_i = -1;
    size_t worst_size = 0; //largest unused buffer seen so far
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer &b = ctx->buffer_pool[i];
        if (b != nullptr && b->size >= size && b->size < best_size) {
            best_i = i;
            best_size = b->size;
        }
        if (b != nullptr && b->size > worst_size) {
            worst_i = i;
            worst_size = b->size;
        }
    }
    if(best_i != -1) {
        //found the smallest buffer that fits our needs
        vk_buffer b = ctx->buffer_pool[best_i];
        ctx->buffer_pool[best_i].reset();
        return b;
    }
    if(worst_i != -1) {
        //no buffer that fits our needs, resize largest one to save memory
        vk_buffer& b = ctx->buffer_pool[worst_i];
        ggml_vk_destroy_buffer(b);
    }

    return ggml_vk_create_buffer_device(ctx->device, size);
}

static void ggml_vk_pool_free(ggml_backend_vk_context * ctx, vk_buffer& buffer) {
    VK_LOG_DEBUG("ggml_vk_pool_free(" << buffer->size << ")");
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer& b = ctx->buffer_pool[i];
        if (b == nullptr) {
            b = buffer;
            return;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: vk buffer pool full, increase MAX_VK_BUFFERS" << std::endl;
    ggml_vk_destroy_buffer(buffer);
}

// Returns an available temporary buffer that may only be used temporarily, it will be reused
static vk_buffer ggml_vk_create_buffer_temp(ggml_backend_vk_context * ctx, size_t size) {
    // Try to find existing temp buffer with enough capacity
    for (auto& buffer : ctx->gc.temp_buffers) {
        if (buffer->size >= size) {
            return buffer;
        }
    }

    VK_LOG_MEMORY("ggml_vk_create_buffer_temp(" << size << ")");

    // Otherwise create new buffer
    vk_buffer buf = ggml_vk_pool_malloc(ctx, size);
    ctx->gc.temp_buffers.push_back(buf);

    return buf;
}

static void * ggml_vk_host_malloc(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");
    vk_buffer buf = ggml_vk_create_buffer(device, size,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    if(!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        device->device.freeMemory(buf->device_memory);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));

    return buf->ptr;
}

static void ggml_vk_host_free(vk_device& device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    vk_buffer buf;
    size_t index;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(buf);

    device->pinned_memory.erase(device->pinned_memory.begin() + index);
}

static void ggml_vk_host_get(vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    buf = nullptr;
    buf_offset = 0;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}

static vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, bool one_time = true) {
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(device, p);
    if (one_time) {
        s.buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    } else {
        s.buffer.begin({ vk::CommandBufferUsageFlags{} });
    }

    return s;
}

template <typename T> size_t push_constant_size(const T &t) {
    static_assert(std::is_class<T>::value, "T must be a struct/class");
    GGML_UNUSED(t);
    return sizeof(T);
}
template <typename T> size_t push_constant_size(const std::vector<T> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * t.size();
}
template <typename T, uint32_t N> size_t push_constant_size(const std::array<T, N> &t) {
    GGML_UNUSED(t);
    return sizeof(T) * N;
}

template <typename T> const T *push_constant_data(const T &t) {
    static_assert(std::is_class<T>::value, "T must be a struct/class");
    return &t;
}
template <typename T> const T *push_constant_data(const std::vector<T> &t) {
    return t.data();
}
template <typename T, uint32_t N> const T *push_constant_data(const std::array<T, N> &t) {
    return t.data();
}

template <typename T>
static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos, const T &push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
    GGML_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());

    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
    vk::WriteDescriptorSet write_descriptor_set{ descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr, descriptor_buffer_infos.begin() };
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    subctx->s->buffer.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size(push_constants), push_constant_data(push_constants));
    subctx->s->buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    subctx->s->buffer.dispatch(wg0, wg1, wg2);
}

static void ggml_vk_end_submission(vk_submission& s, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
    s.buffer.end();

    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
}

static void ggml_vk_ctx_end(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    if (ctx->s == nullptr) {
        return;
    }

    ctx->s->buffer.end();
    ctx->s = nullptr;
}

static void ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    if (subctx->s != nullptr) {
        ggml_vk_ctx_end(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();
}

static size_t ggml_vk_align_size(size_t width, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << width << ", " << align << ")");
    return CEIL_DIV(width, align) * align;
}

static void deferred_memcpy(void * dst, const void * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr) {
    if (memcpys == nullptr) {
        memcpy(dst, src, size);
    } else {
        memcpys->emplace_back(dst, src, size);
    }
}

static void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i1*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(subctx);
    vkCmdCopyBuffer(subctx->s->buffer, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

static void ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * width;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    const size_t copy_size = width*height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, copy_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    VkBufferCopy buf_copy = {
        0,
        offset,
        copy_size};

    ggml_vk_sync_buffers(subctx);
    vkCmdCopyBuffer(subctx->s->buffer, (VkBuffer)staging_buffer->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, width * height, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
}

static void ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, 1, sync_staging);
}

static void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->ptr + offset + i * width, (const uint8_t *) src + i * spitch, width);
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, width, height, true);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences");
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

static void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, 0, size, 1);
}

static void ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(src->buffer, buf->buffer, slices);

        return;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        GGML_ABORT("Asynchronous read from non-pinned memory not supported");
    }

    // Fall back to staging buffer
    const size_t copy_size = dpitch * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, copy_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    ggml_vk_sync_buffers(subctx);
    subctx->s->buffer.copyBuffer(src->buffer, staging_buffer->buffer, slices);

    deferred_memcpy(dst, staging_buffer->ptr, copy_size, &subctx->out_memcpys);
}

static void ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

static void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        memcpy(dst, (uint8_t *) src->ptr + offset, size);
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_read_async(subctx, src, offset, dst, size, true);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read waitForFences");
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

static void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

static void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue.cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences");
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);
        ggml_vk_ensure_sync_staging_buffer(dst->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // memcpy to dst staging buffer
        memcpy(dst->device->sync_staging->ptr, src->device->sync_staging->ptr, size);
        // Copy to dst buffer
        ggml_vk_buffer_copy(dst, dst_offset, dst->device->sync_staging, 0, size);
    }
}

static void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    ctx->s->buffer.fillBuffer(dst->buffer, offset, size, c);
}

static void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue.cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences");
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

static uint32_t ggml_vk_guess_split_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, const vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ")");

    uint32_t split_k = 1;
    if (ctx->device->shader_core_count != 0 && m >= pipeline->wg_denoms[0] && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        if (k >= 2048) {
            if (m_tiles * n_tiles <= ctx->device->shader_core_count / 2) {
                split_k = ctx->device->shader_core_count / (m_tiles * n_tiles);
            } else if (m_tiles * n_tiles <= ctx->device->shader_core_count * 2 / 3) {
                split_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            split_k = std::min(split_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            while (true) {
                if (split_k == 1) {
                    break;
                }
                uint32_t k_split = CEIL_DIV(k, split_k);
                k_split = ROUNDUP_POW2(k_split, 256);
                if (k_split * (split_k - 1) < k) {
                    break;
                }
                split_k--;
            }
        }
    }

    return split_k;
}

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline(" << m << ", " << n << ", " << aligned << ", " << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ")");

    if (ctx->device->coopmat2) {
        const uint32_t shader_core_count = ctx->device->shader_core_count;
        const uint32_t tiles_l = CEIL_DIV(m, mmp->a_l->wg_denoms[0]) * CEIL_DIV(n, mmp->a_l->wg_denoms[1]);
        const uint32_t tiles_m = CEIL_DIV(m, mmp->a_m->wg_denoms[0]) * CEIL_DIV(n, mmp->a_m->wg_denoms[1]);

        // Use large shader when the N dimension is greater than the medium shader's tile size
        uint32_t crossover_large = mmp->m->wg_denoms[1];

        // Prefer large over medium if either:
        // - medium or large tiles would overfill the GPU
        // - large tiles with a split_k==3 fits in the GPU and medium tiles with split_k==2 does not
        //   (medium with split_k==2 is probably better if it fits - more workgroups running and less split_k overhead)
        bool prefer_large = tiles_m > shader_core_count || tiles_l > shader_core_count ||
                            // split_k==3 with large tiles likely better than medium tiles with no split_k.
                            (tiles_l <= shader_core_count / 3 && tiles_m > shader_core_count / 2);

        if ((ctx->device->mul_mat_l[src0_type] && (n > crossover_large && prefer_large)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device->mul_mat_s[src0_type])) {
            return aligned ? mmp->a_l : mmp->l;
        }
        // Use medium shader when the N dimension is greater than the small shader's tile size
        uint32_t crossover_medium = mmp->s->wg_denoms[1];
        if ((ctx->device->mul_mat_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_s[src0_type]) {
            return aligned ? mmp->a_m : mmp->m;
        }
        return aligned ? mmp->a_s : mmp->s;
    }

    if ((ctx->device->mul_mat_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device->mul_mat_l[src0_type])) {
        return aligned ? mmp->a_s : mmp->s;
    }
    if ((ctx->device->mul_mat_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_l[src0_type]) {
        return aligned ? mmp->a_m : mmp->m;
    }
    return aligned ? mmp->a_l : mmp->l;

    GGML_UNUSED(src1_type);
}

static uint32_t ggml_vk_guess_matmul_pipeline_align(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ")");
    return ggml_vk_guess_matmul_pipeline(ctx, mmp, m, n, true, src0_type, src1_type)->align;
}

static void ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " << split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    ggml_vk_sync_buffers(subctx);
    if (split_k == 1) {
        const vk_mat_mat_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k, ne02, ne12, broadcast2, broadcast3, padded_n };
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { m, n, batch });
        return;
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_split = CEIL_DIV(k, split_k);
    k_split = ROUNDUP_POW2(k_split, 256);

    const vk_mat_mat_push_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k_split, ne02, ne12, broadcast2, broadcast3, padded_n };
    // Make sure enough workgroups get assigned for split k to work
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, split_k_buffer }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, batch });
    ggml_vk_sync_buffers(subctx);
    const std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), split_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_split_k_reduce, { split_k_buffer, d }, pc2, { m * n * batch, 1, 1 });
}

static vk_pipeline ggml_vk_guess_matmul_id_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_id_pipeline(" << m << ", " << n << ", " << aligned << ", " << ggml_type_name(src0_type) << ")");

    if (ctx->device->coopmat2) {
        // Use large shader when the N dimension is greater than the medium shader's tile size
        uint32_t crossover_large = mmp->m->wg_denoms[1];
        if ((ctx->device->mul_mat_id_l[src0_type] && (n > crossover_large)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->device->mul_mat_id_s[src0_type])) {
            return aligned ? mmp->a_l : mmp->l;
        }
        // Use medium shader when the N dimension is greater than the small shader's tile size
        uint32_t crossover_medium = mmp->s->wg_denoms[1];
        if ((ctx->device->mul_mat_id_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_id_s[src0_type]) {
            return aligned ? mmp->a_m : mmp->m;
        }
        return aligned ? mmp->a_s : mmp->s;
    }

    if ((ctx->device->mul_mat_id_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->device->mul_mat_id_l[src0_type])) {
        return aligned ? mmp->a_s : mmp->s;
    }
    if ((ctx->device->mul_mat_id_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_id_l[src0_type]) {
        return aligned ? mmp->a_m : mmp->m;
    }
    return aligned ? mmp->a_l : mmp->l;
}

static uint32_t ggml_vk_guess_matmul_id_pipeline_align(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, ggml_type src0_type) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << ggml_type_name(src0_type) << ")");
    return ggml_vk_guess_matmul_id_pipeline(ctx, mmp, m, n, true, src0_type)->align;
}

static void ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t padded_n) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ggml_vk_sync_buffers(subctx);
    const vk_mat_mat_id_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, padded_n };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids }, pc, { m, nei1, n_as });
}

static bool ggml_vk_dim01_contiguous(const ggml_tensor * tensor) {
    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

static vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src, const ggml_tensor * dst, ggml_type to) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    bool contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_f32;
        } else {
            return ctx->device->pipeline_cpy_f32_f32;
        }
    }
    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_F16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_f16;
        } else {
            return ctx->device->pipeline_cpy_f32_f16;
        }
    }
    if (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f16_f16;
        } else {
            return ctx->device->pipeline_cpy_f16_f16;
        }
    }
    if (src->type == GGML_TYPE_F16 && to == GGML_TYPE_F32) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f16_f32;
        } else {
            return ctx->device->pipeline_cpy_f16_f32;
        }
    }
    if (src->type == GGML_TYPE_F32 && to == GGML_TYPE_BF16) {
        if (contig) {
            return ctx->device->pipeline_contig_cpy_f32_bf16;
        } else {
            return ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    if (src->type == GGML_TYPE_F32) {
        switch (to) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return ctx->device->pipeline_cpy_f32_quant[to];
        default:
            break;
        }
    }

    if (to == GGML_TYPE_F32) {
        switch (src->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return ctx->device->pipeline_cpy_quant_f32[src->type];
        default:
            break;
        }
    }

    if (src->type == to) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(to) || ggml_type_size(src->type) == 2 || ggml_type_size(src->type) == 4);
        if ((ggml_type_size(src->type) % 4) == 0) {
            if (contig) {
                return ctx->device->pipeline_contig_cpy_f32_f32;
            } else {
                return ctx->device->pipeline_cpy_f32_f32;
            }
        } else {
            if (contig) {
                return ctx->device->pipeline_contig_cpy_f16_f16;
            } else {
                return ctx->device->pipeline_cpy_f16_f16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(to) << std::endl;
    GGML_ABORT("fatal error");
}

static void ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, const ggml_tensor * tensor, vk_subbuffer&& in, vk_subbuffer&& out) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");
    const int tensor_type_size = ggml_type_size(tensor->type);

    const uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    if (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } else if (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } else {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_push_constants pc = {
        (uint32_t)ne,
        (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3], (uint32_t)tensor->nb[0] / tensor_type_size, (uint32_t)tensor->nb[1] / tensor_type_size, (uint32_t)tensor->nb[2] / tensor_type_size, (uint32_t)tensor->nb[3] / tensor_type_size,
        (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3],                       1                   , (uint32_t)tensor->ne[0]                   , (uint32_t)(tensor->ne[0] * tensor->ne[1]) , (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]),
        0,
        0.0f, 0.0f,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    init_pushconst_fastdiv(pc);
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, pc, elements);
}

static vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    switch(type) {
        case GGML_TYPE_Q8_1:
            return ctx->device->pipeline_quantize_q8_1;
        default:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::endl;
            GGML_ABORT("fatal error");
    }
}

static void ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, vk_subbuffer&& in, vk_subbuffer&& out, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, std::array<uint32_t, 1>{ne}, { ne, 1, 1 });
}

static void ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;

    // Check for mmq first
    vk_matmul_pipeline mmp = quantize_y ? ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;

    if (mmp == nullptr) {
        // Fall back to f16 dequant mul mat
        mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
        quantize_y = false;
    }

    const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
    const bool qy_needs_dequant = !quantize_y && ((src1->type != f16_type && !y_f32_kernel) || y_non_contig);

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, f16_type, y_f32_kernel ? GGML_TYPE_F32 : f16_type, (ggml_prec)dst->op_params[0]);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint32_t kpad = quantize_y ? 0 : ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ctx, mmp, ne01, ne11, qx_needs_dequant ? f16_type : src0->type, quantize_y ? GGML_TYPE_Q8_1 : (y_f32_kernel ? GGML_TYPE_F32 : src1->type)));
    const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, quantize_y ? GGML_TYPE_Q8_1 : (y_f32_kernel ? GGML_TYPE_F32 : src1->type));

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    const int x_ne = ne01 * ne00;
    const int y_ne = padded_n * ne10;
    const int d_ne = ne11 * ne01;

    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, pipeline);

    const uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = quantize_y ? (y_ne * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1)) : (y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne);
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    vk_pipeline to_q8_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    if (quantize_y) {
        to_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    if (dryrun) {
        const uint64_t x_sz_upd = x_sz * ne02 * ne03;
        const uint64_t y_sz_upd = y_sz * ne12 * ne13;
        const uint64_t split_k_size = split_k > 1 ? d_sz * ne12 * ne13 * split_k : 0;
        if (
                (qx_needs_dequant && x_sz_upd > ctx->device->max_memory_allocation_size) ||
                (qy_needs_dequant && y_sz_upd > ctx->device->max_memory_allocation_size) ||
                (split_k > 1 && split_k_size > ctx->device->max_memory_allocation_size)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) {
            ctx->prealloc_size_x = x_sz_upd;
        }
        if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz_upd) {
            ctx->prealloc_size_y = y_sz_upd;
        }
        if (split_k > 1 && ctx->prealloc_size_split_k < split_k_size) {
            ctx->prealloc_size_split_k = split_k_size;
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        if (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, to_q8_1, 1);
        }
        if (split_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1);
        }
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    GGML_ASSERT(d_D->size >= d_buf_offset + d_sz * ne02 * ne03);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz * ne02 * ne03);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
    } else if (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_ne * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1));
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz * ne02 * ne03 }, vk_subbuffer{ d_X, 0, x_sz * ne02 * ne03 } }, pc, { (uint32_t)(x_ne * ne02 * ne03), 1, 1});
    }
    if (y_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }
    if (quantize_y) {
        ggml_vk_quantize_q8_1(ctx, subctx, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE }, y_ne * ne12 * ne13);
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 },
        { d_D, d_buf_offset, d_sz * ne12 * ne13 }, { ctx->prealloc_split_k, 0, d_sz * ne12 * ne13 * split_k },
        ne01, ne11, ne10,
        ne10, ne10, ne01, stride_batch_x, stride_batch_y, ne20*ne21,
        split_k, ne12*ne13, ne02, ne12, r2, r3, padded_n
    );  // NOLINT
}

static void ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    bool batch_n = ne11 > 1;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = ne11 * ne10;
    const uint64_t d_ne = ne11 * ne01;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    vk_pipeline dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, src1->type, ne11);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    if (dryrun) {
        const uint64_t x_sz_upd = x_sz * ne02 * ne03;
        const uint64_t y_sz_upd = y_sz * ne12 * ne13;
        if (
                (qx_needs_dequant && x_sz_upd > ctx->device->max_memory_allocation_size) ||
                (qy_needs_dequant && y_sz_upd > ctx->device->max_memory_allocation_size)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) {
            ctx->prealloc_size_x = x_sz_upd;
        }
        if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) {
            ctx->prealloc_size_y = y_sz_upd;
        }

        // Request descriptor sets
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, 1);
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if(!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if(!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(src1->type) * y_ne);
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : ne00*ne01;
    uint32_t stride_batch_y = batch_n ? ne10 : (ne10*ne11);
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    // compute
    const vk_mat_vec_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
        stride_batch_x, stride_batch_y, stride_batch_d,
        (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
    };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                              { vk_subbuffer{ d_X, x_buf_offset, x_sz * ne02 * ne03 }, vk_subbuffer{ d_Y, y_buf_offset, y_sz * ne12 * ne13 }, vk_subbuffer{ d_D, d_buf_offset, d_sz * ne22 * ne23} },
                              pc, { groups_x, (uint32_t)(ne12 * ne13), groups_z });
}

static void ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src1_uma = d_Qy != nullptr;
    }

    const uint64_t x_ne = ne00 * ne01 * ne02;
    const uint64_t y_ne = ne10 * ne11 * ne12;
    const uint64_t d_ne = ne01 * ne11 * ne12;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t d_sz = sizeof(float) * d_ne;

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
    if (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) {
        gqa_ratio = 1;
    }

    if (dryrun) {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1], 1);
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_Qx = src0_buf_ctx->dev_buffer;
    const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    GGML_ASSERT(d_Qx != nullptr);
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }

    const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

    const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

    // compute
    const std::array<uint32_t, 6> pc = { (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12, (uint32_t)(qy_shader_offset / ggml_type_size(src1->type)), (uint32_t)(d_shader_offset / ggml_type_size(dst->type)) };

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    if (gqa_ratio > 1) {
        workgroups_z /= gqa_ratio;
    }

    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1], { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset }, vk_subbuffer{ d_D, d_buffer_offset, d_sz + d_shader_offset } }, pc, { 1, (uint32_t)ne01, workgroups_z });
}

static void ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t nb01 = src0->nb[1];
    const uint64_t nb02 = src0->nb[2];

    const uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    const uint32_t nb03 = (uint32_t)(src0->nb[3] / sizeof(ggml_fp16_t));
    const uint32_t nb13 = (uint32_t)(src1->nb[3] / sizeof(float));
    const uint32_t nb23 = (uint32_t)(dst->nb[3] / sizeof(float));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;

    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src1_uma = d_Qy != nullptr;
    }

    const uint64_t d_ne = ne01 * ne11 * ne12 * ne03;

    const uint32_t row_stride_x = nb01 / sizeof(ggml_fp16_t);
    const uint32_t channel_stride_x = nb02 / sizeof(ggml_fp16_t);
    const uint32_t channel_stride_y = nb12 / sizeof(float);

    const uint64_t qx_sz = ggml_nbytes(src0);
    const uint64_t qy_sz = ggml_nbytes(src1);
    const uint64_t d_sz = sizeof(float) * d_ne;

    if (dryrun) {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_nc_f16_f32, 1);
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_Qx = src0_buf_ctx->dev_buffer;
    const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    GGML_ASSERT(d_Qx != nullptr);
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }

    const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

    const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

    // compute
    const std::array<uint32_t, 12> pc = { (uint32_t)ne00, (uint32_t)ne01, row_stride_x, channel_stride_x, channel_stride_y, (uint32_t)(ne12 / ne02), (uint32_t)ne12, (uint32_t)(qy_shader_offset / ggml_type_size(src1->type)), (uint32_t)(d_shader_offset / ggml_type_size(dst->type)), nb03, nb13, nb23 };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_mul_mat_vec_nc_f16_f32,
        { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz }, vk_subbuffer{ d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset }, vk_subbuffer{ d_D, d_buffer_offset, d_sz + d_shader_offset } }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

static void ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");
    if (src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, src0, src1, dst, dryrun);
    } else if (src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, src0, src1, dst, dryrun);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, src0, src1, dst, dryrun);
    } else {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, dryrun);
    }
}

static void ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];
    GGML_ASSERT(nei0 * nei1 <= 4096);

    const uint32_t nbi1 = ids->nb[1];
    const uint32_t nbi2 = ids->nb[2];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    const uint64_t n_as = ne02;

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;
    ggml_backend_vk_buffer_context * ids_buf_ctx = (ggml_backend_vk_buffer_context *)ids->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;
    vk_buffer d_ids = nullptr;
    size_t ids_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool ids_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        ggml_vk_host_get(ctx->device, ids->data, d_ids, ids_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
        ids_uma = d_ids != nullptr;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    vk_matmul_pipeline mmp = ggml_vk_get_mul_mat_mat_id_pipeline(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);

    const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
    const bool qy_needs_dequant = (src1->type != f16_type && !y_f32_kernel) || y_non_contig;

    if (qx_needs_dequant) {
        // Fall back to dequant + f16 mulmat
        mmp = ggml_vk_get_mul_mat_mat_id_pipeline(ctx, f16_type, y_f32_kernel ? GGML_TYPE_F32 : f16_type, (ggml_prec)dst->op_params[0]);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint32_t kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_id_pipeline_align(ctx, mmp, ne01, nei1, qx_needs_dequant ? f16_type : src0->type));
    const bool aligned = ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_id_pipeline(ctx, mmp, ne01, nei1, aligned, qx_needs_dequant ? f16_type : src0->type);

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) :ne11;
    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = padded_n * ne10;
    const uint64_t d_ne = ne21 * ne20;

    const uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t ids_sz = nbi2;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    if (dryrun) {
        const uint64_t x_sz_upd = x_sz * ne02 * ne03;
        const uint64_t y_sz_upd = y_sz * ne12 * ne13;
        if (
                (qx_needs_dequant && x_sz_upd > ctx->device->max_memory_allocation_size) ||
                (qy_needs_dequant && y_sz_upd > ctx->device->max_memory_allocation_size)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) {
            ctx->prealloc_size_x = x_sz_upd;
        }
        if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) {
            ctx->prealloc_size_y = y_sz_upd;
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (!ids_uma) {
        d_ids = ids_buf_ctx->dev_buffer;
        ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
        GGML_ASSERT(d_ids != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz * ne02 * ne03);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_offset, qx_sz * ne02 * ne03 }, vk_subbuffer{ d_X, 0, x_sz * ne02 * ne03 } }, pc, { (uint32_t)(x_ne * ne02 * ne03), 1, 1});
    }
    if (y_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 },
        { d_D, d_buf_offset, d_sz * ne22 * ne23 }, { d_ids, ids_buf_offset, ids_sz },
        ne01, ne21, ne10, ne10, ne10, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_size(ids->type), ne11, padded_n
    );  // NOLINT
}

static void ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];

    const uint64_t nbi2 = ids->nb[2];

    GGML_ASSERT(nei1 == 1);

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = (ggml_backend_vk_buffer_context *)src1->buffer->context;
    ggml_backend_vk_buffer_context * ids_buf_ctx = (ggml_backend_vk_buffer_context *)ids->buffer->context;

    vk_buffer d_Qx = nullptr;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;
    vk_buffer d_ids = nullptr;
    size_t ids_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool ids_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        ggml_vk_host_get(ctx->device, ids->data, d_ids, ids_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
        ids_uma = d_ids != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = ne11 * ne10;
    const uint64_t d_ne = ne21 * ne20;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t ids_sz = nbi2;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    vk_pipeline dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    if (dryrun) {
        const uint64_t x_sz_upd = x_sz * ne02 * ne03;
        const uint64_t y_sz_upd = y_sz * ne12 * ne13;
        if (
                (qx_needs_dequant && x_sz_upd > ctx->device->max_memory_allocation_size) ||
                (qy_needs_dequant && y_sz_upd > ctx->device->max_memory_allocation_size)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) {
            ctx->prealloc_size_x = x_sz_upd;
        }
        if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) {
            ctx->prealloc_size_y = y_sz_upd;
        }

        // Request descriptor sets
        if (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
        }
        if (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, 1);
        return;
    }

    vk_buffer d_D = dst_buf_ctx->dev_buffer;
    const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if(!src0_uma) {
        d_Qx = src0_buf_ctx->dev_buffer;
        qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if(!src1_uma) {
        d_Qy = src1_buf_ctx->dev_buffer;
        qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if(!ids_uma) {
        d_ids = ids_buf_ctx->dev_buffer;
        ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
        GGML_ASSERT(d_ids != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    if (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(src1->type) * y_ne);
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    // compute
    const vk_mat_vec_id_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
        (uint32_t)x_ne, stride_batch_y, (uint32_t)(ne20*ne21),
        (uint32_t)nei0, (uint32_t)ne11,
    };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
        { vk_subbuffer{ d_X, x_buf_offset, x_sz * ne02 * ne03 },
        vk_subbuffer{ d_Y, y_buf_offset, y_sz * ne12 * ne13 }, vk_subbuffer{ d_D, d_buf_offset, d_sz * ne22 * ne23}, vk_subbuffer{ d_ids, ids_buf_offset, ids_sz } },
        pc, { groups_x, (uint32_t)nei0, groups_z });
}

static void ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    if (src2->ne[1] == 1 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, src0, src1, src2, dst, dryrun);
    } else {
        // Split based on number of ids, to fit in shared memory
        const uint32_t nei0 = (uint32_t)src2->ne[0];
        const uint32_t nei1 = (uint32_t)src2->ne[1];

        GGML_ASSERT(nei0 <= 4096);
        const uint32_t split_size = std::min(nei1, 4096u / nei0);

        ggml_tensor src1_copy = *src1;
        ggml_tensor src2_copy = *src2;
        ggml_tensor dst_copy = *dst;

        for (uint32_t token_start = 0; token_start < nei1; token_start += split_size) {
            const uint32_t n_tokens = std::min(split_size, nei1 - token_start);

            src1_copy.view_offs = src1->view_offs + token_start * src1_copy.nb[2];
            src2_copy.view_offs = src2->view_offs + token_start * src2_copy.nb[1];
            dst_copy.view_offs = dst->view_offs + token_start * dst_copy.nb[2];

            src1_copy.ne[2] = n_tokens;
            src2_copy.ne[1] = n_tokens;
            dst_copy.ne[2] = n_tokens;

            ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, &src1_copy, &src2_copy, &dst_copy, dryrun);
        }
    }
}

static bool ggml_vk_flash_attn_scalar_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv) {
    // Needs to be kept up to date on shader changes
    GGML_UNUSED(hsv);
    const uint32_t wg_size = scalar_flash_attention_workgroup_size;
    const uint32_t Br = get_fa_scalar_num_large_rows(hsv);
    const uint32_t Bc = scalar_flash_attention_Bc;

    const uint32_t tmpsh = wg_size * sizeof(float);
    const uint32_t tmpshv4 = wg_size * 4 * sizeof(float);

    const uint32_t masksh = Bc * Br * sizeof(float);

    const uint32_t Qf = Br * (hsk / 4 + 2) * 4 * sizeof(float);

    const uint32_t total_size = tmpsh + tmpshv4 + masksh + Qf;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv, bool f32acc) {
    // Needs to be kept up to date on shader changes
    GGML_UNUSED(hsv);
    const uint32_t wg_size = scalar_flash_attention_workgroup_size;
    const uint32_t Br = coopmat1_flash_attention_num_large_rows;
    const uint32_t Bc = scalar_flash_attention_Bc;

    const uint32_t acctype = f32acc ? 4 : 2;
    const uint32_t f16vec4 = 8;

    const uint32_t tmpsh = wg_size * sizeof(float);
    const uint32_t tmpshv4 = wg_size * 4 * acctype;

    const uint32_t Qf = Br * (hsk / 4 + 2) * f16vec4;

    const uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    const uint32_t sfsh = Bc * sfshstride * acctype;

    const uint32_t kshstride = hsk / 4 + 2;
    const uint32_t ksh = Bc * kshstride * f16vec4;

    const uint32_t slope = Br * sizeof(float);

    const uint32_t total_size = tmpsh + tmpshv4 + Qf + sfsh + ksh + slope;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask, const ggml_tensor * sinks, ggml_tensor * dst, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    if (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const uint32_t nem1 = mask ? mask->ne[1] : 0;
    const uint32_t nem2 = mask ? mask->ne[2] : 0;
    const uint32_t nem3 = mask ? mask->ne[3] : 0;

    const uint32_t HSK = nek0;
    const uint32_t HSV = nev0;
    uint32_t N = neq1;
    const uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    assert(dst->type == GGML_TYPE_F32);
    assert(q->type == GGML_TYPE_F32);
    assert(k->type == v->type);

    FaCodePath path = ctx->device->coopmat2 ? FA_COOPMAT2 :
                      ctx->device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    if (path == FA_COOPMAT1) {
        const bool coopmat_shape_supported = (dst->op_params[3] == GGML_PREC_F32 && ctx->device->coopmat_support_16x16x16_f32acc) ||
                                             (dst->op_params[3] != GGML_PREC_F32 && ctx->device->coopmat_support_16x16x16_f16acc);

        const bool coopmat_shmem_supported = ggml_vk_flash_attn_coopmat_shmem_support(ctx->device, HSK, HSV, dst->op_params[3] == GGML_PREC_F32);

        if (!coopmat_shape_supported || !coopmat_shmem_supported) {
            path = FA_SCALAR;
        }
    }

    uint32_t gqa_ratio = 1;
    uint32_t qk_ratio = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    uint32_t max_gqa;
    switch (path) {
    case FA_SCALAR:
    case FA_COOPMAT1:
        // We may switch from coopmat1 to scalar, so use the scalar limit for both
        max_gqa = get_fa_scalar_num_large_rows(HSV);
        break;
    case FA_COOPMAT2:
        max_gqa = get_fa_num_small_rows(FA_COOPMAT2);
        break;
    default:
        GGML_ASSERT(0);
    }

    if (N == 1 && qk_ratio > 1 && qk_ratio <= max_gqa &&
        qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= N;
    }

    vk_pipeline *pipelines;
    bool small_rows = N <= get_fa_num_small_rows(path);

    // coopmat1 does not actually support "small rows" (it needs 16 rows).
    // So use scalar instead.
    if (small_rows && path == FA_COOPMAT1) {
        path = FA_SCALAR;
    }

    // scalar is faster than coopmat2 when N==1
    if (N == 1 && path == FA_COOPMAT2) {
        path = FA_SCALAR;
    }

    // with large hsk/hsv, scalar path may need to use small_rows to fit in shared memory
    if (path == FA_SCALAR &&
        !ggml_vk_flash_attn_scalar_shmem_support(ctx->device, HSK, HSV)) {
        small_rows = true;
    }

    bool f32acc = path == FA_SCALAR || dst->op_params[3] == GGML_PREC_F32;

    FaHeadSizes head_sizes = fa_get_head_sizes(k->ne[0], v->ne[0]);

    switch (path) {
    case FA_SCALAR:
        pipelines = &ctx->device->pipeline_flash_attn_f32_f16[k->type][head_sizes][f32acc][small_rows][0];
        break;
    case FA_COOPMAT1:
        pipelines = &ctx->device->pipeline_flash_attn_f32_f16_cm1[k->type][head_sizes][f32acc][small_rows][0];
        break;
    case FA_COOPMAT2:
        pipelines = &ctx->device->pipeline_flash_attn_f32_f16_cm2[k->type][head_sizes][f32acc][small_rows][0];
        break;
    default:
        GGML_ASSERT(0);
    }
    assert(pipelines);

    const uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_size(q->type));
    const uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_size(k->type));
    const uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_size(v->type));

    bool aligned = (KV % pipelines[1]->align) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // mask dim1 is padded to 64, we rely on this to avoid clamping mask loads
    GGML_ASSERT((nem1 % GGML_KQ_MASK_PAD) == 0);

    vk_pipeline pipeline = pipelines[aligned];
    assert(pipeline);

    uint32_t split_kv = KV;
    uint32_t split_k = 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    const uint32_t shader_core_count = ctx->device->shader_core_count ? ctx->device->shader_core_count : 16;

    // Try to use split_k when KV is large enough to be worth the overhead
    if (workgroups_x == 1 && shader_core_count > 0) {
        // Try to run two workgroups per SM.
        split_k = shader_core_count * 2 / (workgroups_y * workgroups_z);
        if (split_k > 1) {
            // Try to evenly split KV into split_k chunks, but it needs to be a multiple
            // of "align", so recompute split_k based on that.
            split_kv = ROUNDUP_POW2(std::max(1u, KV / split_k), pipelines[1]->align);
            split_k = CEIL_DIV(KV, split_kv);
            workgroups_x = split_k;
        }
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    const uint64_t split_k_size = split_k > 1 ? (HSV * ne1 * sizeof(float) + ne1 * sizeof(float) * 2) * split_k * ne3 : 0;
    if (split_k_size > ctx->device->max_memory_allocation_size) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    if (ctx->prealloc_size_split_k < split_k_size) {
        ctx->prealloc_size_split_k = split_k_size;
    }

    if (dryrun) {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (split_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);
        }
        return;
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    const uint32_t n_head_kv   = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_buffer d_Q = nullptr, d_K = nullptr, d_V = nullptr, d_D = nullptr, d_M = nullptr, d_S = nullptr;
    size_t q_buf_offset = 0, k_buf_offset = 0, v_buf_offset = 0, d_buf_offset = 0, m_buf_offset = 0, s_buf_offset = 0;

    bool Q_uma = false, K_uma = false, V_uma = false, D_uma = false, M_uma = false, S_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, q->data, d_Q, q_buf_offset);
        ggml_vk_host_get(ctx->device, k->data, d_K, k_buf_offset);
        ggml_vk_host_get(ctx->device, v->data, d_V, v_buf_offset);
        ggml_vk_host_get(ctx->device, dst->data, d_D, d_buf_offset);
        Q_uma = d_Q != nullptr;
        K_uma = d_K != nullptr;
        V_uma = d_V != nullptr;
        D_uma = d_D != nullptr;
        if (mask) {
            ggml_vk_host_get(ctx->device, mask->data, d_M, m_buf_offset);
            M_uma = d_M != nullptr;
        }
        if (sinks) {
            ggml_vk_host_get(ctx->device, sinks->data, d_S, s_buf_offset);
            S_uma = d_S != nullptr;
        }
    }


    ggml_backend_vk_buffer_context * d_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * q_buf_ctx = (ggml_backend_vk_buffer_context *)q->buffer->context;
    ggml_backend_vk_buffer_context * k_buf_ctx = (ggml_backend_vk_buffer_context *)k->buffer->context;
    ggml_backend_vk_buffer_context * v_buf_ctx = (ggml_backend_vk_buffer_context *)v->buffer->context;

    if (!Q_uma) {
        d_Q = q_buf_ctx->dev_buffer;
        q_buf_offset = vk_tensor_offset(q) + q->view_offs;
    }
    if (!K_uma) {
        d_K = k_buf_ctx->dev_buffer;
        k_buf_offset = vk_tensor_offset(k) + k->view_offs;
    }
    if (!V_uma) {
        d_V = v_buf_ctx->dev_buffer;
        v_buf_offset = vk_tensor_offset(v) + v->view_offs;
    }
    if (!D_uma) {
        d_D = d_buf_ctx->dev_buffer;
        d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    }

    if (!M_uma) {
        d_M = d_Q;
        m_buf_offset = q_buf_offset;
        if (mask) {
            ggml_backend_vk_buffer_context * m_buf_ctx = (ggml_backend_vk_buffer_context*)mask->buffer->context;
            d_M = m_buf_ctx->dev_buffer;
            m_buf_offset = vk_tensor_offset(mask) + mask->view_offs;
        }
    }

    if (!S_uma) {
        d_S = d_Q;
        s_buf_offset = q_buf_offset;
        if (sinks) {
            ggml_backend_vk_buffer_context * s_buf_ctx = (ggml_backend_vk_buffer_context*)sinks->buffer->context;
            d_S = s_buf_ctx->dev_buffer;
            s_buf_offset = vk_tensor_offset(sinks) + sinks->view_offs;
        }
    }

    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | ((mask != nullptr) << 16) | n_head_log2;

    const vk_flash_attn_push_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, (uint32_t)nbk2, (uint32_t)nbk3,
                                              v_stride, (uint32_t)nbv2, (uint32_t)nbv3,
                                              scale, max_bias, logit_softcap,
                                              mask_n_head_log2, m0, m1,
                                              gqa_ratio, split_kv, split_k };

    ggml_vk_sync_buffers(subctx);

    if (split_k > 1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {
                                        vk_subbuffer{d_Q, q_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_K, k_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_V, v_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_M, m_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_S, s_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{ctx->prealloc_split_k, 0, VK_WHOLE_SIZE},
                                    },
                                    // We only use split_k when group query attention is enabled, which means
                                    // there's no more than one tile of rows (i.e. workgroups_x would have been
                                    // one). We reuse workgroups_x to mean the number of splits, so we need to
                                    // cancel out the divide by wg_denoms[0].
                                    pc, { workgroups_x * pipeline->wg_denoms[0], workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(subctx);
        const std::array<uint32_t, 5> pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne3, split_k, (sinks != nullptr) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_split_k_reduce,
                                    {
                                        vk_subbuffer{ctx->prealloc_split_k, 0, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_S, s_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_D, d_buf_offset, VK_WHOLE_SIZE},
                                    },
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)ne3 });
    } else {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {
                                        vk_subbuffer{d_Q, q_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_K, k_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_V, v_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_M, m_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_S, s_buf_offset, VK_WHOLE_SIZE},
                                        vk_subbuffer{d_D, d_buf_offset, VK_WHOLE_SIZE},
                                    },
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }
}

static std::array<uint32_t, 3> ggml_vk_get_conv_elements(const ggml_tensor *dst) {
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    // src0 - kernel:   [KW, KH, Cin, Cout]
    // src1 - input:    [W, H, Cin, N]
    // dst - result:    [OW, OH, Cout, N]

    // Copied from ggml.c: int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d)
    auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    };
    // parallelize in {OW/BS_K, OH/BS_NPQ, 1}
    int64_t W    = src1->ne[0];
    int64_t H    = src1->ne[1];
    int64_t KW   = src0->ne[0];
    int64_t KH   = src0->ne[1];
    int64_t Cout = src0->ne[3];
    int64_t N    = src1->ne[3];
    int64_t OH   = calc_conv_output_size(H, KH, dst->op_params[1], dst->op_params[3], dst->op_params[5]);
    int64_t OW   = calc_conv_output_size(W, KW, dst->op_params[0], dst->op_params[2], dst->op_params[4]);
    int64_t NPQ  = N * OW * OH;

    // Tile output matrix to (K/NB_K, NPQ/NB_NPQ, 1) workgroups
    std::array<uint32_t, 3> elements = { static_cast<uint32_t>(Cout), static_cast<uint32_t>(NPQ), 1 };
    return elements;
}

static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_get_rows_f32[src0->type];
        }
        return nullptr;
    case GGML_OP_ACC:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_acc_f32;
        }
        return nullptr;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            return nullptr;
        }
        switch (op) {
        case GGML_OP_ADD:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_SUB:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_MUL:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_DIV:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        default:
            break;
        }
        return nullptr;
    case GGML_OP_ADD_ID:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add_id_f32;
        }
        return nullptr;
    case GGML_OP_CONCAT:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_concat_f32;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_concat_f16;
        }
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_concat_i32;
        }
        return nullptr;
    case GGML_OP_UPSCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            int mode = ggml_get_op_params_i32(dst, 0);
            switch (mode) {
                case GGML_SCALE_MODE_NEAREST:
                    return ctx->device->pipeline_upscale_nearest_f32;
                case GGML_SCALE_MODE_BILINEAR:
                    return ctx->device->pipeline_upscale_bilinear_f32;
                case GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS:
                    return ctx->device->pipeline_upscale_bilinear_ac_f32;
            }
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_scale_f32;
        }
        return nullptr;
    case GGML_OP_SQR:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sqr_f32;
        }
        return nullptr;
    case GGML_OP_SIN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sin_f32;
        }
        return nullptr;
    case GGML_OP_COS:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_cos_f32;
        }
        return nullptr;
    case GGML_OP_CLAMP:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_clamp_f32;
        }
        return nullptr;
    case GGML_OP_PAD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pad_f32;
        }
        return nullptr;
    case GGML_OP_ROLL:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_roll_f32;
        }
        return nullptr;
    case GGML_OP_REPEAT:
        if (ggml_type_size(src0->type) == sizeof(float) && ggml_type_size(dst->type) == sizeof(float)) {
            return ctx->device->pipeline_repeat_f32;
        }
        return nullptr;
    case GGML_OP_REPEAT_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_repeat_back_f32;
        }
        return nullptr;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        return ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    case GGML_OP_SET_ROWS:
        return ctx->device->pipeline_set_rows[dst->type];
    case GGML_OP_SILU_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_silu_back_f32;
        }
        return nullptr;
    case GGML_OP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_norm_f32;
        }
        return nullptr;
    case GGML_OP_GROUP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_group_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->num_additional_fused_ops > 0 ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rms_norm_back_f32;
        }
        return nullptr;
    case GGML_OP_L2_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_l2_norm_f32;
        }
        return nullptr;
    case GGML_OP_UNARY:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_unary_op(dst)) {
            case GGML_UNARY_OP_SILU:
                return ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU:
                return ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_ERF:
                return ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_QUICK:
                return ctx->device->pipeline_gelu_quick[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_RELU:
                return ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_TANH:
                return ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SIGMOID:
                return ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_GLU:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_glu_op(dst)) {
            case GGML_GLU_OP_GEGLU:
                return ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_REGLU:
                return ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU:
                return ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU_OAI:
                return ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_ERF:
                return ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_QUICK:
                return ctx->device->pipeline_geglu_quick[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_DIAG_MASK_INF:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_diag_mask_inf_f32;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        if (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_soft_max_back_f32;
        }
        return nullptr;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        {
            const int mode = ((const int32_t *) dst->op_params)[2];
            const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

            if (is_neox) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_neox_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f16;
                }
            } else if (is_mrope && !is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_multi_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_multi_f16;
                }
            } else if (is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_vision_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_vision_f16;
                }
            } else {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_norm_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f16;
                }
            }
            return nullptr;
        }
    case GGML_OP_ARGSORT:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_argsort_f32;
        }
        return nullptr;
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sum_rows_f32;
        }
        return nullptr;
    case GGML_OP_ARGMAX:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_argmax_f32;
        }
        return nullptr;
    case GGML_OP_COUNT_EQUAL:
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            return ctx->device->pipeline_count_equal_i32;
        }
        return nullptr;
    case GGML_OP_IM2COL:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_f32_f16;
        }
        return nullptr;
    case GGML_OP_TIMESTEP_EMBEDDING:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_timestep_embedding_f32;
        }
        return nullptr;
    case GGML_OP_CONV_TRANSPOSE_1D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_conv_transpose_1d_f32;
        }
        return nullptr;
    case GGML_OP_POOL_2D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pool2d_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV6:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv6_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV7:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv7_f32;
        }
        return nullptr;
    case GGML_OP_OPT_STEP_ADAMW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_opt_step_adamw_f32;
        }
        return nullptr;
    case GGML_OP_LEAKY_RELU:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_leaky_relu_f32;
        }
        return nullptr;
    case GGML_OP_CONV_2D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            auto elements = ggml_vk_get_conv_elements(dst);
            vk_conv_shapes shape;

            uint32_t tiles[CONV_SHAPE_COUNT];
            for (uint32_t i = 0; i < CONV_SHAPE_COUNT; ++i) {
                tiles[i] = CEIL_DIV(elements[0], ctx->device->pipeline_conv2d_f32[i]->wg_denoms[0]) * CEIL_DIV(elements[1], ctx->device->pipeline_conv2d_f32[i]->wg_denoms[1]);
            }

            // We can't query number of shader cores on Intel, use 32 as a placeholder
            // so small convolutions will still choose a smaller tile.
            const uint32_t shader_core_count = ctx->device->shader_core_count > 0 ? ctx->device->shader_core_count : 32;

            if (elements[0] > 64 && tiles[CONV_SHAPE_128x128] >= shader_core_count * 2) {
                shape = CONV_SHAPE_128x128;
            } else if (elements[0] <= 32 && tiles[CONV_SHAPE_32x256] >= shader_core_count * 2) {
                shape = CONV_SHAPE_32x256;
            } else {
                shape = CONV_SHAPE_64x32;
            }

            if (src0->type == GGML_TYPE_F32) {
                return ctx->device->pipeline_conv2d_f32[shape];
            } else if (src0->type == GGML_TYPE_F16) {
                return ctx->device->pipeline_conv2d_f16_f32[shape];
            }
        }
        return nullptr;
    case GGML_OP_CONV_2D_DW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src1)) {
                return ctx->device->pipeline_conv2d_dw_whcn_f32;
            } else if (ggml_is_contiguous_channels(src1)) {
                return ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        }
        return nullptr;
    default:
        return nullptr;
    }

    GGML_UNUSED(src2);
}

static bool ggml_vk_op_supports_incontiguous(ggml_op op) {
    switch (op) {
    case GGML_OP_CPY:
    case GGML_OP_GET_ROWS:
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_ADD_ID:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SQR:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_ROPE:
    case GGML_OP_RMS_NORM:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_IM2COL:
    case GGML_OP_SET_ROWS:
        return true;
    default:
        return false;
    }
}

static uint32_t get_misalign_bytes(ggml_backend_vk_context * ctx, const ggml_tensor * t)
{
    return ((vk_tensor_offset(t) + t->view_offs) & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1));;
}

template <typename T> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, T &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    GGML_UNUSED(p);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(dst);
    static_assert(!std::is_const<T>::value, "unexpected type");
    GGML_ASSERT(!src0 || get_misalign_bytes(ctx, src0) == 0);
    GGML_ASSERT(!src1 || get_misalign_bytes(ctx, src1) == 0);
    GGML_ASSERT(!src2 || get_misalign_bytes(ctx, src2) == 0);
    GGML_ASSERT(!dst  || get_misalign_bytes(ctx, dst) == 0);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_binary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    GGML_ASSERT(dst->op != GGML_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0));

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
}

template <> void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.a_offset = a_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
}

template<typename PC>
static void ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, ggml_op op, PC&& pc, bool dryrun = false) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ", " << (dryrun ? "dryrun" : "") << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(ggml_vk_op_supports_incontiguous(op) || ggml_vk_dim01_contiguous(src0));  // NOLINT
    GGML_ASSERT(dst->buffer != nullptr);
    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];
    const uint64_t ne0 = ne00 * ne01;

    const bool use_src1 = src1 != nullptr;
    const uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const uint64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const uint64_t ne1 = ne10 * ne11;
    // const uint64_t nb10 = use_src1 ? src1->nb[0] : 0;

    const bool use_src2 = src2 != nullptr;
    const uint64_t ne20 = use_src2 ? src2->ne[0] : 0;
    const uint64_t ne21 = use_src2 ? src2->ne[1] : 0;
    const uint64_t ne22 = use_src2 ? src2->ne[2] : 0;
    const uint64_t ne23 = use_src2 ? src2->ne[3] : 0;
    const uint64_t ne2 = ne20 * ne21;

    const uint64_t ned0 = dst->ne[0];
    const uint64_t ned1 = dst->ne[1];
    const uint64_t ned2 = dst->ne[2];
    const uint64_t ned3 = dst->ne[3];
    const uint64_t ned = ned0 * ned1;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);

    if (pipeline == nullptr) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        if (src1 != nullptr) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::endl;
        GGML_ABORT("fatal error");
    }

    if (dryrun) {
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        return;
    }

    const bool op_supports_incontiguous = ggml_vk_op_supports_incontiguous(op);

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src0_buf_ctx = (ggml_backend_vk_buffer_context *)src0->buffer->context;
    ggml_backend_vk_buffer_context * src1_buf_ctx = use_src1 ? (ggml_backend_vk_buffer_context *)src1->buffer->context : nullptr;
    ggml_backend_vk_buffer_context * src2_buf_ctx = use_src2 ? (ggml_backend_vk_buffer_context *)src2->buffer->context : nullptr;

    vk_buffer d_X = nullptr;
    size_t x_buf_offset = 0;
    vk_buffer d_Y = nullptr;
    size_t y_buf_offset = 0;
    vk_buffer d_Z = nullptr;
    size_t z_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool src2_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_X, x_buf_offset);
        src0_uma = d_X != nullptr;
        if (use_src1) {
            ggml_vk_host_get(ctx->device, src1->data, d_Y, y_buf_offset);
            src1_uma = d_Y != nullptr;
        }
        if (use_src2) {
            ggml_vk_host_get(ctx->device, src2->data, d_Z, z_buf_offset);
            src2_uma = d_Z != nullptr;
        }
    }

    uint64_t x_sz = ggml_type_size(src0->type)/ggml_blck_size(src0->type) * ne0;
    uint64_t y_sz = use_src1 ? ggml_type_size(src1->type) * ne1 : 0;
    uint64_t z_sz = use_src2 ? ggml_type_size(src2->type) * ne2 : 0;
    uint64_t d_sz = ggml_type_size(dst->type) * ned;

    vk_buffer d_D = dst_buf_ctx->dev_buffer;

    // Workaround for tiny tensor inputs on ROPE
    if (op == GGML_OP_ROPE && use_src1 && y_sz > d_D->size) {
        y_sz = VK_WHOLE_SIZE;
    }

    GGML_ASSERT(d_D != nullptr);
    uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
    if(!src0_uma) {
        d_X = src0_buf_ctx->dev_buffer;
        x_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
        GGML_ASSERT(d_X != nullptr);
    }
    if (use_src1 && !src1_uma) {
        d_Y = src1_buf_ctx->dev_buffer;
        y_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
        GGML_ASSERT(d_Y != nullptr);
    }
    if (use_src2 && !src2_uma) {
        d_Z = src2_buf_ctx->dev_buffer;
        z_buf_offset = vk_tensor_offset(src2) + src2->view_offs;
        GGML_ASSERT(d_Z != nullptr);
    }
    // Compute misalignment offset for descriptors and store it in in push constants, then align the descriptor offsets.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, dst);
    x_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
    y_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
    z_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
    d_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);

    if (op_supports_incontiguous) {
        x_sz = ggml_nbytes(src0);
        y_sz = use_src1 ? ggml_nbytes(src1) : 0;
        z_sz = use_src2 ? ggml_nbytes(src2) : 0;
        d_sz = ggml_nbytes(dst);

        if (x_buf_offset + x_sz >= d_X->size) {
            x_sz = VK_WHOLE_SIZE;
        }
        if (use_src1 && y_buf_offset + y_sz >= d_Y->size) {
            y_sz = VK_WHOLE_SIZE;
        }
        if (use_src2 && z_buf_offset + z_sz >= d_Z->size) {
            z_sz = VK_WHOLE_SIZE;
        }
        if (d_buf_offset + d_sz >= d_D->size) {
            d_sz = VK_WHOLE_SIZE;
        }
    }

    std::array<uint32_t, 3> elements;

    // Single call if dimension 2 is contiguous
    GGML_ASSERT(op_supports_incontiguous || (ggml_is_contiguous(src0) && (src1 == nullptr || ggml_is_contiguous(src1))));

    switch (op) {
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_L2_NORM:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_ARGMAX:
        {
            const uint32_t nr = ggml_nrows(src0);
            if (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } else if (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } else {
                elements = { nr, 1, 1 };
            }
        } break;
    case GGML_OP_RMS_NORM:
        elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
        break;

    case GGML_OP_SUM:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        break;
    case GGML_OP_GROUP_NORM:
        {
            const uint32_t num_groups = dst->op_params[0];
            elements = { num_groups * (uint32_t)src0->ne[3], 1, 1 };
        } break;
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        break;
    case GGML_OP_GET_ROWS:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        break;
    case GGML_OP_ARGSORT:
        elements = { (uint32_t)ne00, (uint32_t)ggml_nrows(src0), 1 };
        break;
    case GGML_OP_IM2COL:
        {
            const bool is_2D = dst->op_params[6] == 1;

            const uint32_t IC = src1->ne[is_2D ? 2 : 1];

            const uint32_t KH = is_2D ? src0->ne[1] : 1;
            const uint32_t KW =         src0->ne[0];

            const uint32_t OH = is_2D ? dst->ne[2] : 1;
            const uint32_t OW =         dst->ne[1];

            const uint32_t batch = src1->ne[is_2D ? 3 : 2];

            elements = { OW * KW * KH, OH, batch * IC };
        } break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        {
            const uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } break;
    case GGML_OP_POOL_2D:
        {
            const uint32_t N = dst->ne[3];
            const uint32_t OC = dst->ne[2];
            const uint32_t OH = dst->ne[1];
            const uint32_t OW = dst->ne[0];
            elements = { N * OC * OH * OW, 1, 1};
        } break;
    case GGML_OP_CONV_2D:
        {
            elements = ggml_vk_get_conv_elements(dst);
        } break;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_DIV:
    case GGML_OP_MUL:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_ROLL:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_CPY:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_UNARY:
    case GGML_OP_GLU:
    case GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            if (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_size(src0->type);
                if ((ggml_type_size(src0->type) % 4) == 0) {
                    ne *= ggml_type_size(src0->type) / 4;
                } else {
                    ne *= ggml_type_size(src0->type) / 2;
                }
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            if (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_size(dst->type));
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }
        } break;
    case GGML_OP_ADD_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } break;
    case GGML_OP_SET_ROWS:
        {
            uint32_t ne = ggml_nelements(src0);
            if (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_size(dst->type));
            } else {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }
        }
        break;
    default:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        break;
    }

    if (!op_supports_incontiguous) {
        if (x_sz != VK_WHOLE_SIZE) {
            x_sz *= ne02 * ne03;
        }
        if (use_src1 && y_sz != VK_WHOLE_SIZE) {
            y_sz *= ne12 * ne13;
        }
        if (use_src2 && z_sz != VK_WHOLE_SIZE) {
            z_sz *= ne22 * ne23;
        }
        if (d_sz != VK_WHOLE_SIZE) {
            d_sz *= ned2 * ned3;
        }
    }

    if (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf_y;
        if (use_src1) {
            subbuf_y = { d_Y, y_buf_offset, y_sz };
        } else {
            subbuf_y = { d_X, 0, x_sz };
        }

        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, subbuf_y, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf_y;
        if (use_src1) {
            subbuf_y = { d_Y, y_buf_offset, y_sz };
        } else {
            subbuf_y = { d_X, 0, x_sz };
        }

        vk_subbuffer subbuf_z;
        if (use_src2) {
            subbuf_z = { d_Z, z_buf_offset, z_sz };
        } else {
            subbuf_z = { d_X, 0, x_sz };
        }

        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, subbuf_y, subbuf_z, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        // Empty src2 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf_z;
        if (use_src2) {
            subbuf_z = { d_Z, z_buf_offset, z_sz };
        } else {
            subbuf_z = { d_X, 0, x_sz };
        }

        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, vk_subbuffer{ d_Y, y_buf_offset, y_sz }, subbuf_z, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (op == GGML_OP_IM2COL) {
        // im2col uses only src1 and dst buffers
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_Y, y_buf_offset, y_sz }, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (op == GGML_OP_COUNT_EQUAL) {
        ggml_vk_sync_buffers(subctx);
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_buffer_memset_async(subctx, d_D, d_buf_offset, 0, d_sz);
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, vk_subbuffer{ d_Y, y_buf_offset, y_sz }, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (use_src2) {
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, vk_subbuffer{ d_Y, y_buf_offset, y_sz }, vk_subbuffer{ d_Z, z_buf_offset, z_sz }, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else if (use_src1) {
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, vk_subbuffer{ d_Y, y_buf_offset, y_sz }, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    } else {
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { vk_subbuffer{ d_X, x_buf_offset, x_sz }, vk_subbuffer{ d_D, d_buf_offset, d_sz } }, pc, elements);
    }
}

static void ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_GET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_ACC, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t)nb1, (uint32_t)nb2, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, offset,
    }, dryrun);
}

static void ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_ADD, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t src2_type_size = ggml_type_size(src2->type);

    ggml_vk_op_f32<vk_op_add_id_push_constants>(ctx, subctx, src0, src1, src2, dst, GGML_OP_ADD_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_size,
        (uint32_t)src0->nb[2] / src0_type_size,
        (uint32_t)src1->nb[1] / src1_type_size,
        (uint32_t)src2->nb[1] / src2_type_size,
    }, dryrun);
}

static void ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_op_rwkv_wkv6_push_constants&& pc, int version, bool dryrun = false) {
    GGML_ASSERT(version == 6 || version == 7);
    int num_srcs = version == 6 ? 6 : 7;

    for (int i = 0; i < num_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->buffer != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    if (dryrun) {
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        return;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    ggml_backend_vk_buffer_context * src_buf_ctxs[7] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    for (int i = 0; i < num_srcs; i++) {
        src_buf_ctxs[i] = (ggml_backend_vk_buffer_context *)dst->src[i]->buffer->context;
    }

    ggml_vk_sync_buffers(subctx);

    vk_buffer d_D = nullptr, d_srcs[7] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    size_t dst_offset = 0, src_offsets[7] = { 0, 0, 0, 0, 0, 0, 0 };
    bool dst_uma = false, srcs_uma[7] = { false, false, false, false, false, false, false };

    if (ctx->device->uma) {
        for (int i = 0; i < num_srcs; i++) {
            ggml_vk_host_get(ctx->device, dst->src[i]->data, d_srcs[i], src_offsets[i]);
            srcs_uma[i] = d_srcs[i] != nullptr;
        }

        ggml_vk_host_get(ctx->device, dst->data, d_D, dst_offset);
        dst_uma = d_D != nullptr;
    }

    uint64_t src_sizes[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < num_srcs; i++) {
        src_sizes[i] = ggml_nbytes(dst->src[i]);
        if (!srcs_uma[i]) {
            d_srcs[i] = src_buf_ctxs[i]->dev_buffer;
            src_offsets[i] = vk_tensor_offset(dst->src[i]) + dst->src[i]->view_offs;
        }
    }

    const uint64_t dst_size = ggml_nbytes(dst);
    if (!dst_uma) {
        d_D = dst_buf_ctx->dev_buffer;
        dst_offset = vk_tensor_offset(dst) + dst->view_offs;
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    if (version == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {
            vk_subbuffer{ d_srcs[0], src_offsets[0], src_sizes[0] },
            vk_subbuffer{ d_srcs[1], src_offsets[1], src_sizes[1] },
            vk_subbuffer{ d_srcs[2], src_offsets[2], src_sizes[2] },
            vk_subbuffer{ d_srcs[3], src_offsets[3], src_sizes[3] },
            vk_subbuffer{ d_srcs[4], src_offsets[4], src_sizes[4] },
            vk_subbuffer{ d_srcs[5], src_offsets[5], src_sizes[5] },
            vk_subbuffer{ d_D, dst_offset, dst_size }
        }, pc, elements);
    } else if (version == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {
            vk_subbuffer{ d_srcs[0], src_offsets[0], src_sizes[0] },
            vk_subbuffer{ d_srcs[1], src_offsets[1], src_sizes[1] },
            vk_subbuffer{ d_srcs[2], src_offsets[2], src_sizes[2] },
            vk_subbuffer{ d_srcs[3], src_offsets[3], src_sizes[3] },
            vk_subbuffer{ d_srcs[4], src_offsets[4], src_sizes[4] },
            vk_subbuffer{ d_srcs[5], src_offsets[5], src_sizes[5] },
            vk_subbuffer{ d_srcs[6], src_offsets[6], src_sizes[6] },
            vk_subbuffer{ d_D, dst_offset, dst_size }
        }, pc, elements);
    } else {
        // shouldn't happen
        GGML_ASSERT(false);
    }
}

static void ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, bool dryrun = false) {
    const size_t seq_length = dst->src[0]->ne[2];
    const size_t n_embed = dst->ne[0];
    const size_t n_heads = dst->src[0]->ne[1];
    const size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6,
        dryrun
    );
}

static void ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, bool dryrun = false) {
    const size_t seq_length = dst->src[0]->ne[2];
    const size_t n_embed = dst->ne[0];
    const size_t n_heads = dst->src[0]->ne[1];
    const size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7,
        dryrun
    );
}

static void ggml_vk_op_f32_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_op_push_constants&& pc, bool dryrun = false) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];
    const ggml_tensor * gm = dst->src[2];
    const ggml_tensor * gv = dst->src[3];
    const ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->buffer != nullptr);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_STEP_ADAMW);
    GGML_ASSERT(pipeline != nullptr);

    if (dryrun) {
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        return;
    }

    ggml_backend_vk_buffer_context * x_buf_ctx = (ggml_backend_vk_buffer_context *)x->buffer->context;
    ggml_backend_vk_buffer_context * g_buf_ctx = (ggml_backend_vk_buffer_context *)g->buffer->context;
    ggml_backend_vk_buffer_context * gm_buf_ctx = (ggml_backend_vk_buffer_context *)gm->buffer->context;
    ggml_backend_vk_buffer_context * gv_buf_ctx = (ggml_backend_vk_buffer_context *)gv->buffer->context;
    ggml_backend_vk_buffer_context * p_buf_ctx = (ggml_backend_vk_buffer_context *)p->buffer->context;

    ggml_vk_sync_buffers(subctx);

    vk_buffer d_X = nullptr, d_G = nullptr, d_GM = nullptr, d_GV = nullptr, d_P = nullptr;
    size_t x_offset = 0, g_offset = 0, gm_offset = 0, gv_offset = 0, p_offset = 0;
    bool X_uma = false, G_uma = false, GM_uma = false, GV_uma = false, P_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, x->data, d_X, x_offset);
        ggml_vk_host_get(ctx->device, g->data, d_G, g_offset);
        ggml_vk_host_get(ctx->device, gm->data, d_GM, gm_offset);
        ggml_vk_host_get(ctx->device, gv->data, d_GV, gv_offset);
        ggml_vk_host_get(ctx->device, p->data, d_P, p_offset);

        X_uma = d_X != nullptr;
        G_uma = d_G != nullptr;
        GM_uma = d_GM != nullptr;
        GV_uma = d_GV != nullptr;
        P_uma = d_P != nullptr;
    }

    if (!X_uma) {
        d_X = x_buf_ctx->dev_buffer;
        x_offset = vk_tensor_offset(x) + x->view_offs;
    }
    if (!G_uma) {
        d_G = g_buf_ctx->dev_buffer;
        g_offset = vk_tensor_offset(g) + g->view_offs;
    }
    if (!GM_uma) {
        d_GM = gm_buf_ctx->dev_buffer;
        gm_offset = vk_tensor_offset(gm) + gm->view_offs;
    }
    if (!GV_uma) {
        d_GV = gv_buf_ctx->dev_buffer;
        gv_offset = vk_tensor_offset(gv) + gv->view_offs;
    }
    if (!P_uma) {
        d_P = p_buf_ctx->dev_buffer;
        p_offset = vk_tensor_offset(p) + p->view_offs;
    }

    const uint64_t x_size = ggml_nbytes(x);
    const uint64_t g_size = ggml_nbytes(g);
    const uint64_t gm_size = ggml_nbytes(gm);
    const uint64_t gv_size = ggml_nbytes(gv);
    const uint64_t p_size = ggml_nbytes(p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {
        vk_subbuffer{ d_X, x_offset, x_size },
        vk_subbuffer{ d_G, g_offset, g_size },
        vk_subbuffer{ d_GM, gm_offset, gm_size },
        vk_subbuffer{ d_GV, gv_offset, gv_size },
        vk_subbuffer{ d_P, p_offset, p_size },
    }, pc, elements);
}

static void ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, bool dryrun = false) {
    const size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_step_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f },
        dryrun
    );
}

static void ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    int * op_params = (int *)dst->op_params;

    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_CONCAT, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, op_params[0],
    }, dryrun);
}

static void ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    float sf0 = (float)dst->ne[0] / src0->ne[0];
    float sf1 = (float)dst->ne[1] / src0->ne[1];
    float sf2 = (float)dst->ne[2] / src0->ne[2];
    float sf3 = (float)dst->ne[3] / src0->ne[3];

    if (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = (float)(dst->ne[0] - 1) / (src0->ne[0] - 1);
        sf1 = (float)(dst->ne[1] - 1) / (src0->ne[1] - 1);
    }

    ggml_vk_op_f32<vk_op_upscale_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1],
        (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],(uint32_t)dst->ne[3],
        sf0, sf1, sf2, sf3,
    }, dryrun);
}

static void ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SCALE, std::move(p), dryrun);
}

static void ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SQR, vk_op_unary_push_constants_init(src0, dst), dryrun);
}

static void ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SIN, vk_op_unary_push_constants_init(src0, dst), dryrun);
}

static void ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_COS, vk_op_unary_push_constants_init(src0, dst), dryrun);
}

static void ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_CLAMP, std::move(p), dryrun);
}

static void ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_PAD, std::move(p), dryrun);
}

static void ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t s2 = ggml_get_op_params_i32(dst, 2);
    const int32_t s3 = ggml_get_op_params_i32(dst, 3);
    const uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    const uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    memcpy(&p.param1, &s01_packed, sizeof(float));
    memcpy(&p.param2, &s23_packed, sizeof(float));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_ROLL, std::move(p), dryrun);
}

static void ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_REPEAT, std::move(p), dryrun);
}

static void ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_REPEAT_BACK, std::move(p), dryrun);
}

static void ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    if (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_size(src0->type);
        if ((ggml_type_size(src0->type) % 4) == 0) {
            ne *= ggml_type_size(src0->type) / 4;
        } else {
            ne *= ggml_type_size(src0->type) / 2;
        }
    }

    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_CPY, std::move(p), dryrun);
}

static void ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    if (ggml_is_empty(src0) || ggml_is_empty(src1)) {
        return;
    }

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_SET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_SILU_BACK, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    float * op_params = (float *)dst->op_params;

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_NORM, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f }, dryrun);
}

static void ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    const int * int_op_params = (const int *)dst->op_params;
    const float * float_op_params = (const float *)dst->op_params;

    const uint32_t num_groups = int_op_params[0];
    const float eps = float_op_params[1];
    const uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_GROUP_NORM, { group_size, 0, eps, 0.0f }, dryrun);
}

static void ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, float * op_params, bool dryrun = false) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_RMS_NORM, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        op_params[0], 0.0f, 0,
    }, dryrun);
}

static void ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_RMS_NORM_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f }, dryrun);
}

static void ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_L2_NORM, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f }, dryrun);
}

static void ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_UNARY, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const float * op_params_f = (const float *)dst->op_params;

    const bool swapped = (bool)dst->op_params[1];
    const bool split = src1 != nullptr;
    const float alpha = op_params_f[2];
    const float limit = op_params_f[3];

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (!split) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } else {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    const uint32_t mode = split ? 2 : (swapped ? 1 : 0);

    ggml_vk_op_f32<vk_op_glu_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            limit
        }, dryrun);
}

static void ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_mask_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_DIAG_MASK_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] }, dryrun);
}

static void ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, bool dryrun = false) {
    float * op_params = (float *)dst->op_params;

    float scale = op_params[0];
    float max_bias = op_params[1];

    const uint32_t ncols =   (uint32_t)src0->ne[0];
    const uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    const uint32_t nrows_y = (uint32_t)src0->ne[1];

    const uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    const uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    const uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    const uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    const uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    const uint32_t n_head_kv   = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_vk_op_f32<vk_op_soft_max_push_constants>(ctx, subctx, src0, src1, src2, dst, GGML_OP_SOFT_MAX, {
        ncols,
        src1 != nullptr ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_head_log2,
        nrows_x,
        src2 != nullptr
    }, dryrun);
}

static void ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_SOFT_MAX_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], op_params[1] }, dryrun);
}

static void ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, bool backprop, bool dryrun = false) {
    const int n_dims        = ((int32_t *) dst->op_params)[1];
    const int mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    const float freq_base   = ((float *)   dst->op_params)[5];
    const float freq_scale  = ((float *)   dst->op_params)[6];
    const float ext_factor  = ((float *)   dst->op_params)[7];
    const float attn_factor = ((float *)   dst->op_params)[8];
    const float beta_fast   = ((float *)   dst->op_params)[9];
    const float beta_slow   = ((float *)   dst->op_params)[10];
    int sections[4] {};
    if (mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(sections, (int32_t *) dst->op_params + 11, sizeof(int)*4);
    }

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    uint32_t s1 = src0->nb[1] / ggml_type_size(src0->type);
    uint32_t s2 = src0->nb[2] / ggml_type_size(src0->type);

    ggml_vk_op_f32<vk_op_rope_push_constants>(ctx, subctx, src0, src1, src2, dst, GGML_OP_ROPE, {
        (uint32_t)src0->ne[0], (uint32_t)n_dims, freq_scale, (uint32_t)src0->ne[1],
        freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale,
        src2 != nullptr, (uint32_t)src0->ne[2], s1, s2,
        sections[0], sections[1], sections[2], sections[3], backprop
    }, dryrun);
}

static void ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    int32_t * op_params = (int32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];

    uint32_t ncols_pad = 1;
    while (ncols_pad < ncols) {
        ncols_pad *= 2;
    }

    GGML_ASSERT(ncols_pad <= 1024);

    ggml_vk_op_f32<vk_op_argsort_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_ARGSORT, {
        ncols,
        ncols_pad,
        op_params[0],
    }, dryrun);
}

static void ggml_vk_sum(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SUM, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SUM_ROWS, { (uint32_t)src0->ne[0], 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_count_equal(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_COUNT_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f }, dryrun);
}

static void ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    const int32_t s0 = dst->op_params[0];
    const int32_t s1 = dst->op_params[1];
    const int32_t p0 = dst->op_params[2];
    const int32_t p1 = dst->op_params[3];
    const int32_t d0 = dst->op_params[4];
    const int32_t d1 = dst->op_params[5];

    const bool is_2D = dst->op_params[6] == 1;

    const uint32_t IC = src1->ne[is_2D ? 2 : 1];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t IW =         src1->ne[0];

    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t KW =         src0->ne[0];

    const uint32_t OH = is_2D ? dst->ne[2] : 1;
    const uint32_t OW =         dst->ne[1];

    const uint32_t offset_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const uint32_t batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    const uint32_t pelements = OW * KW * KH;

    ggml_vk_op_f32<vk_op_im2col_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_IM2COL, {
        batch_offset, offset_delta,
        IC, IW, IH, OW, OH, KW, KH,
        pelements,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1,
    }, dryrun);
}

static void ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    const uint32_t dim = dst->op_params[0];
    const uint32_t max_period = dst->op_params[1];
    const uint32_t nb1 = dst->nb[1] / ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    }, dryrun);
}

static void ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    const int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_push_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p), dryrun);
}

static void ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    const int32_t k1 = dst->op_params[1];
    const int32_t k0 = dst->op_params[2];
    const int32_t s1 = dst->op_params[3];
    const int32_t s0 = dst->op_params[4];
    const int32_t p1 = dst->op_params[5];
    const int32_t p0 = dst->op_params[6];

    const uint32_t IH = src0->ne[1];
    const uint32_t IW = src0->ne[0];

    const uint32_t N = dst->ne[3];

    const uint32_t OC = dst->ne[2];
    const uint32_t OH = dst->ne[1];
    const uint32_t OW = dst->ne[0];

    const uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    }, dryrun);
}

static void ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    vk_op_conv2d_push_constants p{};
    p.Cout = static_cast<uint32_t>(ne03);
    p.Cin  = static_cast<uint32_t>(ne02);
    p.N    = static_cast<uint32_t>(ne13);

    p.KW = static_cast<uint32_t>(ne00);
    p.KH = static_cast<uint32_t>(ne01);
    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.s0 = static_cast<uint32_t>(dst->op_params[0]);
    p.s1 = static_cast<uint32_t>(dst->op_params[1]);
    p.p0 = static_cast<uint32_t>(dst->op_params[2]);
    p.p1 = static_cast<uint32_t>(dst->op_params[3]);
    p.d0 = static_cast<uint32_t>(dst->op_params[4]);
    p.d1 = static_cast<uint32_t>(dst->op_params[5]);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    GGML_ASSERT(ne03 == ne2);
    GGML_ASSERT(ne02 == ne12);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_CONV_2D, std::move(p), dryrun);
}

static void ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool dryrun = false) {
    vk_op_conv2d_dw_push_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_CONV_2D_DW, std::move(p), dryrun);
}

static void ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst, bool dryrun = false) {
    const float * op_params = (const float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_LEAKY_RELU, { (uint32_t)ggml_nelements(src0), 0, op_params[0], 0.0f }, dryrun);
}

#ifdef GGML_VULKAN_RUN_TESTS
static void ggml_vk_print_matrix_area(const void * data, ggml_type type, int ne0, int ne1, int i0, int i1, int i2) {
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                float val;
                if (type == GGML_TYPE_F32) {
                    val = *((const float *) data + i2*ne1*ne0 + idx1*ne0 + idx0);
                } else if (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((const ggml_fp16_t *) data + i2*ne1*ne0 + idx1*ne0 + idx0));
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
static void ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, int split_k, int shader_size) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << shader_size << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    vk_pipeline p;
    std::string shname;
    if (shader_size == 0) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_s;
            shname = "F32_ALIGNED_S";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_s;
            shname = "F32_F16_ALIGNED_S";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32.f32acc->a_s;
            shname = "F16_F32_ALIGNED_S";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16.f32acc->a_s;
            shname = "F16_ALIGNED_S";
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (shader_size == 1) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_m;
            shname = "F32_ALIGNED_M";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_m;
            shname = "F32_F16_ALIGNED_M";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32.f32acc->a_m;
            shname = "F16_F32_ALIGNED_M";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16.f32acc->a_m;
            shname = "F16_ALIGNED_M";
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (shader_size == 2) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_l;
            shname = "F32_ALIGNED_L";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_l;
            shname = "F32_F16_ALIGNED_L";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32.f32acc->a_l;
            shname = "F16_F32_ALIGNED_L";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16.f32acc->a_l;
            shname = "F16_ALIGNED_L";
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->s;
                shname = "F32_S";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->s;
                shname = "F32_F16_S";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32.f32acc->s;
                shname = "F16_F32_S";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16.f32acc->s;
                shname = "F16_S";
            }
        } else if (shader_size == 1) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->m;
                shname = "F32_M";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->m;
                shname = "F32_F16_M";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32.f32acc->m;
                shname = "F16_F32_M";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16.f32acc->m;
                shname = "F16_M";
            }
        } else if (shader_size == 2) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->l;
                shname = "F32_L";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->l;
                shname = "F32_F16_L";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32.f32acc->l;
                shname = "F16_F32_L";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16.f32acc->l;
                shname = "F16_L";
            }
        }
    }

    ggml_pipeline_request_descriptor_sets(ctx, p, num_it);
    if (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    }

    if (ctx->device->need_compiles) {
        ggml_vk_load_shaders(ctx->device);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_buffer d_X = ggml_vk_create_buffer_check(ctx->device, sizeof(X_TYPE) * x_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_Y = ggml_vk_create_buffer_check(ctx->device, sizeof(Y_TYPE) * y_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_D = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);

    X_TYPE* x = (X_TYPE *) malloc(sizeof(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(sizeof(Y_TYPE) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        if (std::is_same<float, X_TYPE>()) {
            x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } else if (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = ggml_fp32_to_fp16(1.0f);
            // x[i] = ggml_fp32_to_fp16(i + 1);
            // x[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } else {
            GGML_ABORT("fatal error");
        }
    }
    for (size_t i = 0; i < y_ne; i++) {
        if (std::is_same<float, Y_TYPE>()) {
            y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } else if (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            y[i] = ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = ggml_fp32_to_fp16(i + 1);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_buffer_write(d_X, 0, x, sizeof(X_TYPE) * k * m * batch);
    ggml_vk_buffer_write(d_Y, 0, y, sizeof(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    for (size_t i = 0; i < num_it; i++) {
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), ggml_vk_subbuffer(ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n
        );
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_matmul waitForFences");
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_buffer_read(d_D, 0, d, sizeof(float) * d_ne);

    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    if (std::is_same<float, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } else if (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } else {
        GGML_ABORT("fatal error");
    }
    if (std::is_same<float, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } else if (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } else {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = x;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    double tflops = 2.0*m*n*k*batch*num_it / (time / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    free(d_chk);

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);

    ggml_vk_destroy_buffer(d_X);
    ggml_vk_destroy_buffer(d_Y);
    ggml_vk_destroy_buffer(d_D);

    free(x);
    free(y);
    free(d);
}

static void ggml_vk_print_tensor_area(const ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == GGML_TYPE_F32) {
                    val = *(float *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void ggml_vk_quantize_data(const float * from, void * to, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

static void ggml_vk_dequantize_data(const void * from, float * to, size_t ne, ggml_type quant) {
    if (quant == GGML_TYPE_F32) {
        memcpy(to, from, sizeof(float) * ne);
        return;
    }

    const auto * tt = ggml_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

static void ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    const size_t x_sz = sizeof(float) * ne;
    const size_t x_sz_f16 = sizeof(ggml_fp16_t) * ne;
    const size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
    float * x = (float *) malloc(x_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz_f16, vk::MemoryPropertyFlagBits::eDeviceLocal);
    float * x_ref = (float *) malloc(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(x_sz_f16);

    for (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_data(x, qx, ne, quant);
    ggml_vk_dequantize_data(qx, x_ref, ne, quant);

    ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    if (ctx->device->need_compiles) {
        ggml_vk_load_shaders(ctx->device);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    const std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences");
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    double ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(x_buf, 0, x_chk, x_sz_f16);

    int first_err = -1;

    double avg_err = 0.0;
    for (size_t i = 0; i < ne; i++) {
        double error = std::fabs(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        if (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    ggml_vk_destroy_buffer(x_buf);
    ggml_vk_destroy_buffer(qx_buf);

    free(x);
    free(qx);
    free(x_ref);
    free(x_chk);
}

// This does not work without ggml q8_1 quantization support
//
// typedef uint16_t ggml_half;
// typedef uint32_t ggml_half2;
//
// #define QK8_1 32
// typedef struct {
//     union {
//         struct {
//             ggml_half d; // delta
//             ggml_half s; // d * sum(qs[i])
//         } GGML_COMMON_AGGR_S;
//         ggml_half2 ds;
//     } GGML_COMMON_AGGR_U;
//     int8_t qs[QK8_1]; // quants
// } block_q8_1;
//
// static void ggml_vk_test_quantize(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
//     VK_LOG_DEBUG("ggml_vk_test_quantize(" << ne << ")");
//     GGML_ASSERT(quant == GGML_TYPE_Q8_1);
//
//     const size_t x_sz = sizeof(float) * ne;
//     const size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
//     float * x = (float *) malloc(x_sz);
//     block_q8_1 * qx     = (block_q8_1 *)malloc(qx_sz);
//     block_q8_1 * qx_res = (block_q8_1 *)malloc(qx_sz);
//     vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
//     vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
//
//     for (size_t i = 0; i < ne; i++) {
//         x[i] = rand() / (float)RAND_MAX;
//     }
//
//     vk_pipeline p = ggml_vk_get_quantize_pipeline(ctx, quant);
//
//     ggml_pipeline_request_descriptor_sets(ctx, p, 1);
//
//     if (ctx->device->need_compiles) {
//         ggml_vk_load_shaders(ctx->device);
//     }
//
//     ggml_pipeline_allocate_descriptor_sets(ctx);
//
//     ggml_vk_buffer_write(x_buf, 0, x, x_sz);
//
//     vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
//     ggml_vk_ctx_begin(ctx->device, subctx);
//     ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(x_buf), ggml_vk_subbuffer(qx_buf), ne);
//     ggml_vk_ctx_end(subctx);
//
//     auto begin = std::chrono::high_resolution_clock::now();
//
//     ggml_vk_submit(subctx, ctx->fence);
//     VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_quantize waitForFences");
//     ctx->device->device.resetFences({ ctx->fence });
//     ggml_vk_queue_command_pools_cleanup(ctx->device);
//
//     auto end = std::chrono::high_resolution_clock::now();
//
//     double ms_quant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
//     ggml_vk_buffer_read(qx_buf, 0, qx, qx_sz);
//
//     ggml_vk_quantize_data(x, qx_res, ne, quant);
//
//     int first_err = -1;
//
//     for (size_t i = 0; i < ne / 32; i++) {
//         double error = std::fabs(ggml_fp16_to_fp32(qx_res[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) - ggml_fp16_to_fp32(qx[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         error = std::fabs(ggml_fp16_to_fp32(qx_res[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) - ggml_fp16_to_fp32(qx[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         for (size_t j = 0; j < 32; j++) {
//             uint64_t error = std::abs(qx_res[i].qs[j] - qx[i].qs[j]);
//
//             if (first_err < 0 && error > 1) {
//                 first_err = i;
//             }
//         }
//     }
//
//     std::cerr << "TEST QUANTIZE " << ggml_type_name(quant) << " time=" << ms_quant << "ms " << (first_err == -1 ? "CORRECT" : "INCORRECT") << std::endl;
//
//     if (first_err != -1) {
//         std::cerr << "first_error = " << first_err << std::endl;
//         std::cerr << "Actual result: " << std::endl << std::endl;
//         std::cout << "d=" << ggml_fp16_to_fp32(qx[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) << " s=" << ggml_fp16_to_fp32(qx[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl << std::endl << "Expected result: " << std::endl << std::endl;
//         std::cout << "d=" << ggml_fp16_to_fp32(qx_res[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) << " s=" << ggml_fp16_to_fp32(qx_res[first_err].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx_res[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl;
//     }
//
//     ggml_vk_destroy_buffer(x_buf);
//     ggml_vk_destroy_buffer(qx_buf);
//
//     free(x);
//     free(qx);
//     free(qx_res);
// }

static void ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, size_t split_k, size_t shader_size, ggml_type quant, bool mmq = false) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    vk_matmul_pipeline2 * pipelines;

    if (mmq) {
        pipelines = ctx->device->pipeline_dequant_mul_mat_mat_q8_1;
    } else {
        pipelines = ctx->device->pipeline_dequant_mul_mat_mat;
    }

    const bool fp16acc = ctx->device->fp16;

    vk_pipeline p;
    std::string shname;
    if (shader_size == 0) {
        p = fp16acc ? pipelines[quant].f16acc->a_s : pipelines[quant].f32acc->a_s;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_S";
    } else if (shader_size == 1) {
        p = fp16acc ? pipelines[quant].f16acc->a_m : pipelines[quant].f32acc->a_m;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_M";
    } else if (shader_size == 2) {
        p = fp16acc ? pipelines[quant].f16acc->a_l : pipelines[quant].f32acc->a_l;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = mmq ? 0 : ggml_vk_align_size(k, p->align);

    if (mmq || k != kpad) {
        if (shader_size == 0) {
            p = fp16acc ? pipelines[quant].f16acc->s : pipelines[quant].f32acc->s;
            shname = std::string(ggml_type_name(quant)) + "_S";
        } else if (shader_size == 1) {
            p = fp16acc ? pipelines[quant].f16acc->m : pipelines[quant].f32acc->m;
            shname = std::string(ggml_type_name(quant)) + "_M";
        } else if (shader_size == 2) {
            p = fp16acc ? pipelines[quant].f16acc->l : pipelines[quant].f32acc->l;
            shname = std::string(ggml_type_name(quant)) + "_L";
        } else {
            GGML_ASSERT(0);
        }
    }

    if (p == nullptr) {
        std::cerr << "error: no pipeline for ggml_vk_test_dequant_matmul " << ggml_type_name(quant) << std::endl;
        return;
    }

    const size_t x_sz = sizeof(float) * x_ne;
    const size_t y_sz = sizeof(float) * y_ne;
    const size_t qx_sz = x_ne * ggml_type_size(quant)/ggml_blck_size(quant);
    const size_t qy_sz = mmq ? y_ne * ggml_type_size(GGML_TYPE_Q8_1)/ggml_blck_size(GGML_TYPE_Q8_1) : y_sz;
    const size_t d_sz = sizeof(float) * d_ne;
    float * x = (float *) malloc(x_sz);
    float * y = (float *) malloc(y_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer y_buf = ggml_vk_create_buffer_check(ctx->device, y_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer qy_buf = ggml_vk_create_buffer_check(ctx->device, qy_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_buf = ggml_vk_create_buffer_check(ctx->device, d_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    float * d = (float *) malloc(d_sz);
    float * d_chk = (float *) malloc(d_sz);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    ggml_vk_quantize_data(x, qx, x_ne, quant);

    for (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    ggml_pipeline_request_descriptor_sets(ctx, p, num_it);
    if (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    }
    if (mmq) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_quantize_q8_1, num_it);
    }

    if (ctx->device->need_compiles) {
        ggml_vk_load_shaders(ctx->device);
    }

    ggml_pipeline_allocate_descriptor_sets(ctx);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_buffer_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(ctx->device, subctx);
    if (mmq) {
        for (size_t i = 0; i < num_it; i++) {
            ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n
            );
        }
    } else {
        for (size_t i = 0; i < num_it; i++) {
            ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n
            );
        }
    }
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences");
    ctx->device->device.resetFences({ ctx->fence });
    ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = qx;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    double tflops = 2.0*m*n*k*batch*num_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    if (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    if (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        std::cerr << "src0: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(x, GGML_TYPE_F32, k, m, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "src1: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(y, GGML_TYPE_F32, k, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    ggml_vk_destroy_buffer(qx_buf);
    ggml_vk_destroy_buffer(y_buf);
    ggml_vk_destroy_buffer(qy_buf);
    ggml_vk_destroy_buffer(d_buf);

    free(x);
    free(qx);
    free(y);
    free(d);
    free(d_chk);
}
#endif

static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx) {
#if defined(GGML_VULKAN_RUN_TESTS)
    const std::vector<size_t> vals {
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
    };
    const size_t num_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q4_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q4_0, true);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, GGML_TYPE_Q8_0, true);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, GGML_TYPE_Q8_0, true);

    abort();

    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2);
        std::cerr << '\n' << std::endl;

        if (vals[i + 2] % 32 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_0);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_0);
            std::cerr << '\n' << std::endl;
        }

        if (vals[i + 2] % 256 == 0) {
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n';
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, GGML_TYPE_Q4_K);
            ggml_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, GGML_TYPE_Q4_K);
            std::cerr << '\n' << std::endl;
        }
    }

    GGML_ABORT("fatal error");
#endif

    if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        if (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        if (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
    }
    if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        if (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
}

static bool ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, int tensor_idx, bool use_fence, bool almost_ready);

// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool dryrun, bool last_node, bool almost_ready, bool submit){
    ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_is_empty(node) || !node->buffer) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    ctx->semaphore_idx = 0;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * src2 = node->src[2];
    const ggml_tensor * src3 = node->src[3];

    switch (node->op) {
    // Return on empty ops to avoid generating a compute_ctx and setting exit_tensor
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_NONE:
        return false;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(node)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_GET_ROWS:
    case GGML_OP_ADD:
    case GGML_OP_ADD_ID:
    case GGML_OP_ACC:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_ROLL:
    case GGML_OP_CPY:
    case GGML_OP_SET_ROWS:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_SILU_BACK:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_L2_NORM:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_ARGSORT:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_ARGMAX:
    case GGML_OP_COUNT_EQUAL:
    case GGML_OP_IM2COL:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_CONV_TRANSPOSE_1D:
    case GGML_OP_POOL_2D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_RWKV_WKV7:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_FLASH_ATTN_EXT:
    case GGML_OP_OPT_STEP_ADAMW:
        break;
    default:
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(node->op) << std::endl;
        GGML_ABORT("fatal error");
        return false;
    }

    vk_context compute_ctx;

    if (!dryrun) {
        if (ctx->compute_ctx.expired()) {
            compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
            ctx->compute_ctx = compute_ctx;
            ggml_vk_ctx_begin(ctx->device, compute_ctx);
        } else {
            compute_ctx = ctx->compute_ctx.lock();
        }
    } else {
        switch (node->op) {
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_ACC:
        case GGML_OP_GET_ROWS:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_CONCAT:
        case GGML_OP_UPSCALE:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_PAD:
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_SILU_BACK:
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ARGSORT:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_IM2COL:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_LEAKY_RELU:
            {
                // These operations all go through ggml_vk_op_f32, so short-circuit and
                // do the only thing needed for the dryrun.
                vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, node, node->op);
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                return false;
            }
        default:
            break;
        }
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_ACC:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_ADD:
        ggml_vk_add(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_MUL:
        ggml_vk_mul(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

        break;
    case GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_RMS_NORM:
        if (ctx->num_additional_fused_ops > 0) {
            // fused rms_norm + mul
            ggml_tensor *mul = cgraph->nodes[node_idx + 1];
            ggml_tensor *other_src = mul->src[0] == node ? mul->src[1] : mul->src[0];
            ggml_vk_rms_norm(ctx, compute_ctx, src0, other_src, mul, (float *)node->op_params, dryrun);
        } else {
            ggml_vk_rms_norm(ctx, compute_ctx, src0, src0, node, (float *)node->op_params, dryrun);
        }
        break;
    case GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
            ggml_vk_unary(ctx, compute_ctx, src0, node, dryrun);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(node)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, node, dryrun);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SOFT_MAX:
        ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node, dryrun);

        break;
    case GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, src0, src1, src2, node, false, dryrun);

        break;
    case GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, src0, src1, src2, node, true, dryrun);

        break;
    case GGML_OP_ARGSORT:
        ggml_vk_argsort(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SUM:
        ggml_vk_sum(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_COUNT_EQUAL:
        ggml_vk_count_equal(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_CONV_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, node, dryrun);

        break;
    case GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, src0, src1, node, dryrun);

        break;
    case GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

        break;

    case GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node, dryrun);

        break;

    case GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, node, dryrun);

        break;

    case GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, node, dryrun);

        break;

    case GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, node, dryrun);

        break;
    default:
        return false;
    }

    if (dryrun) {
        return false;
    }

    ctx->tensor_ctxs[node_idx] = compute_ctx;

#if defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_node = true;
#endif

    if (submit || last_node) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        if (last_node) {
            compute_ctx->exit_tensor_idx = node_idx_begin;
        }
        else {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        bool ok = ggml_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, false, almost_ready);
        if (!ok) {
            if (node->op == GGML_OP_UNARY) {
                std::cerr << __func__ << ": error: op not supported UNARY " << node->name << " (" << ggml_unary_op_name(static_cast<ggml_unary_op>(node->op_params[0])) << ")" << std::endl;
            } else if (node->op == GGML_OP_GLU) {
                std::cerr << __func__ << ": error: op not supported GLU " << node->name << " (" << ggml_glu_op_name(static_cast<ggml_glu_op>(node->op_params[0])) << ")" << std::endl;
            } else {
                std::cerr << __func__ << ": error: op not supported " << node->name << " (" << ggml_op_name(node->op) << ")" << std::endl;
            }
        }

    }
    return true;
}

static bool ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, int tensor_idx, bool use_fence = true, bool almost_ready = false) {
    GGML_UNUSED(cgraph);
    ggml_backend_buffer * buf = nullptr;

    switch (tensor->op) {
    case GGML_OP_ADD:
    case GGML_OP_ACC:
    case GGML_OP_GET_ROWS:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_ADD_ID:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_ROLL:
    case GGML_OP_CPY:
    case GGML_OP_SET_ROWS:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_SILU_BACK:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_L2_NORM:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_NONE:
    case GGML_OP_ARGSORT:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_ARGMAX:
    case GGML_OP_COUNT_EQUAL:
    case GGML_OP_IM2COL:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_CONV_TRANSPOSE_1D:
    case GGML_OP_POOL_2D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_RWKV_WKV7:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_OPT_STEP_ADAMW:
        buf = tensor->buffer;

        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
            buf = tensor->buffer;
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(tensor)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            buf = tensor->buffer;
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_FLASH_ATTN_EXT:
        buf = tensor->buffer;

        break;
    default:
        return false;
    }

    if (buf == nullptr) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // always wait for the GPU work to be done for the last submit
    if (tensor_idx == subctx->exit_tensor_idx) {
        use_fence = true;
    }

    // Only run if ctx hasn't been submitted yet
    if (!subctx->seqs.empty()) {
#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
        use_fence = true;
#endif

        // Do staging buffer copies
        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        if (almost_ready && !ctx->almost_ready_fence_pending && !use_fence) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = true;
        } else {
            ggml_vk_submit(subctx, use_fence ? ctx->fence : vk::Fence{});
        }

        if (use_fence) {
            ggml_vk_wait_for_fence(ctx);
        }
#ifdef GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#endif
    }

    if (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
    }

    return true;
}

// Clean up after graph processing is done
static void ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    for (auto& buffer : ctx->gc.temp_buffers) {
        ggml_vk_pool_free(ctx, buffer);
    }
    ctx->gc.temp_buffers.clear();

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);

    for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

// Clean up on backend free
static void ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);

    for (auto& buffer : ctx->buffer_pool) {
        ggml_vk_destroy_buffer(buffer);
    }

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    for (auto& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ctx->transfer_cmd_pool.destroy(ctx->device->device);
}

static int ggml_vk_get_device_count() {
    ggml_vk_instance_init();

    return vk_instance.device_indices.size();
}

static void ggml_vk_get_device_description(int device, char * description, size_t description_size) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", props.deviceName.data());
}

// backend interface

#define UNUSED GGML_UNUSED

// device backend

static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_vk_buffer_type_name;
}

static void ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    ggml_vk_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    return vk_ptr_base;

    UNUSED(buffer);
}

static enum ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

static void ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static void ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));

        return true;
    }
    return false;

    UNUSED(buffer);
}

static void ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ggml_vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

// vk buffer type
static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } catch (const vk::SystemError& e) {
        return nullptr;
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);

    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->suballocation_block_size;
}

static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_num << ")");

    vk_device dev = ggml_vk_get_device(dev_num);

    return &dev->buffer_type;
}

// host buffer type

static const char * ggml_backend_vk_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_VK_NAME "_Host";

    UNUSED(buft);
}

static const char * ggml_backend_vk_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_VK_NAME "_Host";

    UNUSED(buffer);
}

static void ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(vk_instance.devices[0], buffer->context);
}

static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    void * ptr = nullptr;
    try {
        ptr = ggml_vk_host_malloc(vk_instance.devices[0], size);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_vk_host_buffer_free_buffer;

    return buffer;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->suballocation_block_size;

    UNUSED(buft);
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_vk_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    return &ggml_backend_vk_buffer_type_host;
}


// backend

static const char * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return ctx->name.c_str();
}

static void ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return &ctx->device->buffer_type;
}

static void ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_write_async(transfer_ctx, buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static void ggml_backend_vk_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read_async(transfer_ctx, buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    if ((dst->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || dst->buffer->buft == ggml_backend_vk_host_buffer_type()) && ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_context transfer_ctx;

        if (ctx->transfer_ctx.expired()) {
            // Initialize new transfer context
            transfer_ctx = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);
            ctx->transfer_ctx = transfer_ctx;
            ggml_vk_ctx_begin(ctx->device, transfer_ctx);
        } else {
            transfer_ctx = ctx->transfer_ctx.lock();
        }

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy_async(transfer_ctx, dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    if(ctx->transfer_ctx.expired()) {
        return;
    }

    vk_context transfer_ctx = ctx->transfer_ctx.lock();

    ggml_vk_ctx_end(transfer_ctx);

    for (auto& cpy : transfer_ctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ggml_vk_submit(transfer_ctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    for (auto& cpy : transfer_ctx->out_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_ctx.reset();
}

static bool ggml_vk_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

static bool ggml_vk_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }
        // rms_norm shader assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    }
    return true;
}

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsLabelEXT dul = {};
        dul.pLabelName = "ggml_backend_vk_graph_compute";
        dul.color = std::array<float,4>{1.0f, 1.0f, 1.0f, 1.0f};
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(ctx->device->compute_queue.queue, reinterpret_cast<VkDebugUtilsLabelEXT*>(&dul));
    }

    uint64_t total_mat_mul_bytes = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (!ctx->device->disable_fusion && ggml_vk_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ctx->num_additional_fused_ops = 1;
        }
        ggml_vk_build_graph(ctx, cgraph, i, nullptr, 0, true, false, false, false);
        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT || cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
            total_mat_mul_bytes += ggml_nbytes(cgraph->nodes[i]->src[0]);
        } else if (cgraph->nodes[i]->op == GGML_OP_CONV_2D) {
            // Return CRSxNPQxsizeof(*) to account as many bytes as mul_mat has in im2col->mul_mat mode.
            auto CRS_size =
                cgraph->nodes[i]->src[0]->ne[0] * cgraph->nodes[i]->src[0]->ne[1] * cgraph->nodes[i]->src[0]->ne[2];
            auto NPQ_size = cgraph->nodes[i]->ne[0] * cgraph->nodes[i]->ne[1] * cgraph->nodes[i]->ne[3];
            total_mat_mul_bytes += NPQ_size * CRS_size * ggml_type_size(cgraph->nodes[i]->type);
        }
        i += ctx->num_additional_fused_ops;
        ctx->num_additional_fused_ops = 0;
    }
    if (ctx->device->need_compiles) {
        ggml_vk_load_shaders(ctx->device);
    }
    ggml_vk_preallocate_buffers(ctx);
    ggml_pipeline_allocate_descriptor_sets(ctx);

    int last_node = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 && ggml_vk_is_empty(cgraph->nodes[last_node])) {
        last_node -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    bool first_node_in_batch = true; // true if next node will be first node in a batch
    int submit_node_idx = 0; // index to first node in a batch

    vk_context compute_ctx;
    if (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        if (ctx->device->num_queries < cgraph->n_nodes + 1) {
            if (ctx->device->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->device->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.queryCount = cgraph->n_nodes + 100;
            ctx->device->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->device->num_queries = query_create_info.queryCount;
        }

        ctx->device->device.resetQueryPool(ctx->device->query_pool, 0, cgraph->n_nodes+1);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = compute_ctx;
        ggml_vk_ctx_begin(ctx->device, compute_ctx);
        compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->device->query_pool, 0);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of matmul work by looking at the weight matrix size, and submit every 100MB
    // (and scaled down based on model size, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without as much matmul.
    int nodes_per_submit = 100;
    int submitted_nodes = 0;
    int submit_count = 0;
    uint64_t mul_mat_bytes = 0;
    uint64_t mul_mat_bytes_per_submit = std::min(uint64_t(100*1000*1000), total_mat_mul_bytes / 40u);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (first_node_in_batch) {
            submit_node_idx = i;
        }

        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT || cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
            mul_mat_bytes += ggml_nbytes(cgraph->nodes[i]->src[0]);
        }

        if (!ctx->device->disable_fusion && ggml_vk_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ctx->num_additional_fused_ops = 1;
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        bool submit = (submitted_nodes >= nodes_per_submit) ||
                      (mul_mat_bytes >= mul_mat_bytes_per_submit) ||
                      (i + ctx->num_additional_fused_ops == last_node) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_node_idx], submit_node_idx, false, i + ctx->num_additional_fused_ops == last_node, almost_ready, submit);

        if (vk_perf_logger_enabled) {
            if (ctx->compute_ctx.expired()) {
                compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
                ctx->compute_ctx = compute_ctx;
                ggml_vk_ctx_begin(ctx->device, compute_ctx);
            } else {
                compute_ctx = ctx->compute_ctx.lock();
            }
            // If there are fused ops, just write out timestamps for all nodes to keep the accounting simple
            for (int j = 0; j < ctx->num_additional_fused_ops + 1; ++j) {
                compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->device->query_pool, i+j+1);
            }
        }

        if (enqueued) {
            ++submitted_nodes;

#ifndef GGML_VULKAN_CHECK_RESULTS
            if (first_node_in_batch) {
                first_node_in_batch = false;
            }
#endif
        }

        if (submit && enqueued) {
            first_node_in_batch = true;
            submitted_nodes = 0;
            mul_mat_bytes = 0;
            if (submit_count < 3) {
                mul_mat_bytes_per_submit *= 2;
            }
            submit_count++;
        }
        i += ctx->num_additional_fused_ops;
        ctx->num_additional_fused_ops = 0;
    }

    if (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX), "GGML_VULKAN_PERF waitForFences");
        ctx->device->device.resetFences({ ctx->device->fence });

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->device->query_pool, 0, cgraph->n_nodes + 1, (cgraph->n_nodes + 1)*sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "get timestamp results");
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (!ggml_vk_is_empty(cgraph->nodes[i])) {
                ctx->device->perf_logger->log_timing(cgraph->nodes[i], uint64_t((timestamps[i+1] - timestamps[i]) * ctx->device->properties.limits.timestampPeriod));
            }
        }

        ctx->device->perf_logger->print_timings();
    }

    ggml_vk_graph_cleanup(ctx);

    return GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ NULL,  // ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ NULL,  // ggml_backend_vk_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ NULL,  // ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_vk_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_num << ")");

    ggml_backend_vk_context * ctx = new ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_num);

    ggml_backend_t vk_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_num),
        /* .context = */ ctx,
    };

    return vk_backend;
}

bool ggml_backend_is_vk(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

int ggml_backend_vk_get_device_count() {
    return ggml_vk_get_device_count();
}

void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());
    int dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_size);
}

void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];

    vk::PhysicalDeviceMemoryProperties memprops = vkdev.getMemoryProperties();

    for (const vk::MemoryHeap& heap : memprops.memoryHeaps) {
        if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            *total = heap.size;
            *free = heap.size;
            break;
        }
    }
}

//////////////////////////

struct ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_vk_device_get_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_memory(ctx->device, free, total);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_vk_host_buffer_type();
}

static enum ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    ggml_backend_vk_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_init(ctx->device);
}

static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                const vk_device& device = ggml_vk_get_device(ctx->device);
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    if (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        return false;
                    }
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
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
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        break;
                    default:
                        return false;
                }
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                if (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    return false;
                }
                if (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    return false;
                }

                return true;
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                bool coopmat2 = device->coopmat2;
                FaHeadSizes head_sizes = fa_get_head_sizes(op->src[1]->ne[0], op->src[2]->ne[0]);
                if (head_sizes == FA_HEAD_SIZE_UNSUPPORTED) {
                    return false;
                }
                if (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    return false;
                }
                // It's straightforward to support different K/V dequant, but would
                // significantly increase the number of pipelines
                if (op->src[1]->type != op->src[2]->type) {
                    return false;
                }
                switch (op->src[1]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    // supported in scalar and coopmat2 paths
                    break;
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                // K dequants currently disabled because D dimension is rounded up to 256 and runs inefficiently
                //case GGML_TYPE_Q2_K:
                //case GGML_TYPE_Q3_K:
                //case GGML_TYPE_Q4_K:
                //case GGML_TYPE_Q5_K:
                //case GGML_TYPE_Q6_K:
                //case GGML_TYPE_IQ1_S:
                //case GGML_TYPE_IQ1_M:
                //case GGML_TYPE_IQ2_XXS:
                //case GGML_TYPE_IQ2_XS:
                //case GGML_TYPE_IQ2_S:
                //case GGML_TYPE_IQ3_XXS:
                //case GGML_TYPE_IQ3_S:
                //case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ4_NL:
                    // currently supported only in coopmat2 path
                    if (!coopmat2) {
                        return false;
                    }
                    break;
                default:
                    return false;
                }
                if (!coopmat2 && !device->subgroup_shuffle) {
                    // scalar FA uses subgroupShuffle
                    return false;
                }
                return true;
            }
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_SET_ROWS:
            {
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;

                if (src0_type == GGML_TYPE_F32) {
                    switch (src1_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }
                if (src1_type == GGML_TYPE_F32) {
                    switch (src0_type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }

                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }

                // We can handle copying from a type to the same type if it's
                // contiguous (memcpy). We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                if (src0_type == src1_type &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                    (ggml_type_size(src0_type) % 2) == 0) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REPEAT:
            return ggml_type_size(op->type) == sizeof(float) && ggml_type_size(op->src[0]->type) == sizeof(float);
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_SQR:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_UPSCALE:
        case GGML_OP_ACC:
        case GGML_OP_CONCAT:
        case GGML_OP_SCALE:
        case GGML_OP_PAD:
        case GGML_OP_ROLL:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ARGSORT:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_IM2COL:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_POOL_2D:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_OPT_STEP_ADAMW:
            return true;
        case GGML_OP_CONV_TRANSPOSE_1D:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
            {
                // Op is disabled for Apple because it segfaults at pipeline create time on MoltenVK
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                const vk_device& device = ggml_vk_get_device(ctx->device);
                bool is_Apple = ggml_vk_get_device(ctx->device)->vendor_id == VK_VENDOR_ID_APPLE;
                // Channel-contiguous format is not supported yet.
                return ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op)) && !is_Apple;
            }
        default:
            return false;
    }

    UNUSED(dev);
}

static bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return false;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return buft_ctx->device->idx == ctx->device;
}

static bool ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_memory,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_VK_NAME;
}

static size_t ggml_backend_vk_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_vk_get_device_count();
}

static ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int i = 0; i < ggml_backend_vk_get_device_count(); i++) {
                ggml_backend_vk_device_context * ctx = new ggml_backend_vk_device_context;
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                devices.push_back(new ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    try {
        ggml_vk_instance_init();
        return &reg;
    } catch (const vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        return nullptr;
    }
}

// Extension availability
static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef GGML_VULKAN_VALIDATE
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}

// Extension availability
static bool ggml_vk_instance_debug_utils_ext_available(
    const std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    for (const auto & properties : instance_extensions) {
        if (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            return true;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::endl;
    return false;

    UNUSED(instance_extensions);
}

static bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    switch (props.vendorID) {
    case VK_VENDOR_ID_INTEL:
        // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant performance boost,
        // while some older hardware (ex. Arc A770) has performance regressions
        return arch == vk_device_architecture::INTEL_XE2;
    case VK_VENDOR_ID_AMD:
        if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            return arch == vk_device_architecture::AMD_RDNA3;
        }
        return true;
    default:
        return true;
    }
}

// checks

#ifdef GGML_VULKAN_CHECK_RESULTS
static void ggml_vk_print_graph_origin(const ggml_tensor * tensor, std::vector<const ggml_tensor *>& done, int level = 0) {
    if (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        return;
    }
    for (int j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

static void ggml_vk_print_tensor_area(const ggml_tensor * tensor, const void * data, int i0, int i1, int i2, int i3) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == GGML_TYPE_F32) {
                    val = *(const float *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(const ggml_fp16_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else if (tensor->type == GGML_TYPE_I32) {
                    val = *(const int32_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void ggml_vk_print_tensor(const ggml_tensor * tensor, const char * name) {
    void * tensor_data = tensor->data;

    const bool is_gpu = tensor->buffer != nullptr && ggml_backend_buffer_is_vk(tensor->buffer);

    if (is_gpu) {
        const size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer buffer_gpu = buf_ctx->dev_buffer;
        ggml_vk_buffer_read(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    if (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    if (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<const ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    if (is_gpu) {
        free(tensor_data);
    }
}

void * comp_result;
size_t comp_size;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
static void ggml_vk_check_results_0(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx];
    if (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        return;
    }

    bool fused_rms_norm_mul = false;
    int rms_norm_idx = -1;
    if (ctx->num_additional_fused_ops == 1 &&
        tensor->op == GGML_OP_RMS_NORM &&
        cgraph->nodes[tensor_idx + 1]->op == GGML_OP_MUL) {
        fused_rms_norm_mul = true;
        tensor = cgraph->nodes[tensor_idx + 1];
    }

    check_counter++;
    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ggml_ctx = ggml_init(iparams);

    std::array<struct ggml_tensor *, 6> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    std::array<size_t, 6> src_size = {0, 0, 0, 0, 0, 0};
    std::array<void *, 6> src_buffer = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    const char * srci_name[6] = {"src0", "src1", "src2", "src3", "src4", "src5"};

    struct ggml_tensor * tensor_clone = nullptr;

    for (int i = 0; i < 6; i++) {
        ggml_tensor * srci = tensor->src[i];
        if (fused_rms_norm_mul) {
            rms_norm_idx = tensor->src[0]->op == GGML_OP_RMS_NORM ? 0 : 1;
            ggml_tensor *rms_norm = tensor->src[rms_norm_idx];
            switch (i) {
            case 0: srci = rms_norm->src[0]; break;
            case 1: srci = tensor->src[1 - rms_norm_idx]; break;
            default: continue;
            }
        }
        if (srci == nullptr) {
            continue;
        }
        ggml_tensor * srci_clone = ggml_dup_tensor(ggml_ctx, srci);
        size_t srci_size = ggml_nbytes(srci);

        src_clone[i] = srci_clone;
        src_size[i] = ggml_nbytes(srci);
        src_buffer[i] = malloc(srci_size);

        srci_clone->data = src_buffer[i];
        if (ggml_backend_buffer_is_host(srci->buffer)) {
            memcpy(srci_clone->data, srci->data, srci_size);
            memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (ggml_backend_buffer_is_vk(srci->buffer)) {
            ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)srci->buffer->context;
            vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
            uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
            if (!ggml_is_contiguous(srci) && ggml_vk_dim01_contiguous(srci)) {
                for (int i3 = 0; i3 < srci->ne[3]; i3++) {
                    for (int i2 = 0; i2 < srci->ne[2]; i2++) {
                        const int idx = i3*srci->ne[2] + i2;
                        ggml_vk_buffer_read(buffer_gpu, offset + idx * srci->nb[2], ((char *)srci_clone->data + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                    }
                }

                srci_clone->nb[0] = srci->nb[0];
                srci_clone->nb[1] = srci->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                }
            } else {
                if (offset + srci_size >= buffer_gpu->size) {
                    srci_size = buffer_gpu->size - offset;
                }
                ggml_vk_buffer_read(buffer_gpu, offset, srci_clone->data, srci_size);
                memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ABORT("fatal error");
        }

        if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
            ggml_vk_print_tensor(srci, srci_name[i]);
        }
    }

    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        const float * params = (const float *)tensor->op_params;
        tensor_clone = ggml_flash_attn_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
        if (src_clone[4]) {
            ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
        }
    } else if (tensor->op == GGML_OP_MUL_MAT) {
        tensor_clone = ggml_mul_mat(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_MUL_MAT_ID) {
        tensor_clone = ggml_mul_mat_id(ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
    } else if (tensor->op == GGML_OP_SUB) {
        tensor_clone = ggml_sub(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_MUL) {
        if (fused_rms_norm_mul) {
            tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(float *)tensor->src[rms_norm_idx]->op_params);
            tensor_clone = ggml_mul(ggml_ctx, tensor_clone, src_clone[1 - rms_norm_idx]);
        } else {
            tensor_clone = ggml_mul(ggml_ctx, src_clone[0], src_clone[1]);
        }
    } else if (tensor->op == GGML_OP_DIV) {
        tensor_clone = ggml_div(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_CONCAT) {
        tensor_clone = ggml_concat(ggml_ctx, src_clone[0], src_clone[1], *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_UPSCALE) {
        tensor_clone = ggml_upscale_ext(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
    } else if (tensor->op == GGML_OP_SCALE) {
        const float * params = (const float *)tensor->op_params;
        tensor_clone = ggml_scale_bias(ggml_ctx, src_clone[0], params[0], params[1]);
    } else if (tensor->op == GGML_OP_SQR) {
        tensor_clone = ggml_sqr(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_SIN) {
        tensor_clone = ggml_sin(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_COS) {
        tensor_clone = ggml_cos(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_CLAMP) {
        const float * params = (const float *)tensor->op_params;
        tensor_clone = ggml_clamp(ggml_ctx, src_clone[0], params[0], params[1]);
    } else if (tensor->op == GGML_OP_PAD) {
        tensor_clone = ggml_pad(ggml_ctx, src_clone[0], tensor->ne[0] - src_clone[0]->ne[0], tensor->ne[1] - src_clone[0]->ne[1], tensor->ne[2] - src_clone[0]->ne[2], tensor->ne[3] - src_clone[0]->ne[3]);
    } else if (tensor->op == GGML_OP_REPEAT) {
        tensor_clone = ggml_repeat(ggml_ctx, src_clone[0], tensor);
    } else if (tensor->op == GGML_OP_REPEAT_BACK) {
        tensor_clone = ggml_repeat_back(ggml_ctx, src_clone[0], tensor);
    } else if (tensor->op == GGML_OP_ADD) {
        tensor_clone = ggml_add(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_ACC) {
        tensor_clone = ggml_acc(ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
    } else if (tensor->op == GGML_OP_NORM) {
        tensor_clone = ggml_norm(ggml_ctx, src_clone[0], *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_GROUP_NORM) {
        const float * float_params = (const float *)tensor->op_params;
        tensor_clone = ggml_group_norm(ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
    } else if (tensor->op == GGML_OP_RMS_NORM) {
        tensor_clone = ggml_rms_norm(ggml_ctx, src_clone[0], *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_RMS_NORM_BACK) {
        const float eps = ((float *) tensor->op_params)[0];
        tensor_clone = ggml_rms_norm_back(ggml_ctx, src_clone[0], src_clone[1], eps);
    } else if (tensor->op == GGML_OP_SILU_BACK) {
        tensor_clone = ggml_silu_back(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_L2_NORM) {
        const float eps = ((float *) tensor->op_params)[0];
        tensor_clone = ggml_l2_norm(ggml_ctx, src_clone[0], eps);
    } else if (tensor->op == GGML_OP_SOFT_MAX) {
        if (src1 != nullptr) {
            const float * params = (const float *)tensor->op_params;
            tensor_clone = ggml_soft_max_ext(ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
        } else {
            tensor_clone = ggml_soft_max(ggml_ctx, src_clone[0]);
        }
    } else if (tensor->op == GGML_OP_SOFT_MAX_BACK) {
        tensor_clone = ggml_soft_max_ext_back(ggml_ctx, src_clone[0], src_clone[1], ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_DIAG_MASK_INF) {
        tensor_clone = ggml_diag_mask_inf(ggml_ctx, src_clone[0], tensor->op_params[0]);
    } else if (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
        const int n_dims      = ((int32_t *) tensor->op_params)[1];
        const int mode        = ((int32_t *) tensor->op_params)[2];
        //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
        const int n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
        const float freq_base       = ((float *) tensor->op_params)[5];
        const float freq_scale      = ((float *) tensor->op_params)[6];
        const float ext_factor      = ((float *) tensor->op_params)[7];
        const float attn_factor     = ((float *) tensor->op_params)[8];
        const float beta_fast       = ((float *) tensor->op_params)[9];
        const float beta_slow       = ((float *) tensor->op_params)[10];
        if (mode & GGML_ROPE_TYPE_MROPE) {
            int32_t *sections = ((int32_t *) tensor->op_params) + 11;
            if (tensor->op == GGML_OP_ROPE) {
                tensor_clone = ggml_rope_multi(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                tensor_clone = ggml_rope_multi_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
        } else {
            if (tensor->op == GGML_OP_ROPE) {
                tensor_clone = ggml_rope_ext(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                tensor_clone = ggml_rope_ext_back(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
        }
    } else if (tensor->op == GGML_OP_UNARY) {
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_SILU:
            tensor_clone = ggml_silu(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_GELU:
            tensor_clone = ggml_gelu(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_GELU_ERF:
            tensor_clone = ggml_gelu_erf(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            tensor_clone = ggml_gelu_quick(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_RELU:
            tensor_clone = ggml_relu(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_TANH:
            tensor_clone = ggml_tanh(ggml_ctx, src_clone[0]);
            break;
        case GGML_UNARY_OP_SIGMOID:
            tensor_clone = ggml_sigmoid(ggml_ctx, src_clone[0]);
            break;
        default:
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
    } else if (tensor->op == GGML_OP_GLU) {
        if (src_clone[1] == nullptr) {
            tensor_clone = ggml_glu(ggml_ctx, src_clone[0], (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
        } else {
            tensor_clone = ggml_glu_split(ggml_ctx, src_clone[0], src_clone[1], (ggml_glu_op) tensor->op_params[0]);
        }
    } else if (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
        if (src1 == nullptr) {
            tensor_clone = ggml_dup(ggml_ctx, src_clone[0]);
            tensor_clone->type = tensor->type;
        } else {
            tensor_clone = ggml_cpy(ggml_ctx, src_clone[0], src_clone[1]);
        }
    } else if (tensor->op == GGML_OP_CONT) {
        tensor_clone = ggml_cont_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_RESHAPE) {
        tensor_clone = ggml_reshape_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_VIEW) {
        tensor_clone = ggml_view_4d(ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_PERMUTE) {
        int32_t * params = (int32_t *)tensor->op_params;
        tensor_clone = ggml_permute(ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
    } else if (tensor->op == GGML_OP_TRANSPOSE) {
        tensor_clone = ggml_transpose(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_GET_ROWS) {
        tensor_clone = ggml_get_rows(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_ARGSORT) {
        tensor_clone = ggml_argsort(ggml_ctx, src_clone[0], (ggml_sort_order) *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SUM) {
        tensor_clone = ggml_sum(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_SUM_ROWS) {
        tensor_clone = ggml_sum_rows(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_ARGMAX) {
        tensor_clone = ggml_argmax(ggml_ctx, src_clone[0]);
    } else if (tensor->op == GGML_OP_COUNT_EQUAL) {
        tensor_clone = ggml_count_equal(ggml_ctx, src_clone[0], src_clone[1]);
    } else if (tensor->op == GGML_OP_IM2COL) {
        const int32_t s0 = tensor->op_params[0];
        const int32_t s1 = tensor->op_params[1];
        const int32_t p0 = tensor->op_params[2];
        const int32_t p1 = tensor->op_params[3];
        const int32_t d0 = tensor->op_params[4];
        const int32_t d1 = tensor->op_params[5];

        const bool is_2D = tensor->op_params[6] == 1;
        tensor_clone = ggml_im2col(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
    } else if (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
        const int32_t dim = tensor->op_params[0];
        const int32_t max_period = tensor->op_params[1];
        tensor_clone = ggml_timestep_embedding(ggml_ctx, src_clone[0], dim, max_period);
    } else if (tensor->op == GGML_OP_CONV_TRANSPOSE_1D){
        const int32_t s0 = tensor->op_params[0];
        const int32_t p0 = tensor->op_params[1];
        const int32_t d0 = tensor->op_params[2];
        tensor_clone = ggml_conv_transpose_1d(ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
    } else if (tensor->op == GGML_OP_POOL_2D) {
        enum ggml_op_pool op = static_cast<ggml_op_pool>(tensor->op_params[0]);
        const int32_t k0 = tensor->op_params[1];
        const int32_t k1 = tensor->op_params[2];
        const int32_t s0 = tensor->op_params[3];
        const int32_t s1 = tensor->op_params[4];
        const int32_t p0 = tensor->op_params[5];
        const int32_t p1 = tensor->op_params[6];

        tensor_clone = ggml_pool_2d(ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
    } else if (tensor->op == GGML_OP_CONV_2D) {
        const int32_t s0 = tensor->op_params[0];
        const int32_t s1 = tensor->op_params[1];
        const int32_t p0 = tensor->op_params[2];
        const int32_t p1 = tensor->op_params[3];
        const int32_t d0 = tensor->op_params[4];
        const int32_t d1 = tensor->op_params[5];
        tensor_clone = ggml_conv_2d(ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
    } else if (tensor->op == GGML_OP_LEAKY_RELU) {
        const float * op_params = (const float *)tensor->op_params;
        tensor_clone = ggml_leaky_relu(ggml_ctx, src_clone[0], op_params[0], false);
    } else if (tensor->op == GGML_OP_RWKV_WKV6) {
        tensor_clone = ggml_rwkv_wkv6(ggml_ctx, src_clone[0], src_clone[1],
        src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
    } else if (tensor->op == GGML_OP_RWKV_WKV7) {
        tensor_clone = ggml_rwkv_wkv7(ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
        src_clone[4], src_clone[5], src_clone[6]);
    } else if (tensor->op == GGML_OP_OPT_STEP_ADAMW) {
        src_clone[0]->flags = src0->flags;
        tensor_clone = ggml_opt_step_adamw(ggml_ctx, src_clone[0], src_clone[1],
        src_clone[2], src_clone[3], src_clone[4]);
    }
    else {
        std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_cgraph * cgraph_cpu = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph_cpu, 8);

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_size = ggml_nbytes(tensor_clone);

    comp_result = malloc(comp_size);
    memcpy(comp_result, tensor_clone->data, comp_size);
    memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * GGML_MAX_DIMS);

    for (int i = 0; i < 6; i++) {
        if (src_buffer[i] != nullptr) {
            free(src_buffer[i]);
        }
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

static void ggml_vk_check_results_1(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int tensor_idx) {
    ggml_tensor * tensor = cgraph->nodes[tensor_idx];
    if (tensor->op == GGML_OP_TRANSPOSE || tensor->op == GGML_OP_SET_ROWS) {
        return;
    }
    bool fused_rms_norm_mul = false;
    if (ctx->num_additional_fused_ops == 1 &&
        tensor->op == GGML_OP_RMS_NORM &&
        cgraph->nodes[tensor_idx + 1]->op == GGML_OP_MUL) {
        fused_rms_norm_mul = true;
        tensor = cgraph->nodes[tensor_idx + 1];
    }

    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    void * tensor_data = tensor->data;

    if (ggml_backend_buffer_is_vk(tensor->buffer)) {
        size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        if (offset + tensor_size >= buffer_gpu->size) {
            tensor_size = buffer_gpu->size - offset;
        }

        ggml_vk_buffer_read(buffer_gpu, offset, tensor_data, tensor_size);
    }

    float first_error_result = -1.0f;
    float first_error_correct = -1.0f;
    std::array<int, 4> first_error = { -1, -1, -1, -1 };
    double avg_err = 0.0;
    size_t counter = 0;

    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    const bool buffer_size_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_size;
                    float correct = 0.0f;
                    float result = 0.0f;

                    if (buffer_size_fit) {
                        if (tensor->type == GGML_TYPE_F32) {
                            correct = *(float *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(float *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else if (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } else if (tensor->type == GGML_TYPE_BF16) {
                            correct = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_bf16_to_fp32(*(ggml_bf16_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } else if (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int32_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else if (tensor->type == GGML_TYPE_I64) {
                            correct = *(int64_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int64_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::endl;
                        }
                    } else {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::endl;
                        GGML_ABORT("fatal error");
                    }

                    if ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !buffer_size_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        if (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        if (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        if (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        if (src3 != nullptr) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<const ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    const double denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    if (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_result = result;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    if (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) {
                        avg_err += std::fabs(correct - result) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        if (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<const ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    if (avg_err > 0.5 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        if (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << ggml_op_name(src3->op) << " type=" << ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<const ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } else {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    free(comp_result);
    comp_result = nullptr;
    comp_size = 0;

    if (ggml_backend_buffer_is_vk(tensor->buffer)) {
        free(tensor_data);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)
