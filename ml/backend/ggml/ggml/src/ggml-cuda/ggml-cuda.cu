#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_reset_device(int device) {
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaDeviceReset());
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            CUDA_CHECK(cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

static std::string ggml_cuda_parse_uuid(cudaDeviceProp prop, int device_num) {
    char id[64];

#if !defined(GGML_USE_HIP)
    snprintf(id, sizeof(id),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)prop.uuid.bytes[0],
        (unsigned char)prop.uuid.bytes[1],
        (unsigned char)prop.uuid.bytes[2],
        (unsigned char)prop.uuid.bytes[3],
        (unsigned char)prop.uuid.bytes[4],
        (unsigned char)prop.uuid.bytes[5],
        (unsigned char)prop.uuid.bytes[6],
        (unsigned char)prop.uuid.bytes[7],
        (unsigned char)prop.uuid.bytes[8],
        (unsigned char)prop.uuid.bytes[9],
        (unsigned char)prop.uuid.bytes[10],
        (unsigned char)prop.uuid.bytes[11],
        (unsigned char)prop.uuid.bytes[12],
        (unsigned char)prop.uuid.bytes[13],
        (unsigned char)prop.uuid.bytes[14],
        (unsigned char)prop.uuid.bytes[15]
        );
#else
#ifdef _WIN32
        snprintf(id, sizeof(id), "%d", device_num);
#else
    try {
        std::string uuid = std::string(prop.uuid.bytes, 16);

        size_t pos = 0;
        unsigned long long v = stoull(uuid, &pos, 16);
        if (v == 0 || pos != uuid.size() || (!uuid.empty() && uuid[0] == '-'))
            throw std::invalid_argument("invalid uuid");

        snprintf(id, sizeof(id), "GPU-%016llx", v);
    } catch (const std::exception &e) {
        snprintf(id, sizeof(id), "%d", device_num);
    }
#endif
#endif

    return id;
}

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    no\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    GGML_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if defined(GGML_USE_HIP)
        if (std::getenv("GGML_CUDA_INIT") != NULL) {
            GGML_LOG_INFO("%s: initializing rocBLAS on device %d\n", __func__, id);
            CUDA_CHECK(cudaSetDevice(id));
            // rocblas_initialize will SIGABRT if the GPU isn't supported
            rocblas_initialize();
            GGML_LOG_INFO("%s: rocBLAS initialized on device %d\n", __func__, id);
        }
#endif

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;
#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, ID: %s\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize, ggml_cuda_parse_uuid(prop, id).c_str());
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, ID: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                        ggml_cuda_parse_uuid(prop, id).c_str());
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#ifdef __CUDA_ARCH_LIST__
        if (std::getenv("GGML_CUDA_INIT") != NULL) {
            GGML_ASSERT(ggml_cuda_has_arch(info.devices[id].cc) && "ggml was not compiled with support for this arch");
        }
#endif // defined(__CUDA_ARCH_LIST__)
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, ID: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                        ggml_cuda_parse_uuid(prop, id).c_str());
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

#define CUDA_ALIGNMENT 128

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;
    bool allocate = true;
    size_t last_alloc = 0;

    explicit ggml_cuda_pool_leg(int device, bool alloc) :
        device(device),
        allocate(alloc) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                if (allocate) {
                    CUDA_CHECK(cudaFree(b.ptr));
                }
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        if (allocate) {
            ggml_cuda_set_device(device);
            if (ggml_cuda_device_malloc(&ptr, look_ahead_size, device) != cudaSuccess) {
                    last_alloc = look_ahead_size;
                    throw std::bad_alloc();
            }
        } else {
            ptr = (void *)CUDA_ALIGNMENT;
        }
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        if (allocate) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaFree(ptr));
        }
        pool_size -= size;
    }

    bool alloc_memory() override {
        return allocate;
    }

    size_t alloc_size() override {
        return pool_size + last_alloc;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    bool allocate = true;
    size_t last_alloc = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device, bool alloc) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity),
        allocate(alloc) {
        if (!allocate) {
            pool_addr = (CUdeviceptr)CUDA_ALIGNMENT;
        }
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0 && allocate) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            if (allocate) {
                // allocate more physical memory
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = device;
                CUmemGenericAllocationHandle handle;
                if (cuMemCreate(&handle, reserve_size, &prop, 0) != CUDA_SUCCESS) {
                    last_alloc = reserve_size;
                    throw std::bad_alloc();
                }

                // reserve virtual address space (if not already reserved)
                if (pool_addr == 0) {
                    CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
                }

                // map at the end of the pool
                CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
                if (cuMemMap(start_ptr, reserve_size, 0, handle, 0) != CUDA_SUCCESS) {
                    last_alloc = reserve_size;
                    CU_CHECK(cuMemRelease(handle));
                    throw std::bad_alloc();
                }

                // the memory allocation handle is no longer needed after mapping
                CU_CHECK(cuMemRelease(handle));

                // set access
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                if (cuMemSetAccess((CUdeviceptr)((char *)(pool_addr) + pool_size), reserve_size, &access, 1) != CUDA_SUCCESS) {
                    CU_CHECK(cuMemUnmap(start_ptr, reserve_size));
                    last_alloc = reserve_size;
                    throw std::bad_alloc();
                }

    #if defined(GGML_USE_HIP)
                mappings.push_back({start_ptr, reserve_size});
    #endif
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }

    bool alloc_memory() override {
        return allocate;
    }

    size_t alloc_size() override {
        return pool_size + last_alloc;
    }

};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no,
                                                                               bool alloc) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device, alloc));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device, alloc));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
    delete buffer;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return CUDA_ALIGNMENT;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_noalloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    void * dev_ptr = (void *)ggml_backend_cuda_buffer_type_get_alignment(buft);
    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, {}, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
    /* .noalloc_buffer   = */ ggml_backend_cuda_buffer_type_noalloc_buffer,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

static int64_t get_row_rounding(const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_cuda_info().devices[id].cc;
        row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t * row_low, int64_t * row_high, const ggml_tensor * tensor, const std::array<float, GGML_CUDA_MAX_DEVICES> & tensor_split, int id) {
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows*tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_cuda_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows*tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_cuda_split_buffer_type_context {
    int main_device;
    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    std::string name;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    CUDA_CHECK(cudaFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};


static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
    delete buffer;
}

static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_cuda_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if cudaMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            CUDA_CHECK(cudaMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
            CUDA_CHECK(cudaEventCreateWithFlags(&extra->events[id][is], cudaEventDisableTiming));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_split_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *)buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];
    const size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *)data + offset_split;
        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

static const char * ggml_backend_cuda_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_split_buffer_type_context * ctx = (ggml_backend_cuda_split_buffer_type_context *)buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_CUDA_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_CUDA_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_cuda_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({main_device, tensor_split_arr});
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_cuda_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_CUDA_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
    delete buffer;
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int cc = ggml_cuda_info().devices[id].cc;

    const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);

    const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT;

    if (supports_bf16 && src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), ne, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    } else if (fast_fp16_hardware_available(cc) && use_fp16) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

            const half alpha_f16 = 1.0f;
            const half beta_f16 = 0.0f;

            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f16, src0_ptr,      CUDA_R_16F, ne00,
                                    src1_ptr,      CUDA_R_16F, ne10,
                        &beta_f16,  dst_f16.get(), CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
        }
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED_VARS(dst, src1_ddq_i, src1_padded_row_size);
}

static void ggml_cuda_set_peer_access(const int n_tokens, int main_device) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_CUDA_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        ggml_cuda_set_device(id);

        for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
            if (id == id_other) {
                continue;
            }
            if (id != main_device && id_other != main_device) {
                continue;
            }

            int can_access_peer;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (can_access_peer) {
                if (enable_peer_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
                    if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                } else {
                    cudaError_t err = cudaDeviceDisablePeerAccess(id_other);
                    if (err != cudaErrorPeerAccessNotEnabled) {
                        CUDA_CHECK(err);
                    } else {
                        // reset the error
                        (void)cudaGetLastError();
                    }
                }
            }
        }
    }

    ggml_cuda_set_device(main_device);
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;

    GGML_UNUSED(main_device);
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int64_t i02_divisor = ne12 / ne02;
    const int64_t i03_divisor = ne13 / ne03;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));
    GGML_ASSERT(!(split && ne03 < ne13));

    ggml_tensor_extra_gpu * src0_extra = split ? (ggml_tensor_extra_gpu *) src0->extra : nullptr;


    std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(tensor_split);

            if (id != 0) {
                dev[id].row_low  = ne01*tensor_split[id];
                if (dev[id].row_low < ne01) {
                    dev[id].row_low -= dev[id].row_low % rounding;
                }
            }

            if (id != ggml_backend_cuda_get_device_count() - 1) {
                dev[id].row_high  = ne01*tensor_split[id + 1];
                if (dev[id].row_high < ne01) {
                    dev[id].row_high -= dev[id].row_high % rounding;
                }
            }
        }
    }

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = split ? (char *) src0_extra->data_device[id] : (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer.
            // This buffer needs to be cleared entirely because multiple regions will function as padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            const size_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data, 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && src1_is_contiguous) {
                quantize_src1(
                    dev[id].src1_ddf, nullptr, dev[id].src1_ddq, src0->type, ne10,
                    nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float),
                    src1_padded_col_size, ne11, ne12, ne13, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[id].row_high - dev[id].row_low)*ne1 : ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventRecord(src0_extra->events[ctx.device][0], ctx.stream()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_CUDA_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if ((!split && id != ctx.device) || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            // wait for main GPU data if necessary
            if (split && (id != ctx.device || is != 0)) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, src0_extra->events[ctx.device][0], 0));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                const size_t nbytes_src0_matrix = ne01*ne00*src0_ts / src0_bs;
                char  *  src0_dd_i =  dev[id].src0_dd + ((i03/i03_divisor)*ne02 + (i02/i02_divisor)) * nbytes_src0_matrix;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous) {
                    quantize_src1(
                        src1_ddf_i, nullptr, src1_ddq_i, src0->type, ne10, ne10, ne11*ne10, ne12*ne11*ne10,
                        src1_padded_col_size, src1_ncols, 1, 1, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i03 % i03_divisor == 0 && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                        src0_dd_i, src0, i03/i03_divisor, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + dev[id].row_low;
                        CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(
                            dhf_dst_i, ctx.device, ne0*sizeof(float), dst_dd_i, id, row_diff*sizeof(float), row_diff*sizeof(float), src1_ncols, stream));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != ctx.device || is != 0)) {
                    CUDA_CHECK(cudaEventRecord(src0_extra->events[id][is], stream));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_backend_cuda_get_device_count() > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_CUDA_MAX_STREAMS ? is_max : GGML_CUDA_MAX_STREAMS;

        ggml_cuda_set_device(ctx.device);
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (dev[id].row_low == dev[id].row_high) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), src0_extra->events[id][is], 0));
            }
        }
    }
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        cu_compute_type = CUBLAS_COMPUTE_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        const int64_t smb = ne12 == 1 ? s13       : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (glu->src[0] != ffn_gate_bias || glu->src[1] != ffn_up_bias) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up || ffn_up_bias->src[1] == ffn_up;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate || ffn_gate_bias->src[1] == ffn_gate;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up || ffn_gate_bias->src[0] != ffn_gate) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    } else {
        if (glu->src[0] != ffn_gate && glu->src[1] != ffn_up) {
            return false;
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (ffn_up->src[2] && (ffn_up->src[2] != ffn_gate->src[2])) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 3> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    const bool split = ggml_backend_buft_is_cuda_split(ffn_up->src[0]->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(ffn_gate->src[0]->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft) ||
                       ggml_backend_buft_is_cuda_split(src1->buffer->buft);

    //TODO: add support for fusion for split buffers
    if (split) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16 = false;

    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc            = ggml_cuda_info().devices[id].cc;
            const int warp_size     = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
            use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        }
    } else {
        const int cc            = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size     = ggml_cuda_info().devices[ctx.device].warp_size;
        use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    //TODO update for generic tensor parallelism
    const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;

    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_f) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_vec_q) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_q) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec_f) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buft_is_cuda_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (ne2 == 1) {
            if (ggml_is_quantized(src0->type)) {
                ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
            } else {
                ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
            }
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH
static bool check_node_graph_compatibility(ggml_cgraph * cgraph,
    int batch_size, bool use_cuda_graph) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";
    const std::string nemotron_h_block_out_prefix = "nemotron_h_block_out";
    const std::string mamba2_y_add_d_prefix = "mamba2_y_add_d";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
        }

        if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] != 1) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
        }

        // If we have an explicit batch size hint then we don't need to use the tensor name heuristics
        if (batch_size >= 0) {
            if (batch_size > 1) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%d]\n", __func__, batch_size);
#endif
            }
        } else {
            if (node->op == GGML_OP_ADD &&
                node->src[1] && node->src[1]->ne[1] > 1 &&
                (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
                (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
                strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
                strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
                strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0 &&
                strncmp(node->name, nemotron_h_block_out_prefix.c_str(), nemotron_h_block_out_prefix.size()) != 0 &&
                strncmp(node->name, mamba2_y_add_d_prefix.c_str(), mamba2_y_add_d_prefix.size()) != 0) {
                // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
                // by means of matching node names. See
                // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
                // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
                // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_VIEW
        ) {
            return false;
        }
    }

    if ((node->op == GGML_OP_SCALE || node->op == GGML_OP_GLU) &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static bool is_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {

    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->instance == nullptr) {
        cuda_graph_update_required = true;
    }

    // Check if the graph size has changed
    if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        cuda_graph_update_required = true;
        cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            cuda_graph_update_required = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
    }

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_backend_cuda_context * cuda_ctx) {

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
        cuda_ctx->cuda_graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops, std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    //TODO: remove special case once ggml_can_fuse can handle empty nodes
    std::initializer_list<enum ggml_op> topk_moe_ops =
        ggml_cuda_topk_moe_ops(/*with_norm*/ false, /*delayed_softmax=*/false);
    std::initializer_list<enum ggml_op> topk_moe_ops_with_norm =
        ggml_cuda_topk_moe_ops(/*with_norm=*/true, /*delayed_softmax=*/false);
    std::initializer_list<enum ggml_op> topk_moe_ops_delayed_softmax =
        ggml_cuda_topk_moe_ops(/*with_norm=*/false, /*delayed_softmax=*/true);

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    if (is_equal(topk_moe_ops_with_norm, ops) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3, node_idx + 9 })) {
        ggml_tensor * softmax = cgraph->nodes[node_idx];
        ggml_tensor * weights = cgraph->nodes[node_idx + 9];

        if (ggml_cuda_should_use_topk_moe(softmax, weights)) {
            return true;
        }
    }

    if (is_equal(topk_moe_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3, node_idx + 4 })) {
        ggml_tensor * softmax = cgraph->nodes[node_idx];
        ggml_tensor * weights = cgraph->nodes[node_idx + 4];
        if (ggml_cuda_should_use_topk_moe(softmax, weights)) {
            return true;
        }
    }

    if (is_equal(topk_moe_ops_delayed_softmax, ops) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 1, node_idx + 5 })) {
        ggml_tensor * softmax = cgraph->nodes[node_idx + 4];
        ggml_tensor * weights = cgraph->nodes[node_idx + 5];

        if (ggml_cuda_should_use_topk_moe(softmax, weights)) {
            return true;
        }
    }

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            return true;
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            return true;
        }
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            return true;
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contigous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {

    // flag used to determine whether it is an integrated_gpu
    const bool integrated = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }
            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

#ifdef GGML_CUDA_DEBUG
                const int nodes_fused = i - prev_i - 1;
                if (nodes_fused > 0) {
                    GGML_LOG_INFO("nodes_fused: %d\n", nodes_fused);
                }
#endif
                prev_i = i;

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

                // When reserving, we are forcing CUDA graphs but this operation is not graph-safe so we need to skip it
                if (reserving_graph && node->op == GGML_OP_MUL_MAT_ID && node->ne[2] != 1) {
                    continue;
                }

                // start of fusion operations
                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion) {

                    if (ggml_cuda_can_fuse(cgraph, i, ggml_cuda_topk_moe_ops(/*with norm*/ true), {})) {
                        ggml_tensor * weights          = cgraph->nodes[i + 9];
                        ggml_tensor * selected_experts = cgraph->nodes[i + 3];
                        ggml_tensor * clamp            = cgraph->nodes[i + 7];
                        ggml_cuda_op_topk_moe(*cuda_ctx, node->src[0], weights, selected_experts, /*with norm*/ true,
                                              /*delayed softmax*/ false, clamp);
                        i += 9;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, ggml_cuda_topk_moe_ops(/*with norm*/ false), {})) {
                        ggml_tensor * weights          = cgraph->nodes[i + 4];
                        ggml_tensor * selected_experts = cgraph->nodes[i + 3];
                        ggml_cuda_op_topk_moe(*cuda_ctx, node->src[0], weights, selected_experts, /*with norm*/ false,
                                              /*delayed softmax*/ false);
                        i += 4;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i,
                                           ggml_cuda_topk_moe_ops(/*with norm*/ false, /*delayed softmax*/ true), {})) {
                        ggml_tensor * weights = cgraph->nodes[i + 5];
                        ggml_tensor * ids     = cgraph->nodes[i + 1];

                        ggml_cuda_op_topk_moe(*cuda_ctx, node->src[0], weights, ids, /*with norm*/ false,
                                              /*delayed_softmax*/ true);
                        i += 5;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                        ggml_tensor * rope = cgraph->nodes[i];
                        ggml_tensor * set_rows = cgraph->nodes[i + 2];

                        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
                        i += 2;
                        continue;
                    }

                    if (node->op == GGML_OP_ADD) {
                        int n_fuse = 0;
                        ggml_op ops[8];
                        std::fill(ops, ops + 8, GGML_OP_ADD);

                        for (; n_fuse <= 6; ++n_fuse){
                            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                                break;
                            }
                            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                                break;
                            }
                            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                                break;
                            }
                        }

                        n_fuse++;

                        if (n_fuse > 1) {
                            for (int j = 0; j < n_fuse - 1; ++j) {
                                node->src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
                            }
                            cgraph->nodes[i + n_fuse - 1]->data = node->data;
                            ggml_cuda_op_fused_add(*cuda_ctx, node, n_fuse);
                            i += n_fuse - 1;

                            continue;
                        }
                    }

                    bool fused_mul_mat_vec = false;
                    int fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu         = cgraph->nodes[i + 4];
                            ggml_tensor * gate_bias_n = glu->src[0];
                            ggml_tensor * up_bias_n   = glu->src[1];

                            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
                            ggml_tensor * gate_n      = nullptr;
                            ggml_tensor * up_n        = nullptr;

                            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                                gate_n = cgraph->nodes[i];
                                up_n   = cgraph->nodes[i + 2];
                            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                                gate_n = cgraph->nodes[i + 2];
                                up_n   = cgraph->nodes[i];
                            } else {
                                continue;
                            }

                            auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) {
                                if (op_bias == GGML_OP_ADD) {
                                    if (bias_node->src[0] == mul_node) {
                                        return bias_node->src[1];
                                    }
                                    if (bias_node->src[1] == mul_node) {
                                        return bias_node->src[0];
                                    }
                                    return (ggml_tensor *) nullptr;
                                }
                                GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
                                GGML_ASSERT(bias_node->src[0] == mul_node);
                                return bias_node->src[1];
                            };

                            ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
                            ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

                            if (!up_bias_tensor || !gate_bias_tensor) {
                                continue;
                            }

                            // we don't support repeating adds
                            if (bias_op == GGML_OP_ADD &&
                                (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                 !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                                continue;
                            }

                            const ggml_tensor * src0 = up_n->src[0];
                            const ggml_tensor * src1 = up_n->src[1];
                            const ggml_tensor * ids  = up_n->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate      = gate_n->src[0];
                                fusion_data.x_bias    = up_bias_tensor;
                                fusion_data.gate_bias = gate_bias_tensor;
                                fusion_data.glu_op    = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 5;
                                break;
                            }
                        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
                            ggml_tensor * glu  = cgraph->nodes[i + 2];
                            ggml_tensor * gate = glu->src[0];
                            ggml_tensor * up   = glu->src[1];

                            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1])
                                || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

                            if (!ok) continue;

                            const ggml_tensor * src0 = up->src[0];
                            const ggml_tensor * src1 = up->src[1];
                            const ggml_tensor * ids  = up->src[2];

                            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }

                            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                                ggml_cuda_mm_fusion_args_host fusion_data{};
                                fusion_data.gate   = gate->src[0];
                                fusion_data.glu_op = ggml_get_glu_op(glu);

                                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                                fused_mul_mat_vec = true;
                                fused_node_count = 3;
                                break;
                            }
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    fused_mul_mat_vec = false;
                    fused_node_count = 0;

                    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
                        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

                        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                            continue;
                        }

                        ggml_tensor * mm_node   = cgraph->nodes[i];
                        ggml_tensor * bias_node = cgraph->nodes[i + 1];

                        ggml_tensor * bias_tensor = nullptr;
                        if (bias_op == GGML_OP_ADD) {
                            if (bias_node->src[0] == mm_node) {
                                bias_tensor = bias_node->src[1];
                            } else if (bias_node->src[1] == mm_node) {
                                bias_tensor = bias_node->src[0];
                            } else {
                                continue;
                            }
                        } else {
                            if (bias_node->src[0] != mm_node) {
                                continue;
                            }
                            bias_tensor = bias_node->src[1];
                        }

                        const ggml_tensor * src0 = mm_node->src[0];
                        const ggml_tensor * src1 = mm_node->src[1];
                        const ggml_tensor * ids  = mm_node->src[2];

                        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
                            continue;
                        }

                        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
                            continue;
                        }

                        ggml_cuda_mm_fusion_args_host fusion_data{};
                        fusion_data.x_bias = bias_tensor;

                        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
                            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }

                        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
                            fused_mul_mat_vec = true;
                            fused_node_count = 2;
                            break;
                        }
                    }

                    if (fused_mul_mat_vec) {
                        i += fused_node_count - 1;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
                        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                        i += 2;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
                        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i+1]);
                        i++;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                        i += 2;
                        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i], node);
                        continue;
                    }
                }
#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft) || (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            update_cuda_graph_executable(cuda_ctx);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph, int batch_size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    cuda_ctx->pool_set_alloc(true);

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);

        use_cuda_graph = check_node_graph_compatibility(cgraph, batch_size, use_cuda_graph);

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

// This is used to skip operations that are not graph safe during the reservation process.
bool reserving_graph = false;

static enum ggml_status ggml_backend_cuda_graph_reserve(ggml_backend_t backend, ggml_cgraph * cgraph, bool alloc) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    cuda_ctx->pool_set_alloc(alloc);

    #ifdef USE_CUDA_GRAPH
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }
    #endif

    ggml_cuda_set_device(cuda_ctx->device);

    {
        std::lock_guard<std::mutex> lock(ggml_cuda_lock);
        ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
    }

    reserving_graph = true;

    // Create CuBLAS handles early to avoid synchronous allocations during graph capture.
    cuda_ctx->cublas_handle();

    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));

    enum ggml_status result = GGML_STATUS_SUCCESS;

    try {
        bool use_cuda_graph = false;
        bool cuda_graph_update_required = false;
        bool graph_evaluated_or_captured = false;

        evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);
    } catch (const std::exception &e) {
        result = GGML_STATUS_FAILED;
    }

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph));
    CUDA_CHECK(cudaGraphDestroy(graph));

    reserving_graph = false;

    {
        std::lock_guard<std::mutex> lock(ggml_cuda_lock);
        if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
            ggml_cuda_lock_cv.notify_all();
        }
    }

    return result;
}

static size_t ggml_backend_cuda_buffer_size(ggml_backend_t backend) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    size_t allocs = 0;
    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; i++) {
        allocs += ctx->pool_get_alloc_size(i);
    }
    return allocs;
}

static void ggml_backend_cuda_reset(ggml_backend_t backend) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *)backend->context;
    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; i++) {
        ctx->pools[ctx->device][i] = NULL;
    }
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static bool enable_graph_optimization = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    GGML_ASSERT(ggml_backend_cuda_get_device_count() == 1 && "compute graph optimization is only supported on single GPU in the CUDA backend");
    GGML_LOG_DEBUG("Optimizing CUDA graph %p with %d nodes\n", cgraph->nodes, cgraph->n_nodes);

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
    /* .graph_reserve           = */ ggml_backend_cuda_graph_reserve,
    /* .buffer_size             = */ ggml_backend_cuda_buffer_size,
    /* .reset                   = */ ggml_backend_cuda_reset,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    std::string id;
    int major;
    int minor;
    int driver_major;
    int driver_minor;
    int integrated;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static const char * ggml_backend_cuda_device_get_id(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->id.c_str();
}

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);

#if defined(GGML_USE_HIP)
    if (ggml_hip_mgmt_init() == 0) {
        int status = ggml_hip_get_device_memory(ctx->pci_bus_id.c_str(), free, total, ctx->integrated != 0);
        if (status == 0) {
            GGML_LOG_DEBUG("%s device %s utilizing AMD specific memory reporting free: %zu total: %zu\n", __func__, ctx->pci_bus_id.c_str(), *free, *total);
            ggml_hip_mgmt_release();
            return;
        }
        ggml_hip_mgmt_release();
    }
#else
    if (ggml_nvml_init() == 0) {
        int status = ggml_nvml_get_device_memory(ctx->id.c_str(), free, total);
        if (status == 0) {
            GGML_LOG_DEBUG("%s device %s utilizing NVML memory reporting free: %zu total: %zu\n", __func__, ctx->id.c_str(), *free, *total);
            ggml_nvml_release();
            return;
        }
        ggml_nvml_release();
    }
#endif
    CUDA_CHECK(cudaMemGetInfo(free, total));

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx->device));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__)

}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

#define GGML_HIP_NAME "HIP"
static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->id          = ggml_backend_cuda_device_get_id(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();

    // Memory reporting is disabled to avoid allocation of a CUDA primary context (~300 MB per device).
    // If you need the memory data, call ggml_backend_dev_memory() explicitly.
    props->memory_total = props->memory_free = 0;

#if defined(GGML_USE_HIP)
    int cc = ggml_cuda_info().devices[ctx->device].cc - GGML_CUDA_CC_OFFSET_AMD;
    props->compute_major = cc / 0x100;
    props->compute_minor = cc - (props->compute_major * 0x100);
#else
    props->compute_major = ctx->major;
    props->compute_minor = ctx->minor;
#endif
    props->driver_major = ctx->driver_major;
    props->driver_minor = ctx->driver_minor;
    props->integrated = ctx->integrated;
    props->library = GGML_CUDA_NAME;

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->buffer && ggml_backend_buft_is_cuda_split(a->buffer->buft)) {
                    if (a->ne[2] > 1 || a->ne[3] > 1) {
                        return false;
                    }
                    // for small weight matrices the active device can end up without any rows, don't use row split in those cases
                    // this avoids some edge cases (and the performance would not be good anyways)
                    ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) a->buffer->buft->context;
                    int64_t row_low;
                    int64_t row_high;
                    get_row_split(&row_low, &row_high, a, buft_ctx->tensor_split, dev_ctx->device);
                    if (row_low == row_high) {
                        return false;
                    }
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT && b->ne[2] * b->ne[3] > 1024) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SSM_SCAN: {
            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_2D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_PAD:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (((ggml_backend_buft_is_cuda(buft) || ggml_backend_buft_is_cuda_split(buft)) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft)));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static void ggml_backend_cuda_device_reset(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_reset_device(ctx->device);
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
    /* .reset                   = */ ggml_backend_cuda_device_reset,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_PEER_MAX_BATCH_SIZE
        features.push_back({ "PEER_MAX_BATCH_SIZE", STRINGIFY(GGML_CUDA_PEER_MAX_BATCH_SIZE) });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_split_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            int driverVersion = 0;

            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;
                dev_ctx->id = ggml_cuda_parse_uuid(prop, i);

                char pci_bus_id[16] = {};
                snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
                dev_ctx->pci_bus_id = pci_bus_id;

                dev_ctx->major = prop.major;
                dev_ctx->minor = prop.minor;
                if (driverVersion == 0) {
                    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
                }
                dev_ctx->driver_major = driverVersion / 1000;
                dev_ctx->driver_minor = (driverVersion - (dev_ctx->driver_major * 1000)) / 10;
                dev_ctx->integrated = prop.integrated;
                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
