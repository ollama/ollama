#ifndef __APPLE__
#ifndef __GPU_INFO_NVML_H__
#define __GPU_INFO_NVML_H__

#include <stdio.h>
#include <stdlib.h>
#include "gpu_info.h"

// NVML types and structures
typedef int nvmlReturn_t;
#define NVML_SUCCESS 0
#define NVML_ERROR_UNINITIALIZED 1
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_ERROR_NO_PERMISSION 4
#define NVML_ERROR_ALREADY_INITIALIZED 5
#define NVML_ERROR_NOT_FOUND 6
#define NVML_ERROR_INSUFFICIENT_SIZE 7
#define NVML_ERROR_INSUFFICIENT_POWER 8
#define NVML_ERROR_DRIVER_NOT_LOADED 9
#define NVML_ERROR_TIMEOUT 10
#define NVML_ERROR_IRQ_ISSUE 11
#define NVML_ERROR_LIBRARY_NOT_FOUND 12
#define NVML_ERROR_FUNCTION_NOT_FOUND 13
#define NVML_ERROR_CORRUPTED_INFOROM 14
#define NVML_ERROR_GPU_IS_LOST 15
#define NVML_ERROR_RESET_REQUIRED 16
#define NVML_ERROR_OPERATING_SYSTEM 17
#define NVML_ERROR_LIB_RM_VERSION_MISMATCH 18
#define NVML_ERROR_IN_USE 19
#define NVML_ERROR_MEMORY 20
#define NVML_ERROR_NO_DATA 21
#define NVML_ERROR_VGPU_ECC_NOT_SUPPORTED 22
#define NVML_ERROR_INSUFFICIENT_RESOURCES 23
#define NVML_ERROR_UNKNOWN 999

typedef void* nvmlDevice_t;

typedef struct nvmlMemory_st {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
    int major;
    int minor;
} nvmlMemory_t;

// Just enough typedef's to dlopen/dlsym for memory information
typedef struct nvml_handle {
    void *handle;
    uint16_t verbose;
    nvmlReturn_t (*nvmlInit)(void);
    nvmlReturn_t (*nvmlShutdown)(void);
    nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int *);
    nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, nvmlDevice_t *);
    nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
    nvmlReturn_t (*nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int *, int *);
    nvmlReturn_t (*nvmlErrorString)(nvmlReturn_t);
} nvml_handle_t;

typedef struct nvml_init_resp {
    char *err;  // If err is non-null handle is invalid
    nvml_handle_t nh;
    unsigned int num_devices;
} nvml_init_resp_t;

void nvml_init(char *nvml_lib_path, nvml_init_resp_t *resp);
void nvml_check_vram(nvml_handle_t nh, int device_id, mem_info_t *resp);
void nvml_release(nvml_handle_t nh);

#endif  // __GPU_INFO_NVML_H__
#endif  // __APPLE__
