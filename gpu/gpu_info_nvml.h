#ifndef __APPLE__
#ifndef __GPU_INFO_NVML_H__
#define __GPU_INFO_NVML_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum nvmlReturn_enum {
  NVML_SUCCESS = 0,
  // Other values omitted for now...
} nvmlReturn_t;
typedef void *nvmlDevice_t;  // Opaque is sufficient
typedef struct nvmlMemory_st {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_t;

typedef enum nvmlBrandType_enum
{
    NVML_BRAND_UNKNOWN          = 0,
} nvmlBrandType_t;

typedef struct nvml_handle {
  void *handle;
  uint16_t verbose;
  nvmlReturn_t (*nvmlInit_v2)(void);
  nvmlReturn_t (*nvmlShutdown)(void);
  nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
  nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
  nvmlReturn_t (*nvmlDeviceGetCount_v2)(unsigned int *);
  nvmlReturn_t (*nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int* major, int* minor);
  nvmlReturn_t (*nvmlSystemGetDriverVersion) (char* version, unsigned int  length);
  nvmlReturn_t (*nvmlDeviceGetName) (nvmlDevice_t device, char* name, unsigned int  length);
  nvmlReturn_t (*nvmlDeviceGetSerial) (nvmlDevice_t device, char* serial, unsigned int  length);
  nvmlReturn_t (*nvmlDeviceGetVbiosVersion) (nvmlDevice_t device, char* version, unsigned int  length);
  nvmlReturn_t (*nvmlDeviceGetBoardPartNumber) (nvmlDevice_t device, char* partNumber, unsigned int  length);
  nvmlReturn_t (*nvmlDeviceGetBrand) (nvmlDevice_t device, nvmlBrandType_t* type);
} nvml_handle_t;

typedef struct nvml_init_resp {
  char *err;  // If err is non-null handle is invalid
  nvml_handle_t ch;
} nvml_init_resp_t;

typedef struct nvml_compute_capability {
  char *err;
  int major;
  int minor;
} nvml_compute_capability_t;

void nvml_init(char *nvml_lib_path, nvml_init_resp_t *resp);
void nvml_check_vram(nvml_handle_t ch, mem_info_t *resp);
void nvml_compute_capability(nvml_handle_t ch, nvml_compute_capability_t *cc);
void nvml_release(nvml_handle_t ch);

#endif  // __GPU_INFO_NVML_H__
#endif  // __APPLE__