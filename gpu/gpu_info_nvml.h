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
void nvml_get_free(nvml_handle_t ch,  int device_id, uint64_t *free, uint64_t *total, uint64_t *used);
void nvml_release(nvml_handle_t ch);

#endif  // __GPU_INFO_NVML_H__
#endif  // __APPLE__