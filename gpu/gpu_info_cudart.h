#ifndef __APPLE__
#ifndef __GPU_INFO_CUDART_H__
#define __GPU_INFO_CUDART_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum cudartReturn_enum {
  CUDART_SUCCESS = 0,
  CUDART_UNSUPPORTED = 1,
  CUDA_ERROR_INSUFFICIENT_DRIVER = 35,
  // Other values omitted for now...
} cudartReturn_t;

typedef enum cudartDeviceAttr_enum {
  cudartDevAttrComputeCapabilityMajor = 75,
  cudartDevAttrComputeCapabilityMinor = 76,
} cudartDeviceAttr_t;

typedef void *cudartDevice_t;  // Opaque is sufficient
typedef struct cudartMemory_st {
  size_t total;
  size_t free;
  size_t used;
} cudartMemory_t;

typedef struct cudartDriverVersion {
  int major;
  int minor;
} cudartDriverVersion_t;

typedef struct cudart_handle {
  void *handle;
  uint16_t verbose;
  cudartReturn_t (*cudaSetDevice)(int device);
  cudartReturn_t (*cudaDeviceSynchronize)(void);
  cudartReturn_t (*cudaDeviceReset)(void);
  cudartReturn_t (*cudaMemGetInfo)(size_t *, size_t *);
  cudartReturn_t (*cudaGetDeviceCount)(int *);
  cudartReturn_t (*cudaDeviceGetAttribute)(int* value, cudartDeviceAttr_t attr, int device);
  cudartReturn_t (*cudaDriverGetVersion) (int *driverVersion);
} cudart_handle_t;

typedef struct cudart_init_resp {
  char *err;  // If err is non-null handle is invalid
  cudart_handle_t ch;
} cudart_init_resp_t;

typedef struct cudart_compute_capability {
  char *err;
  int major;
  int minor;
} cudart_compute_capability_t;


void cudart_init(char *cudart_lib_path, cudart_init_resp_t *resp);
void cudart_check_vram(cudart_handle_t ch, mem_info_t *resp);
void cudart_compute_capability(cudart_handle_t th, cudart_compute_capability_t *cc);
void cudart_release(cudart_handle_t ch);

#endif  // __GPU_INFO_CUDART_H__
#endif  // __APPLE__
