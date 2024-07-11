#ifndef __APPLE__
#ifndef __GPU_INFO_NVCUDA_H__
#define __GPU_INFO_NVCUDA_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_INSUFFICIENT_DRIVER = 35,
  CUDA_ERROR_NO_DEVICE = 100,
  CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  CUDA_ERROR_UNKNOWN = 999,
  // Other values omitted for now...
} CUresult;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,

  // TODO - not yet wired up but may be useful for Jetson or other
  // integrated GPU scenarios with shared memory
  CU_DEVICE_ATTRIBUTE_INTEGRATED = 18

} CUdevice_attribute;

typedef void *nvcudaDevice_t;  // Opaque is sufficient
typedef struct nvcudaMemory_st {
  uint64_t total;
  uint64_t free;
} nvcudaMemory_t;

typedef struct nvcudaDriverVersion {
  int major;
  int minor;
} nvcudaDriverVersion_t;

typedef struct CUuuid_st {
    unsigned char bytes[16];
} CUuuid;

typedef int CUdevice;
typedef void* CUcontext;

typedef struct nvcuda_handle {
  void *handle;
  uint16_t verbose;
  int driver_major;
  int driver_minor;
  CUresult (*cuInit)(unsigned int Flags);
  CUresult (*cuDriverGetVersion)(int *driverVersion);
  CUresult (*cuDeviceGetCount)(int *);
  CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
  CUresult (*cuDeviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev);
  CUresult (*cuDeviceGetUuid)(CUuuid* uuid, CUdevice dev); // signature compatible with cuDeviceGetUuid_v2
  CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);

  // Context specific aspects
  CUresult (*cuCtxCreate_v3)(CUcontext* pctx, void *params, int len, unsigned int flags, CUdevice dev);
  CUresult (*cuMemGetInfo_v2)(uint64_t* free, uint64_t* total);
  CUresult (*cuCtxDestroy)(CUcontext ctx);
} nvcuda_handle_t;

typedef struct nvcuda_init_resp {
  char *err;  // If err is non-null handle is invalid
  nvcuda_handle_t ch;
  int num_devices;
  CUresult cudaErr;
} nvcuda_init_resp_t;

void nvcuda_init(char *nvcuda_lib_path, nvcuda_init_resp_t *resp);
void nvcuda_bootstrap(nvcuda_handle_t ch, int device_id, mem_info_t *resp);
void nvcuda_get_free(nvcuda_handle_t ch,  int device_id, uint64_t *free, uint64_t *total);
void nvcuda_release(nvcuda_handle_t ch);

#endif  // __GPU_INFO_NVCUDA_H__
#endif  // __APPLE__
