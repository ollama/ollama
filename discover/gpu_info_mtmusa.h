#ifndef __APPLE__
#ifndef __GPU_INFO_MTMUSA_H__
#define __GPU_INFO_MTMUSA_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum musaError_enum {
  MUSA_SUCCESS = 0,
  MUSA_ERROR_NO_DEVICE = 100,
  MUSA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
  MUSA_ERROR_UNKNOWN = 999,
  // Other values omitted for now...
} MUresult;

typedef enum MUdevice_attribute_enum {
  MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
} MUdevice_attribute;

typedef struct mtmusaMemory_st {
  uint64_t total;
  uint64_t free;
} mtmusaMemory_t;

typedef struct mtmusaDriverVersion {
  int major;
  int minor;
} mtmusaDriverVersion_t;

typedef struct MUuuid_st {
    unsigned char bytes[16];
} MUuuid;

typedef int MUdevice;
typedef void* MUcontext;

typedef struct mtmusa_handle {
  void *handle;
  uint16_t verbose;
  int driver_major;
  int driver_minor;
  MUresult (*muInit)(unsigned int Flags);
  MUresult (*muDriverGetVersion)(int *driverVersion);
  MUresult (*muDeviceGetCount)(int *);
  MUresult (*muDeviceGet)(MUdevice* device, int ordinal);
  MUresult (*muDeviceGetAttribute)(int* pi, MUdevice_attribute attrib, MUdevice dev);
  MUresult (*muDeviceGetUuid_v2)(MUuuid* uuid, MUdevice dev);
  MUresult (*muDeviceGetName)(char *name, int len, MUdevice dev);

  // Context specific aspects
  MUresult (*muCtxCreate_v2)(MUcontext* pctx, void *params, int len, unsigned int flags, MUdevice dev);
  MUresult (*muMemGetInfo_v2)(uint64_t* free, uint64_t* total);
  MUresult (*muCtxDestroy_v2)(MUcontext ctx);
} mtmusa_handle_t;

typedef struct mtmusa_init_resp {
  char *err;  // If err is non-null handle is invalid
  mtmusa_handle_t ch;
  int num_devices;
  MUresult musaErr;
} mtmusa_init_resp_t;

void mtmusa_init(char *mtmusa_lib_path, mtmusa_init_resp_t *resp);
void mtmusa_bootstrap(mtmusa_handle_t ch, int device_id, mem_info_t *resp);
void mtmusa_get_free(mtmusa_handle_t ch,  int device_id, uint64_t *free, uint64_t *total);
void mtmusa_release(mtmusa_handle_t ch);

#endif  // __GPU_INFO_MTMUSA_H__
#endif  // __APPLE__
