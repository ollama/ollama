#ifndef __APPLE__
#ifndef __GPU_INFO_ASCEND_H__
#define __GPU_INFO_ASCEND_H__
#include "gpu_info.h"

typedef int aclError;

typedef struct aclrtUtilizationExtendInfo aclrtUtilizationExtendInfo;

typedef struct aclrtUtilizationInfo {
    int32_t cubeUtilization; 
    int32_t vectorUtilization;
    int32_t aicpuUtilization;
    int32_t memoryUtilization;
    aclrtUtilizationExtendInfo *utilizationExtend;
} aclrtUtilizationInfo;

typedef enum aclrtMemAttr {
    ACL_DDR_MEM,
    ACL_HBM_MEM,
    ACL_DDR_MEM_HUGE,
    ACL_DDR_MEM_NORMAL,
    ACL_HBM_MEM_HUGE,
    ACL_HBM_MEM_NORMAL,
    ACL_DDR_MEM_P2P_HUGE,
    ACL_DDR_MEM_P2P_NORMAL,
    ACL_HBM_MEM_P2P_HUGE,
    ACL_HBM_MEM_P2P_NORMAL,
} aclrtMemAttr;

typedef enum aclrtDeviceStatus {
    ACL_RT_DEVICE_STATUS_NORMAL = 0,
    ACL_RT_DEVICE_STATUS_ABNORMAL,
    ACL_RT_DEVICE_STATUS_END = 0xFFFF,
} aclrtDeviceStatus;

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum ascendError_enum {
  ACL_SUCCESS = 0,
  ACL_ERROR_REPEAT_INITIALIZE  = 100002,
  ACL_ERROR_REPEAT_FINALIZE = 100037,
  // Other values omitted for now...
} ACLresult;

typedef struct ascend_handle
{
  void *handle;
  uint16_t verbose;

  int driver_major;
  int driver_minor;

  aclError (*aclInit)(char *configPath);
  aclError (*aclFinalize)(void);
  aclError (*aclrtSetDevice)(int32_t deviceId);
  aclError (*aclrtResetDevice)(int32_t deviceId);
  aclError (*aclrtGetVersion)(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);
  aclError (*aclrtGetDeviceCount)(uint32_t *count);
  aclError (*aclrtQueryDeviceStatus)(int32_t deviceId, aclrtDeviceStatus *deviceStatus);
  aclError (*aclrtGetMemInfo)(aclrtMemAttr attr, size_t *free, size_t *total);

  const char *(*aclrtGetSocName)(void);
  const char *(*aclGetRecentErrMsg)(void);
} ascend_handle_t;

typedef struct ascend_init_resp
{
  char *err; // If err is non-null handle is invalid
  int num_devices;
  ascend_handle_t ah;
} ascend_init_resp_t;

void ascend_init(char *ascend_lib_path, ascend_init_resp_t *resp);
void ascend_bootstrap(ascend_handle_t h, int device_id, mem_info_t *resp);
void ascend_release(ascend_handle_t h);

#endif // __GPU_INFO_ASCEND_H__
#endif // __APPLE__
