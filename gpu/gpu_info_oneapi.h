#ifndef __APPLE__
#ifndef __GPU_INFO_ONEAPI_H__
#define __GPU_INFO_ONEAPI_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum ze_result_t {
  ZE_RESULT_SUCCESS = 0,
  // Other values omitted for now...
} ze_result_t;

typedef uint8_t ze_bool_t;
typedef struct _zes_driver_handle_t *zes_driver_handle_t;
typedef struct _zes_device_handle_t *zes_device_handle_t;
typedef struct _zes_mem_handle_t *zes_mem_handle_t;

typedef enum _zes_structure_type_t {
  ZES_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
} zes_structure_type_t;

typedef enum _zes_mem_type_t {
  ZES_MEM_TYPE_FORCE_UINT32 = 0x7fffffff
} zes_mem_type_t;

typedef enum _zes_mem_loc_t {
  ZES_MEM_LOC_SYSTEM = 0,
  ZES_MEM_LOC_DEVICE = 1,
  ZES_MEM_LOC_FORCE_UINT32 = 0x7fffffff
} zes_mem_loc_t;

typedef enum _zes_mem_health_t {
  ZES_MEM_HEALTH_FORCE_UINT32 = 0x7fffffff
} zes_mem_health_t;

typedef struct _zes_mem_properties_t {
  zes_structure_type_t stype;
  void *pNext;
  zes_mem_type_t type;
  ze_bool_t onSubdevice;
  uint32_t subdeviceId;
  zes_mem_loc_t location;
  uint64_t physicalSize;
  int32_t busWidth;
  int32_t numChannels;
} zes_mem_properties_t;

typedef struct _zes_mem_state_t {
  zes_structure_type_t stype;
  const void *pNext;
  zes_mem_health_t health;
  uint64_t free;
  uint64_t size;
} zes_mem_state_t;

typedef struct oneapi_handle {
  void *handle;
  uint16_t verbose;
  ze_result_t (*zesInit)(int);
  ze_result_t (*zesDriverGet)(uint32_t *pCount, zes_driver_handle_t *phDrivers);
  ze_result_t (*zesDeviceGet)(zes_driver_handle_t hDriver, uint32_t *pCount,
                              zes_device_handle_t *phDevices);
  ze_result_t (*zesDeviceEnumMemoryModules)(zes_device_handle_t hDevice,
                                            uint32_t *pCount,
                                            zes_mem_handle_t *phMemory);
  ze_result_t (*zesMemoryGetProperties)(zes_mem_handle_t hMemory,
                                        zes_mem_properties_t *pProperties);
  ze_result_t (*zesMemoryGetState)(zes_mem_handle_t hMemory,
                                   zes_mem_state_t *pState);

} oneapi_handle_t;

typedef struct oneapi_init_resp {
  char *err; // If err is non-null handle is invalid
  oneapi_handle_t oh;
} oneapi_init_resp_t;

typedef struct oneapi_version_resp {
  ze_result_t status;
  char *str; // Contains version or error string if status != 0
} oneapi_version_resp_t;

void oneapi_init(char *oneapi_lib_path, oneapi_init_resp_t *resp);
void oneapi_check_vram(oneapi_handle_t rh, mem_info_t *resp);

#endif // __GPU_INFO_INTEL_H__
#endif // __APPLE__
