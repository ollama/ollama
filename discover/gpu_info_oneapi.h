#ifndef __APPLE__
#ifndef __GPU_INFO_ONEAPI_H__
#define __GPU_INFO_ONEAPI_H__
#include "gpu_info.h"

#define ZE_MAX_DEVICE_NAME 256
#define ZE_MAX_DEVICE_UUID_SIZE 16
#define ZES_STRING_PROPERTY_SIZE 64
#define ZE_BIT(_i) (1 << _i)

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum ze_result_t {
  ZE_RESULT_SUCCESS = 0,
  // Other values omitted for now...
} ze_result_t;

typedef uint8_t ze_bool_t;
typedef struct _zes_driver_handle_t *zes_driver_handle_t;
typedef struct _zes_device_handle_t *zes_device_handle_t;
typedef struct _zes_mem_handle_t *zes_mem_handle_t;

typedef enum _ze_structure_type_t {
  ZE_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff
} ze_structure_type_t;

typedef enum _zes_structure_type_t {
  ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1,
  ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb,
  ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e,
  ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES = 0x2d,
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

typedef struct _ze_device_uuid_t {
  uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];
} ze_device_uuid_t;

typedef struct _zes_uuid_t {
  uint8_t id[ZE_MAX_DEVICE_UUID_SIZE];
} zes_uuid_t;

typedef enum _ze_device_type_t {
  ZE_DEVICE_TYPE_GPU = 1,
  ZE_DEVICE_TYPE_CPU = 2,
  ZE_DEVICE_TYPE_FPGA = 3,
  ZE_DEVICE_TYPE_MCA = 4,
  ZE_DEVICE_TYPE_VPU = 5,
  ZE_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
} ze_device_type_t;

typedef enum _zes_device_type_t {
  ZES_DEVICE_TYPE_GPU = 1,
  ZES_DEVICE_TYPE_CPU = 2,
  ZES_DEVICE_TYPE_FPGA = 3,
  ZES_DEVICE_TYPE_MCA = 4,
  ZES_DEVICE_TYPE_VPU = 5,
  ZES_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
} zes_device_type_t;

typedef uint32_t ze_device_property_flags_t;
typedef enum _ze_device_property_flag_t {
  ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0),
  ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE = ZE_BIT(1),
  ZE_DEVICE_PROPERTY_FLAG_ECC = ZE_BIT(2),
  ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = ZE_BIT(3),
  ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_device_property_flag_t;

typedef uint32_t zes_device_property_flags_t;
typedef enum _zes_device_property_flag_t {
  ZES_DEVICE_PROPERTY_FLAG_INTEGRATED = ZE_BIT(0),
  ZES_DEVICE_PROPERTY_FLAG_SUBDEVICE = ZE_BIT(1),
  ZES_DEVICE_PROPERTY_FLAG_ECC = ZE_BIT(2),
  ZES_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING = ZE_BIT(3),
  ZES_DEVICE_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff
} zes_device_property_flag_t;

typedef struct _ze_device_properties_t {
  ze_structure_type_t stype;
  void *pNext;
  ze_device_type_t type;
  uint32_t vendorId;
  uint32_t deviceId;
  ze_device_property_flags_t flags;
  uint32_t subdeviceId;
  uint32_t coreClockRate;
  uint64_t maxMemAllocSize;
  uint32_t maxHardwareContexts;
  uint32_t maxCommandQueuePriority;
  uint32_t numThreadsPerEU;
  uint32_t physicalEUSimdWidth;
  uint32_t numEUsPerSubslice;
  uint32_t numSubslicesPerSlice;
  uint32_t numSlices;
  uint64_t timerResolution;
  uint32_t timestampValidBits;
  uint32_t kernelTimestampValidBits;
  ze_device_uuid_t uuid;
  char name[ZE_MAX_DEVICE_NAME];
} ze_device_properties_t;

typedef struct _zes_device_properties_t {
  zes_structure_type_t stype;
  void *pNext;
  ze_device_properties_t core;
  uint32_t numSubdevices;
  char serialNumber[ZES_STRING_PROPERTY_SIZE];
  char boardNumber[ZES_STRING_PROPERTY_SIZE];
  char brandName[ZES_STRING_PROPERTY_SIZE];
  char modelName[ZES_STRING_PROPERTY_SIZE];
  char vendorName[ZES_STRING_PROPERTY_SIZE];
  char driverVersion[ZES_STRING_PROPERTY_SIZE];
} zes_device_properties_t;

typedef struct _zes_device_ext_properties_t {
  zes_structure_type_t stype;
  void *pNext;
  zes_uuid_t uuid;
  zes_device_type_t type;
  zes_device_property_flags_t flags;
} zes_device_ext_properties_t;

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

  uint32_t num_drivers;
  zes_driver_handle_t *drivers;
  uint32_t *num_devices;
  zes_device_handle_t **devices;

  // TODO Driver major, minor information
  // int driver_major;
  // int driver_minor;

  ze_result_t (*zesInit)(int);
  ze_result_t (*zesDriverGet)(uint32_t *pCount, zes_driver_handle_t *phDrivers);
  ze_result_t (*zesDeviceGet)(zes_driver_handle_t hDriver, uint32_t *pCount,
                              zes_device_handle_t *phDevices);
  ze_result_t (*zesDeviceGetProperties)(zes_device_handle_t hDevice,
                                        zes_device_properties_t *pProperties);
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
void oneapi_check_vram(oneapi_handle_t h, int driver, int device,
                       mem_info_t *resp);
void oneapi_release(oneapi_handle_t h);
int oneapi_get_device_count(oneapi_handle_t h, int driver);

#endif // __GPU_INFO_INTEL_H__
#endif // __APPLE__
