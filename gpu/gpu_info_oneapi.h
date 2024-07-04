#ifndef __APPLE__
#ifndef __GPU_INFO_ONEAPI_H__
#define __GPU_INFO_ONEAPI_H__
#include "gpu_info.h"

#define SYCL_MAX_CHAR_BUF_SIZE 256

struct dev_info {
  char vendor_name[SYCL_MAX_CHAR_BUF_SIZE];
  char device_name[SYCL_MAX_CHAR_BUF_SIZE];
  uint32_t device_id;
};

struct runtime_info {
  char driver_version[SYCL_MAX_CHAR_BUF_SIZE];
  uint32_t global_mem_size;
  uint32_t free_mem;
};

struct gpu_info {
  struct dev_info dev;
  struct runtime_info runtime;
};

// Just enough typedef's to dlopen/dlsym for memory information
typedef struct oneapi_handle {
  void* handle;
  uint16_t verbose;
  void (*get_dev_info)(int dev_idx, struct gpu_info* info);
  int (*get_device_num)();
} oneapi_handle_t;

typedef struct oneapi_init_resp {
  char* err;  // If err is non-null handle is invalid
  oneapi_handle_t oh;
} oneapi_init_resp_t;

void oneapi_init(char* oneapi_lib_path, oneapi_init_resp_t* resp);
void oneapi_check_dev(oneapi_handle_t h, int dev_idx, struct gpu_info* resp);
int oneapi_get_device_count(oneapi_handle_t h);

#endif  // __GPU_INFO_INTEL_H__
#endif  // __APPLE__
