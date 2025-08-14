#ifndef __APPLE__
#ifndef __GPU_INFO_SYCL_H__
#define __GPU_INFO_SYCL_H__
#include "gpu_info.h"

typedef struct sycl_handle {
  void *handle;
  uint16_t verbose;

  void (*ggml_backend_sycl_get_gpu_list)(int *id_list, int max_len);
  void (*ggml_backend_sycl_print_sycl_devices)(void);
  int  (*ggml_backend_sycl_get_device_count)();
  void (*ggml_backend_sycl_get_device_memory)(uint32_t device, size_t *free, size_t *total);
  void (*ggml_backend_sycl_get_device_description)(uint32_t device, char *description, size_t description_size);
} sycl_handle_t;

typedef struct sycl_init_resp {
  char *err; // If err is non-null handle is invalid
  sycl_handle_t oh;
} sycl_init_resp_t;

void sycl_init(char *oneapi_lib_path, sycl_init_resp_t *resp);
void sycl_release(sycl_handle_t h);
void sycl_get_gpu_list(sycl_handle_t h, int *id_list, int max_len);
void sycl_print_sycl_devices(sycl_handle_t h);
int sycl_get_device_count(sycl_handle_t h);
void sycl_check_vram(sycl_handle_t h, int device, mem_info_t *resp);
#endif // __GPU_INFO_INTEL_H__
#endif // __APPLE__