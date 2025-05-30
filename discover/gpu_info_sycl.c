#ifndef __APPLE__

#include "gpu_info_sycl.h"

#include <inttypes.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

// We need to define the expected function signatures for our C wrapper around the SYCL C++ API
// These functions should be implemented in a separate library that wraps the SYCL C++ API

void sycl_init(char *sycl_lib_path, sycl_init_resp_t *resp) {
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      // These function names should match what's exported by the SYCL wrapper library
      {"sycl_get_platform_count", (void *)&resp->oh.sycl_get_platform_count},
      {"sycl_get_device_count", (void *)&resp->oh.sycl_get_device_count},
      {"sycl_get_device_ids", (void *)&resp->oh.sycl_get_device_ids},
      {"sycl_get_device_name", (void *)&resp->oh.sycl_get_device_name},
      {"sycl_get_device_vendor", (void *)&resp->oh.sycl_get_device_vendor},
      {"sycl_get_device_memory", (void *)&resp->oh.sycl_get_device_memory},
      {"sycl_is_gpu", (void *)&resp->oh.sycl_is_gpu},
      {NULL, NULL},
  };

  resp->oh.handle = (void *)LOAD_LIBRARY(sycl_lib_path, RTLD_LAZY);
  if (!resp->oh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for SYCL devices: %s\n", sycl_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }
  LOG(resp->oh.verbose, "wiring SYCL management library functions in %s\n", sycl_lib_path);

  for (i = 0; l[i].s != NULL; i++) {
    LOG(resp->oh.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->oh.handle, l[i].s);
    if (!*(l[i].p)) {
      resp->oh.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->oh.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->oh.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s, msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  // Initialize SYCL and check for platforms/devices
  int platform_count = (resp->oh.sycl_get_platform_count)();
  if (platform_count <= 0) {
    snprintf(buf, buflen, "No SYCL platforms found");
    resp->err = strdup(buf);
    UNLOAD_LIBRARY(resp->oh.handle);
    return;
  }

  LOG(resp->oh.verbose, "Found %d SYCL platforms\n", platform_count);
  
  // Get device count across all platforms
  int device_count = (resp->oh.sycl_get_device_count)();
  if (device_count <= 0) {
    snprintf(buf, buflen, "No SYCL devices found");
    resp->err = strdup(buf);
    UNLOAD_LIBRARY(resp->oh.handle);
    return;
  }

  LOG(resp->oh.verbose, "Found %d SYCL devices\n", device_count);
  return;
}

void sycl_release(sycl_handle_t h) {
  LOG(h.verbose, "releasing SYCL library\n");
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

void sycl_get_gpu_list(sycl_handle_t h, int *id_list, int max_len) {
  if (h.handle == NULL) {
    return;
  }
  
  (h.sycl_get_device_ids)(id_list, max_len);
}

void sycl_print_sycl_devices(sycl_handle_t h) {
  if (h.handle == NULL) {
    return;
  }
  
  int device_count = (h.sycl_get_device_count)();
  printf("SYCL Devices (%d):\n", device_count);
  
  for (int i = 0; i < device_count; i++) {
    char name[256];
    char vendor[256];
    (h.sycl_get_device_name)(i, name, sizeof(name));
    (h.sycl_get_device_vendor)(i, vendor, sizeof(vendor));
    
    size_t free_mem = 0, total_mem = 0;
    (h.sycl_get_device_memory)(i, &free_mem, &total_mem);
    
    int is_gpu = (h.sycl_is_gpu)(i);
    
    printf("  Device %d: %s by %s (Free: %zu MB, Total: %zu MB) %s\n", 
           i, name, vendor, free_mem / (1024 * 1024), total_mem / (1024 * 1024),
           is_gpu ? "[GPU]" : "");
  }
}

int sycl_get_device_count(sycl_handle_t h) {
  if (h.handle == NULL) {
    return 0;
  }
  
  return (h.sycl_get_device_count)();
}

void sycl_check_vram(sycl_handle_t h, int device, mem_info_t *resp) {
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];

  if (h.handle == NULL) {
    resp->err = strdup("SYCL handle not initialized");
    return;
  }
  
  int device_count = (h.sycl_get_device_count)();
  if (device < 0 || device >= device_count) {
    snprintf(buf, buflen, "SYCL device index out of bounds: %d (max: %d)", device, device_count - 1);
    resp->err = strdup(buf);
    return;
  }
  
  // Check if the device is a GPU
  int is_gpu = (h.sycl_is_gpu)(device);
  if (!is_gpu) {
    snprintf(buf, buflen, "SYCL device %d is not a GPU", device);
    resp->err = strdup(buf);
    return;
  }
  
  resp->total = 0;
  resp->free = 0;
  
  // Get device memory information
  (h.sycl_get_device_memory)(device, &resp->free, &resp->total);
  
  // Get device name
  (h.sycl_get_device_name)(device, &resp->gpu_name[0], GPU_NAME_LEN);
  
  // Set device ID (just use the index for now)
  snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", device);
  
  LOG(h.verbose, "[%s] SYCL totalMem %" PRId64 "mb\n", resp->gpu_id, resp->total / 1024 / 1024);
  LOG(h.verbose, "[%s] SYCL freeMem %" PRId64 "mb\n", resp->gpu_id, resp->free / 1024 / 1024);
}

#endif // __APPLE__