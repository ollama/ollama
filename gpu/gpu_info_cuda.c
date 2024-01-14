#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include "gpu_info_cuda.h"

#include <string.h>

#define CUDA_LOOKUP_SIZE 6

void cuda_init(char *cuda_lib_path, cuda_init_resp_t *resp) {
  nvmlReturn_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[CUDA_LOOKUP_SIZE] = {
      {"nvmlInit_v2", (void *)&resp->ch.initFn},
      {"nvmlShutdown", (void *)&resp->ch.shutdownFn},
      {"nvmlDeviceGetHandleByIndex", (void *)&resp->ch.getHandle},
      {"nvmlDeviceGetMemoryInfo", (void *)&resp->ch.getMemInfo},
      {"nvmlDeviceGetCount_v2", (void *)&resp->ch.getCount},
      {"nvmlDeviceGetCudaComputeCapability", (void *)&resp->ch.getComputeCapability},
  };

  resp->ch.handle = LOAD_LIBRARY(cuda_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Nvidia GPUs: %s",
             cuda_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; i < CUDA_LOOKUP_SIZE; i++) {  // TODO - fix this to use a null terminated list
    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!l[i].p) {
      UNLOAD_LIBRARY(resp->ch.handle);
      resp->ch.handle = NULL;
      char *msg = LOAD_ERR();
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->ch.initFn)();
  if (ret != NVML_SUCCESS) {
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "nvml vram init failure: %d", ret);
    resp->err = strdup(buf);
  }

  return;
}

void cuda_check_vram(cuda_handle_t h, mem_info_t *resp) {
  resp->err = NULL;
  nvmlDevice_t device;
  nvmlMemory_t memInfo = {0};
  nvmlReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (h.handle == NULL) {
    resp->err = strdup("nvml handle sn't initialized");
    return;
  }

  ret = (*h.getCount)(&resp->count);
  if (ret != NVML_SUCCESS) {
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  resp->total = 0;
  resp->free = 0;
  for (i = 0; i < resp->count; i++) {
    ret = (*h.getHandle)(i, &device);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "unable to get device handle %d: %d", i, ret);
      resp->err = strdup(buf);
      return;
    }

    ret = (*h.getMemInfo)(device, &memInfo);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "device memory info lookup failure %d: %d", i, ret);
      resp->err = strdup(buf);
      return;
    }

    resp->total += memInfo.total;
    resp->free += memInfo.free;
  }
}

void cuda_compute_capability(cuda_handle_t h, cuda_compute_capability_t *resp) {
  resp->err = NULL;
  resp->major = 0;
  resp->minor = 0;
  nvmlDevice_t device;
  int major = 0;
  int minor = 0;
  nvmlReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (h.handle == NULL) {
    resp->err = strdup("nvml handle not initialized");
    return;
  }

  unsigned int devices;
  ret = (*h.getCount)(&devices);
  if (ret != NVML_SUCCESS) {
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; i < devices; i++) {
    ret = (*h.getHandle)(i, &device);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "unable to get device handle %d: %d", i, ret);
      resp->err = strdup(buf);
      return;
    }

    ret = (*h.getComputeCapability)(device, &major, &minor);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, ret);
      resp->err = strdup(buf);
      return;
    }
    // Report the lowest major.minor we detect as that limits our compatibility
    if (resp->major == 0 || resp->major > major ) {
      resp->major = major;
      resp->minor = minor;
    } else if ( resp->major == major && resp->minor > minor ) {
      resp->minor = minor;
    }
  }
}
#endif  // __APPLE__