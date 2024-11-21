#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include <string.h>

#include "gpu_info_nvml.h"

void nvml_init(char *nvml_lib_path, nvml_init_resp_t *resp) {
  nvmlReturn_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"nvmlInit_v2", (void *)&resp->ch.nvmlInit_v2},
      {"nvmlShutdown", (void *)&resp->ch.nvmlShutdown},
      {"nvmlDeviceGetHandleByUUID", (void *)&resp->ch.nvmlDeviceGetHandleByUUID},
      {"nvmlDeviceGetMemoryInfo", (void *)&resp->ch.nvmlDeviceGetMemoryInfo},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(nvml_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", nvml_lib_path, msg);
    snprintf(buf, buflen,
             "Unable to load %s library to query for Nvidia GPUs: %s",
             nvml_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  // LOG(resp->ch.verbose, "wiring nvidia management library functions in %s\n", nvml_lib_path);
  
  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    // LOG(resp->ch.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!*(l[i].p)) {
      resp->ch.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ch.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->ch.nvmlInit_v2)();
  if (ret != NVML_SUCCESS) {
    LOG(resp->ch.verbose, "nvmlInit_v2 err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "nvml vram init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }
}


void nvml_get_free(nvml_handle_t h, char *uuid, uint64_t *free, uint64_t *total, uint64_t *used) {
    nvmlDevice_t device;
    nvmlMemory_t memInfo = {0};
    nvmlReturn_t ret;
    ret = (*h.nvmlDeviceGetHandleByUUID)((const char *)(uuid), &device);
    if (ret != NVML_SUCCESS) {
        LOG(1, "unable to get device handle %s: %d", uuid, ret);
        *free = 0;
        return;
    }

    ret = (*h.nvmlDeviceGetMemoryInfo)(device, &memInfo);
    if (ret != NVML_SUCCESS) {
        LOG(1, "device memory info lookup failure %s: %d", uuid, ret);
        *free = 0;
        return;
    }
    *free = memInfo.free;
    *total = memInfo.total;
    *used = memInfo.used;
}


void nvml_release(nvml_handle_t h) {
  LOG(h.verbose, "releasing nvml library\n");
  nvmlReturn_t ret;
  ret = (*h.nvmlShutdown)();
  if (ret != NVML_SUCCESS) {
    LOG(1, "error during nvmlShutdown %d", ret);
  }
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

#endif  // __APPLE__