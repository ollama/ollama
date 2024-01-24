#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include "gpu_info_cuda.h"

#include <string.h>

#define CUDA_LOOKUP_SIZE 12

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
      {"nvmlSystemGetDriverVersion", (void *)&resp->ch.nvmlSystemGetDriverVersion},
      {"nvmlDeviceGetName", (void *)&resp->ch.nvmlDeviceGetName},
      {"nvmlDeviceGetSerial", (void *)&resp->ch.nvmlDeviceGetSerial},
      {"nvmlDeviceGetVbiosVersion", (void *)&resp->ch.nvmlDeviceGetVbiosVersion},
      {"nvmlDeviceGetBoardPartNumber", (void *)&resp->ch.nvmlDeviceGetBoardPartNumber},
      {"nvmlDeviceGetBrand", (void *)&resp->ch.nvmlDeviceGetBrand},
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

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->ch.nvmlSystemGetDriverVersion)(buf, buflen);
  if (ret != NVML_SUCCESS) {
    LOG(resp->ch.verbose, "nvmlSystemGetDriverVersion failed: %d\n", ret);
  } else {
    LOG(resp->ch.verbose, "CUDA driver version: %s\n", buf);
  }
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
    if (h.verbose) {
      nvmlBrandType_t brand = 0;
      // When in verbose mode, report more information about
      // the card we discover, but don't fail on error
      ret = (*h.nvmlDeviceGetName)(device, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "nvmlDeviceGetName failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] CUDA device name: %s\n", i, buf);
      }
      ret = (*h.nvmlDeviceGetBoardPartNumber)(device, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "nvmlDeviceGetBoardPartNumber failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] CUDA part number: %s\n", i, buf);
      }
      ret = (*h.nvmlDeviceGetSerial)(device, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "nvmlDeviceGetSerial failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] CUDA S/N: %s\n", i, buf);
      }
      ret = (*h.nvmlDeviceGetVbiosVersion)(device, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "nvmlDeviceGetVbiosVersion failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] CUDA vbios version: %s\n", i, buf);
      }
      ret = (*h.nvmlDeviceGetBrand)(device, &brand);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "nvmlDeviceGetBrand failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] CUDA brand: %d\n", i, brand);
      }
    }

    LOG(h.verbose, "[%d] CUDA totalMem %ld\n", i, memInfo.total);
    LOG(h.verbose, "[%d] CUDA usedMem %ld\n", i, memInfo.free);

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