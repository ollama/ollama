#ifndef __APPLE__

#include "gpu_info_rocm.h"

#include <string.h>

void rocm_init(char *rocm_lib_path, rocm_init_resp_t *resp) {
  rsmi_status_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"rsmi_init", (void *)&resp->rh.rsmi_init},
      {"rsmi_shut_down", (void *)&resp->rh.rsmi_shut_down},
      {"rsmi_dev_memory_total_get", (void *)&resp->rh.rsmi_dev_memory_total_get},
      {"rsmi_dev_memory_usage_get", (void *)&resp->rh.rsmi_dev_memory_usage_get},
      {"rsmi_version_get", (void *)&resp->rh.rsmi_version_get},
      {"rsmi_num_monitor_devices", (void*)&resp->rh.rsmi_num_monitor_devices},
      {"rsmi_dev_id_get", (void*)&resp->rh.rsmi_dev_id_get},
      {"rsmi_dev_name_get", (void *)&resp->rh.rsmi_dev_name_get},
      {"rsmi_dev_brand_get", (void *)&resp->rh.rsmi_dev_brand_get},
      {"rsmi_dev_vendor_name_get", (void *)&resp->rh.rsmi_dev_vendor_name_get},
      {"rsmi_dev_vram_vendor_get", (void *)&resp->rh.rsmi_dev_vram_vendor_get},
      {"rsmi_dev_serial_number_get", (void *)&resp->rh.rsmi_dev_serial_number_get},
      {"rsmi_dev_subsystem_name_get", (void *)&resp->rh.rsmi_dev_subsystem_name_get},
      {"rsmi_dev_vbios_version_get", (void *)&resp->rh.rsmi_dev_vbios_version_get},
      {NULL, NULL},
  };

  resp->rh.handle = LOAD_LIBRARY(rocm_lib_path, RTLD_LAZY);
  if (!resp->rh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Radeon GPUs: %s\n",
             rocm_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->rh.verbose, "wiring rocm management library functions in %s\n", rocm_lib_path);

  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->rh.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->rh.handle, l[i].s);
    if (!l[i].p) {
      resp->rh.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->rh.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->rh.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->rh.rsmi_init)(0);
  if (ret != RSMI_STATUS_SUCCESS) {
    LOG(resp->rh.verbose, "rsmi_init err: %d\n", ret);
    UNLOAD_LIBRARY(resp->rh.handle);
    resp->rh.handle = NULL;
    snprintf(buf, buflen, "rocm vram init failure: %d", ret);
    resp->err = strdup(buf);
  }

  return;
}

void rocm_check_vram(rocm_handle_t h, mem_info_t *resp) {
  resp->err = NULL;
  resp->igpu_index = -1;
  uint64_t totalMem = 0;
  uint64_t usedMem = 0;
  rsmi_status_t ret;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (h.handle == NULL) {
    resp->err = strdup("rocm handle not initialized");
    return;
  }

  ret = (*h.rsmi_num_monitor_devices)(&resp->count);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }
  LOG(h.verbose, "discovered %d ROCm GPU Devices\n", resp->count);

  resp->total = 0;
  resp->free = 0;
  for (i = 0; i < resp->count; i++) {
    if (h.verbose) {
      // When in verbose mode, report more information about
      // the card we discover, but don't fail on error
      ret = (*h.rsmi_dev_name_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_name_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm device name: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_brand_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_brand_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm brand: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_vendor_name_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_vendor_name_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm vendor: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_vram_vendor_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_vram_vendor_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm VRAM vendor: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_serial_number_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_serial_number_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm S/N: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_subsystem_name_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_subsystem_name_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm subsystem name: %s\n", i, buf);
      }
      ret = (*h.rsmi_dev_vbios_version_get)(i, buf, buflen);
      if (ret != RSMI_STATUS_SUCCESS) {
        LOG(h.verbose, "rsmi_dev_vbios_version_get failed: %d\n", ret);
      } else {
        LOG(h.verbose, "[%d] ROCm vbios version: %s\n", i, buf);
      }
    }

    // Get total memory - used memory for available memory
    ret = (*h.rsmi_dev_memory_total_get)(i, RSMI_MEM_TYPE_VRAM, &totalMem);
    if (ret != RSMI_STATUS_SUCCESS) {
      snprintf(buf, buflen, "rocm total mem lookup failure: %d", ret);
      resp->err = strdup(buf);
      return;
    }
    ret = (*h.rsmi_dev_memory_usage_get)(i, RSMI_MEM_TYPE_VRAM, &usedMem);
    if (ret != RSMI_STATUS_SUCCESS) {
      snprintf(buf, buflen, "rocm usage mem lookup failure: %d", ret);
      resp->err = strdup(buf);
      return;
    }
    LOG(h.verbose, "[%d] ROCm totalMem %ld\n", i, totalMem);
    LOG(h.verbose, "[%d] ROCm usedMem %ld\n", i, usedMem);
    if (totalMem < 1024 * 1024 * 1024) {
      // Do not add up integrated GPU memory capacity, it's a bogus 512M, and actually uses system memory
      LOG(h.verbose, "[%d] ROCm integrated GPU\n", i);
      resp->igpu_index = i;
    } else {
      resp->total += totalMem;
      resp->free += totalMem - usedMem;
    }
  }
}

void rocm_get_version(rocm_handle_t h, rocm_version_resp_t *resp) {
  const int buflen = 256;
  char buf[buflen + 1];
  if (h.handle == NULL) {
    resp->str = strdup("rocm handle not initialized");
    resp->status = 1;
    return;
  }
  rsmi_version_t ver;
  rsmi_status_t ret;
  ret = h.rsmi_version_get(&ver);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "unexpected response on version lookup %d", ret);
    resp->status = 1;
  } else {
    snprintf(buf, buflen, "%d", ver.major);
    resp->status = 0;
  }
  resp->str = strdup(buf);
}

#endif  // __APPLE__
