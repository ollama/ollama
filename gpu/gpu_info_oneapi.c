#ifndef __APPLE__

#include "gpu_info_oneapi.h"

#include <string.h>

void oneapi_init(char *oneapi_lib_path, oneapi_init_resp_t *resp) {
  ze_result_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"zesInit", (void *)&resp->oh.zesInit},
      {"zesDriverGet", (void *)&resp->oh.zesDriverGet},
      {"zesDeviceGet", (void *)&resp->oh.zesDeviceGet},
      {"zesDeviceEnumMemoryModules",
       (void *)&resp->oh.zesDeviceEnumMemoryModules},
      {"zesMemoryGetProperties", (void *)&resp->oh.zesMemoryGetProperties},
      {"zesMemoryGetState", (void *)&resp->oh.zesMemoryGetState},
      {NULL, NULL},
  };

  resp->oh.handle = LOAD_LIBRARY(oneapi_lib_path, RTLD_LAZY);
  if (!resp->oh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Intel GPUs: %s\n",
             oneapi_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->oh.verbose, "wiring oneapi management library functions in %s\n",
      oneapi_lib_path);

  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->oh.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->oh.handle, l[i].s);
    if (!l[i].p) {
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

  ret = (*resp->oh.zesInit)(0);
  if (ret != ZE_RESULT_SUCCESS) {
    LOG(resp->oh.verbose, "zesInit err: %d\n", ret);
    UNLOAD_LIBRARY(resp->oh.handle);
    resp->oh.handle = NULL;
    snprintf(buf, buflen, "oneapi vram init failure: %d", ret);
    resp->err = strdup(buf);
  }

  return;
}

void oneapi_check_vram(oneapi_handle_t h, mem_info_t *resp) {
  ze_result_t ret;
  resp->err = NULL;
  resp->igpu_index = -1;
  uint64_t totalMem = 0;
  uint64_t usedMem = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i, d, m;

  if (h.handle == NULL) {
    resp->err = strdup("oneapi handle not initialized");
    return;
  }

  uint32_t driversCount = 0;
  ret = (*h.zesDriverGet)(&driversCount, NULL);
  if (ret != ZE_RESULT_SUCCESS) {
    snprintf(buf, buflen, "unable to get driver count: %d", ret);
    resp->err = strdup(buf);
    return;
  }
  LOG(h.verbose, "discovered %d LevelZero drivers\n", driversCount);

  zes_driver_handle_t *allDrivers =
      malloc(driversCount * sizeof(zes_driver_handle_t));
  (*h.zesDriverGet)(&driversCount, allDrivers);

  resp->total = 0;
  resp->free = 0;

  for (d = 0; d < driversCount; d++) {
    uint32_t deviceCount = 0;
    ret = (*h.zesDeviceGet)(allDrivers[d], &deviceCount, NULL);
    if (ret != ZE_RESULT_SUCCESS) {
      snprintf(buf, buflen, "unable to get device count: %d", ret);
      resp->err = strdup(buf);
      free(allDrivers);
      return;
    }

    LOG(h.verbose, "discovered %d LevelZero devices\n", deviceCount);

    resp->count += deviceCount;

    zes_device_handle_t *devices =
        malloc(deviceCount * sizeof(zes_device_handle_t));
    (*h.zesDeviceGet)(allDrivers[d], &deviceCount, devices);

    for (i = 0; i < deviceCount; i++) {
      if (h.verbose) {
        // When in verbose mode, report more information about
        // the card we discover, but don't fail on error

        // TODO
      }

      uint32_t memCount = 0;
      ret = (*h.zesDeviceEnumMemoryModules)(devices[i], &memCount, NULL);
      if (ret != ZE_RESULT_SUCCESS) {
        snprintf(buf, buflen,
                 "unable to enumerate LevelZero memory modules: %d", ret);
        resp->err = strdup(buf);
        free(allDrivers);
        free(devices);
        return;
      }

      LOG(h.verbose, "discovered %d LevelZero memory modules\n", memCount);

      zes_mem_handle_t *mems = malloc(memCount * sizeof(zes_mem_handle_t));
      (*h.zesDeviceEnumMemoryModules)(devices[i], &memCount, mems);

      for (m = 0; m < memCount; m++) {
        zes_mem_state_t state;
        ret = (*h.zesMemoryGetState)(mems[m], &state);
        if (ret != ZE_RESULT_SUCCESS) {
          snprintf(buf, buflen, "unable to get memory state: %d", ret);
          resp->err = strdup(buf);
          free(allDrivers);
          free(devices);
          free(mems);
          return;
        }

        resp->total += state.size;
        resp->free += state.free;
      }

      free(mems);
    }

    free(devices);
  }

  free(allDrivers);
}

#endif // __APPLE__
