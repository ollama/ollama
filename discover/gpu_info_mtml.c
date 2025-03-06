#ifndef __APPLE__

#include <string.h>

#include "gpu_info_mtml.h"

void mtml_init(char *mtml_lib_path, mtml_init_resp_t *resp) {
  MtmlReturn ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"mtmlLibraryInit", (void *)&resp->ch.mtmlLibraryInit},
      {"mtmlLibraryInitSystem", (void *)&resp->ch.mtmlLibraryInitSystem},
      {"mtmlLibraryShutDown", (void *)&resp->ch.mtmlLibraryShutDown},
      {"mtmlLibraryInitDeviceByIndex", (void *)&resp->ch.mtmlLibraryInitDeviceByIndex},
      {"mtmlDeviceInitMemory", (void *)&resp->ch.mtmlDeviceInitMemory},
      {"mtmlDeviceFreeMemory", (void *)&resp->ch.mtmlDeviceFreeMemory},
      {"mtmlMemoryGetTotal", (void *)&resp->ch.mtmlMemoryGetTotal},
      {"mtmlMemoryGetUsed", (void *)&resp->ch.mtmlMemoryGetUsed},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(mtml_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", mtml_lib_path, msg);
    snprintf(buf, buflen,
             "Unable to load %s library to query for Moore Threads GPUs: %s",
             mtml_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  // LOG(resp->ch.verbose, "wiring moore threads management library functions in %s\n", mtml_lib_path);

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

  ret = (*resp->ch.mtmlLibraryInit)(&resp->ch.lib);
  if (ret != MTML_SUCCESS) {
    LOG(resp->ch.verbose, "mtmlLibraryInit err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "mtml library init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  ret = (*resp->ch.mtmlLibraryInitSystem)(resp->ch.lib, &resp->ch.sys);
  if (ret != MTML_SUCCESS) {
    LOG(resp->ch.verbose, "mtmlLibraryInitSystem err: %d\n", ret);
    ret = resp->ch.mtmlLibraryShutDown(resp->ch.lib);
    if (ret != MTML_SUCCESS) {
      LOG(1, "error during mtmlLibraryShutDown %d", ret);
    }
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "mtml library system init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }
}


void mtml_get_free(mtml_handle_t h, int device_id, uint64_t *free, uint64_t *total, uint64_t *used) {
    MtmlDevice *device;
    MtmlMemory *memory;
    mtmlMemory_t memInfo = {0};
    MtmlReturn ret;
    ret = (*h.mtmlLibraryInitDeviceByIndex)(h.lib, device_id, &device);
    if (ret != MTML_SUCCESS) {
        LOG(1, "unable to get device handle %d: %d", device_id, ret);
        *free = 0;
        return;
    }

    ret = (*h.mtmlDeviceInitMemory)(device, &memory);
    if (ret != MTML_SUCCESS) {
        LOG(1, "unable to get memory handle for device %d: %d", device_id, ret);
        *free = 0;
        return;
    }

    ret = (*h.mtmlMemoryGetTotal)(memory, &memInfo.total);
    if (ret != MTML_SUCCESS) {
        LOG(1, "unable to get total memory for device %d: %d", device_id, ret);
        *free = 0;
        return;
    }
    ret = (*h.mtmlMemoryGetUsed)(memory, &memInfo.used);
    if (ret != MTML_SUCCESS) {
        LOG(1, "unable to get used memory for device %d: %d", device_id, ret);
        *free = 0;
        return;
    }
    *total = memInfo.total;
    *used = memInfo.used;
    *free = *total - *used;

    ret = (*h.mtmlDeviceFreeMemory)(memory);
    if (ret != MTML_SUCCESS) {
        LOG(1, "unable to free memory handle for device %d: %d", device_id, ret);
    }
}


void mtml_release(mtml_handle_t h) {
  LOG(h.verbose, "releasing mtml library\n");
  MtmlReturn ret;
  ret = (*h.mtmlLibraryShutDown)(h.lib);
  if (ret != MTML_SUCCESS) {
    LOG(1, "error during mtmlLibraryShutDown %d", ret);
  }
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

#endif  // __APPLE__