#ifndef __APPLE__

#include <string.h>
#include "gpu_info_mtmusa.h"

void mtmusa_init(char *mtmusa_lib_path, mtmusa_init_resp_t *resp) {
  MUresult ret;
  resp->err = NULL;
  resp->num_devices = 0;
  resp->musaErr = MUSA_SUCCESS;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {

      {"muInit", (void *)&resp->ch.muInit},
      {"muDriverGetVersion", (void *)&resp->ch.muDriverGetVersion},
      {"muDeviceGetCount", (void *)&resp->ch.muDeviceGetCount},
      {"muDeviceGet", (void *)&resp->ch.muDeviceGet},
      {"muDeviceGetAttribute", (void *)&resp->ch.muDeviceGetAttribute},
      {"muDeviceGetUuid_v2", (void *)&resp->ch.muDeviceGetUuid_v2},
      {"muDeviceGetName", (void *)&resp->ch.muDeviceGetName},
      {"muCtxCreate_v2", (void *)&resp->ch.muCtxCreate_v2},
      {"muMemGetInfo_v2", (void *)&resp->ch.muMemGetInfo_v2},
      {"muCtxDestroy_v2", (void *)&resp->ch.muCtxDestroy_v2},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(mtmusa_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", mtmusa_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for Moore Threads GPUs: %s",
            mtmusa_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    resp->musaErr = -1;
    return;
  }

  for (i = 0; l[i].s != NULL; i++) {
    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!*(l[i].p)) {
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ch.handle);
      resp->ch.handle = NULL;
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
              msg);
      free(msg);
      resp->err = strdup(buf);
      resp->musaErr = -1;
      return;
    }
  }

  ret = (*resp->ch.muInit)(0);
  if (ret != MUSA_SUCCESS) {
    LOG(resp->ch.verbose, "muInit err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "musa driver library init failure: %d", ret);
    resp->err = strdup(buf);
    resp->musaErr = ret;
    return;
  }

  int version = 0;
  resp->ch.driver_major = 0;
  resp->ch.driver_minor = 0;

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->ch.muDriverGetVersion)(&version);
  if (ret != MUSA_SUCCESS) {
    LOG(resp->ch.verbose, "muDriverGetVersion failed: %d\n", ret);
  } else {
    resp->ch.driver_major = version / 1000;
    resp->ch.driver_minor = (version - (resp->ch.driver_major * 1000)) / 10;
    LOG(resp->ch.verbose, "MUSA driver version: %d.%d\n", resp->ch.driver_major, resp->ch.driver_minor);
  }

  ret = (*resp->ch.muDeviceGetCount)(&resp->num_devices);
  if (ret != MUSA_SUCCESS) {
    LOG(resp->ch.verbose, "muDeviceGetCount err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    resp->musaErr = ret;
    return;
  }
}

void mtmusa_bootstrap(mtmusa_handle_t h, int i, mem_info_t *resp) {
  resp->err = NULL;
  mtmusaMemory_t memInfo = {0,0};
  MUresult ret;
  MUdevice device = -1;
  MUcontext ctx = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  MUuuid uuid = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  if (h.handle == NULL) {
    resp->err = strdup("musa driver library handle isn't initialized");
    return;
  }

  ret = (*h.muDeviceGet)(&device, i);
  if (ret != MUSA_SUCCESS) {
    snprintf(buf, buflen, "musa driver library device failed to initialize");
    resp->err = strdup(buf);
    return;
  }

  int major = 0;
  int minor = 0;
  ret = (*h.muDeviceGetAttribute)(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (ret != MUSA_SUCCESS) {
    LOG(h.verbose, "[%d] device major lookup failure: %d\n", i, ret);
  } else {
    ret = (*h.muDeviceGetAttribute)(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (ret != MUSA_SUCCESS) {
      LOG(h.verbose, "[%d] device minor lookup failure: %d\n", i, ret);
    } else {
      resp->minor = minor;
      resp->major = major;
    }
  }

  ret = (*h.muDeviceGetUuid_v2)(&uuid, device);
  if (ret != MUSA_SUCCESS) {
    LOG(h.verbose, "[%d] device uuid lookup failure: %d\n", i, ret);
    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
  } else {
    // GPU-d110a105-ac29-1d54-7b49-9c90440f215b
    snprintf(&resp->gpu_id[0], GPU_ID_LEN,
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        uuid.bytes[0],
        uuid.bytes[1],
        uuid.bytes[2],
        uuid.bytes[3],
        uuid.bytes[4],
        uuid.bytes[5],
        uuid.bytes[6],
        uuid.bytes[7],
        uuid.bytes[8],
        uuid.bytes[9],
        uuid.bytes[10],
        uuid.bytes[11],
        uuid.bytes[12],
        uuid.bytes[13],
        uuid.bytes[14],
        uuid.bytes[15]
      );
  }

  ret = (*h.muDeviceGetName)(&resp->gpu_name[0], GPU_NAME_LEN, device);
  if (ret != MUSA_SUCCESS) {
    LOG(h.verbose, "[%d] device name lookup failure: %d\n", i, ret);
    resp->gpu_name[0] = '\0';
  }

  // To get memory we have to set (and release) a context
  ret = (*h.muCtxCreate_v2)(&ctx, NULL, 0, 0, device);
  if (ret != MUSA_SUCCESS) {
    snprintf(buf, buflen, "musa driver library failed to get device context %d", ret);
    resp->err = strdup(buf);
    return;
  }

  ret = (*h.muMemGetInfo_v2)(&memInfo.free, &memInfo.total);
  if (ret != MUSA_SUCCESS) {
    snprintf(buf, buflen, "musa driver library device memory info lookup failure %d", ret);
    resp->err = strdup(buf);
    // Best effort on failure...
    (*h.muCtxDestroy_v2)(ctx);
    return;
  }

  resp->total = memInfo.total;
  resp->free = memInfo.free;

  LOG(h.verbose, "[%s] MUSA totalMem %lu mb\n", resp->gpu_id, resp->total / 1024 / 1024);
  LOG(h.verbose, "[%s] MUSA freeMem %lu mb\n", resp->gpu_id, resp->free / 1024 / 1024);
  LOG(h.verbose, "[%s] Compute Capability %d.%d\n", resp->gpu_id, resp->major, resp->minor);



  ret = (*h.muCtxDestroy_v2)(ctx);
  if (ret != MUSA_SUCCESS) {
    LOG(1, "musa driver library failed to release device context %d", ret);
  }
}

void mtmusa_get_free(mtmusa_handle_t h, int i, uint64_t *free, uint64_t *total) {
  MUresult ret;
  MUcontext ctx = NULL;
  MUdevice device = -1;
  *free = 0;
  *total = 0;

  ret = (*h.muDeviceGet)(&device, i);
  if (ret != MUSA_SUCCESS) {
    LOG(1, "musa driver library device failed to initialize");
    return;
  }


  // To get memory we have to set (and release) a context
  ret = (*h.muCtxCreate_v2)(&ctx, NULL, 0, 0, device);
  if (ret != MUSA_SUCCESS) {
    LOG(1, "musa driver library failed to get device context %d", ret);
    return;
  }

  ret = (*h.muMemGetInfo_v2)(free, total);
  if (ret != MUSA_SUCCESS) {
    LOG(1, "musa driver library device memory info lookup failure %d", ret);
    // Best effort on failure...
    (*h.muCtxDestroy_v2)(ctx);
    return;
  }

  ret = (*h.muCtxDestroy_v2)(ctx);
  if (ret != MUSA_SUCCESS) {
    LOG(1, "musa driver library failed to release device context %d", ret);
  }
}

void mtmusa_release(mtmusa_handle_t h) {
  LOG(h.verbose, "releasing musa driver library\n");
  UNLOAD_LIBRARY(h.handle);
  // TODO and other context release logic?
  h.handle = NULL;
}

#endif  // __APPLE__