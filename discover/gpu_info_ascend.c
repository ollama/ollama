#ifndef __APPLE__

#include "gpu_info_ascend.h"

#include <string.h>


void ascend_init(char *ascend_lib_path, ascend_init_resp_t *resp)
{
  aclError ret = -1;
  resp->err = NULL;
  resp->num_devices = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  struct lookup
  {
    char *s;
    void **p;
  } l[] = {
      {"aclInit", (void *)&resp->ah.aclInit},
      {"aclFinalize", (void *)&resp->ah.aclFinalize},
      {"aclrtSetDevice", (void *)&resp->ah.aclrtSetDevice},
      {"aclrtResetDevice", (void *)&resp->ah.aclrtResetDevice},
      {"aclrtGetVersion", (void *)&resp->ah.aclrtGetVersion},
      {"aclrtGetDeviceCount", (void *)&resp->ah.aclrtGetDeviceCount},
      {"aclrtQueryDeviceStatus", (void *)&resp->ah.aclrtQueryDeviceStatus},
      {"aclrtGetMemInfo", (void *)&resp->ah.aclrtGetMemInfo},
      {"aclrtGetSocName", (void *)&resp->ah.aclrtGetSocName},
      {"aclGetRecentErrMsg", (void *)&resp->ah.aclGetRecentErrMsg},
      {NULL, NULL},
  };

  resp->ah.handle = LOAD_LIBRARY(ascend_lib_path, RTLD_LAZY);
  if (!resp->ah.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ah.verbose, "library %s load err: %s\n", ascend_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for ascend GPUs: %s",
            ascend_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->ah.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->ah.handle, l[i].s);
    if (!*(l[i].p)) {
      resp->ah.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->ah.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ah.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s, msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }
  LOG(resp->ah.verbose, "calling aclInit\n");
  ret = (*resp->ah.aclInit)(NULL);
  // A process must only call the aclInit function once.
  // If the init function is called, the error will be ignored.
  if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE ) {
    LOG(resp->ah.verbose, "aclInit err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ah.handle);
    resp->ah.handle = NULL;
    snprintf(buf, buflen, "ascend init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  int32_t majorVersion;
  int32_t minorVersion;
  int32_t patchVersion;
  resp->ah.driver_major = 0;
  resp->ah.driver_minor = 0;

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->ah.aclrtGetVersion)(&majorVersion, &minorVersion, &patchVersion);
  if (ret != ACL_SUCCESS) {
    LOG(resp->ah.verbose, "aclrtGetVersion failed: %d\n", ret);
  } else {
    resp->ah.driver_major = majorVersion;
    resp->ah.driver_minor = minorVersion;
    LOG(resp->ah.verbose, "ascend driver version: %d.%d\n", resp->ah.driver_major, resp->ah.driver_minor);
  }

  ret = (*resp->ah.aclrtGetDeviceCount)(&resp->num_devices);
  if (ret != ACL_SUCCESS) {
    LOG(resp->ah.verbose, "aclrtGetDeviceCount err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ah.handle);
    resp->ah.handle = NULL;
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }
}

void ascend_bootstrap(ascend_handle_t h, int device_id, mem_info_t *resp) {
    resp->err = NULL;
    aclError aclRet;
    const int buflen = 256;
    char buf[buflen + 1];

    if (h.handle == NULL) {
        resp->err = strdup("ascend handle isn't initialized");
        return; 
    }

    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", device_id);

    aclRet = (*h.aclrtSetDevice)(device_id);
    if (aclRet != ACL_SUCCESS) {
        snprintf(buf, buflen, "ascend device failed to set: %u\n", device_id);
        resp->err = strdup(buf);
        return;
    }

    aclrtDeviceStatus device_status;
    aclRet = (*h.aclrtQueryDeviceStatus)(device_id, &device_status);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtQueryDeviceStatus %u fail with %d.\n", device_id, aclRet);
        (*h.aclFinalize)();
        return;
    }

    if (device_status != ACL_RT_DEVICE_STATUS_NORMAL) {
        printf("invalid device %u status: %d", device_id, device_status);
        (*h.aclFinalize)();
        return;
    }

    const char *soc_version = (*h.aclrtGetSocName)();
    char soc_name[11] = {0};
    strncpy(soc_name, soc_version, 10);
    snprintf(&resp->gpu_name[0], GPU_NAME_LEN, "%s", soc_name);

    size_t free = 0;
    size_t total = 0;
    aclRet = (*h.aclrtGetMemInfo)(ACL_DDR_MEM, &free, &total);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtGetMemInfo to DDR failed: %u\n", device_id);
        return;
    }
    resp->free += free;
    resp->total += total;

    aclRet = (*h.aclrtGetMemInfo)(ACL_HBM_MEM, &free, &total);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtGetMemInfo to HBM failed: %u\n", device_id);
        return;
    }
    resp->free += free;
    resp->total += total;

    aclRet = (*h.aclrtResetDevice)(device_id);
    if (aclRet != ACL_SUCCESS) {
        printf("aclrtResetDevice failed: %u\n", device_id);
        return;
    }
}

void ascend_release(ascend_handle_t h) {
  int d;
  LOG(h.verbose, "releasing ascned library\n");
  aclError ret;
  ret = (*h.aclFinalize)();
  if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_FINALIZE) {
    LOG(1, "error during aclFinalize %d", ret);
  }
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

#endif // __APPLE__