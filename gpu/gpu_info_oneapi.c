#ifndef __APPLE__

#include <string.h>
#include "gpu_info_oneapi.h"

void oneapi_init(char* oneapi_lib_path, oneapi_init_resp_t* resp) {
  int i, d;
  const int buflen = 256;
  char buf[buflen + 1];
  struct lookup {
    char* s;
    void** p;
  } l[] = {
      {"get_device_num", (void*)&resp->oh.get_device_num},
      {"get_dev_info", (void*)&resp->oh.get_dev_info},
      {NULL, NULL},
  };
  resp->oh.handle = LOAD_LIBRARY(oneapi_lib_path, RTLD_LAZY);
  if (!resp->oh.handle) {
    char* msg = LOAD_ERR();
    snprintf(buf, buflen, "Unable to load library to query for Intel GPUs: %s\n", msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; l[i].s != NULL; i++) {
    LOG(resp->oh.verbose, "dlsym: %s\n", l[i].s);
    *l[i].p = LOAD_SYMBOL(resp->oh.handle, l[i].s);
    if (!*(l[i].p)) {
      resp->oh.handle = NULL;
      char* msg = LOAD_ERR();
      LOG(resp->oh.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->oh.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s, msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }
  return;
}

void oneapi_check_dev(oneapi_handle_t h, int dev_idx, struct intel_gpu_info* resp) { (*h.get_dev_info)(dev_idx, resp); }

int oneapi_get_device_count(oneapi_handle_t h) { return (*h.get_device_num)(); }

#endif  // __APPLE__
