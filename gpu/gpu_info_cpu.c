#include "gpu_info.h"
// Fallbacks for CPU mode

#ifdef _WIN32
#include <sysinfoapi.h>
void cpu_check_ram(mem_info_t *resp) {
  resp->err = NULL;
  MEMORYSTATUSEX info;
  info.dwLength = sizeof(info);
  if (GlobalMemoryStatusEx(&info) != 0) {
    resp->total = info.ullTotalPhys;
    resp->free = info.ullAvailPhys;
    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "0");
  } else {
    resp->err = LOAD_ERR();
  }
  return;
}

#elif __linux__
#include <errno.h>
#include <string.h>
#include <sys/sysinfo.h>
void cpu_check_ram(mem_info_t *resp) {
  struct sysinfo info;
  resp->err = NULL;
  if (sysinfo(&info) != 0) {
    resp->err = strdup(strerror(errno));
  } else {
    resp->total = info.totalram * info.mem_unit;
    resp->free = info.freeram * info.mem_unit;
    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "0");
  }
  return;
}

#elif __APPLE__
// Unused - see gpu_darwin.go
#else
#error "Unsupported platform"
#endif
