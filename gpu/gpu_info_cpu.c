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
    resp->major = 0;
    resp->minor = 0;
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
    resp->major = 0;
    resp->minor = 0;
    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "0");
  }
  return;
}

#elif __APPLE__
// TODO consider an Apple implementation that does something useful
// mem_info_t cpu_check_ram() {
//   mem_info_t resp = {0, 0, NULL};
//   return resp;
// }
#else
#error "Unsupported platform"
#endif
