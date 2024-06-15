#include "gpu_info.h"

#ifdef __linux__
#include <sys/capability.h>
#endif

typedef VkInstance vk_handle_t;

typedef struct vk_init_resp
{
  char *err; // If err is non-null handle is invalid
  int num_devices;
  vk_handle_t oh;
} vk_init_resp_t;

void vk_init(char* vk_lib_path, char* cap_lib_path, vk_init_resp_t *resp);
void vk_check_vram(vk_handle_t rh, int i, mem_info_t *resp);
void vk_free(vk_handle_t rh);
