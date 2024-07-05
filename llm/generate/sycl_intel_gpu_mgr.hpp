#include <sycl/sycl.hpp>
#include <map>
using namespace sycl;

#define SYCL_MAX_CHAR_BUF_SIZE 256

struct gpu_meta_info {
  std::string name;
  std::string arch;
  std::string codename;
};

static std::map<uint32_t, gpu_meta_info> supported_gpus;

class env_initer {
 public:
  env_initer() {
    setenv("ZES_ENABLE_SYSMAN", "1", 1);
    // TODO(zhe): leave the map-init related reference here.
    supported_gpus[0x5690] = {"Intel® Arc™ A770M Graphics", "Xe-HPG", "Alchemist"};
  }
};
static env_initer initer;

extern "C" {
struct dev_info {
  char vendor_name[SYCL_MAX_CHAR_BUF_SIZE];
  char device_name[SYCL_MAX_CHAR_BUF_SIZE];
  uint32_t device_id;
};

struct runtime_info {
  char driver_version[SYCL_MAX_CHAR_BUF_SIZE];
  int level_zero_idx;
  uint64_t global_mem_size;
  uint64_t free_mem;
};

struct gpu_info {
  dev_info dev;
  runtime_info runtime;
};

int get_device_num();
void get_dev_info(int dev_idx, gpu_info* info);
}