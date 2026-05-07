#include "sycl_hw.hpp"

// TODO: currently not used
/*
sycl_hw_info get_device_hw_info(sycl::device *device_ptr) {
  sycl_hw_info res;
  int32_t id = device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
  res.device_id = id;

  syclex::architecture arch = device_ptr->get_info<syclex::info::device::architecture>();
  res.arch = arch;

  return res;
}
*/
