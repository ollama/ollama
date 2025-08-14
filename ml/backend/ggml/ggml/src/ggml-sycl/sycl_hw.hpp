#ifndef SYCL_HW_HPP
#define SYCL_HW_HPP

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <map>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

struct sycl_hw_info {
  syclex::architecture arch;
  int32_t device_id;
};

bool is_in_vector(std::vector<int> &vec, int item);

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);


#endif // SYCL_HW_HPP
