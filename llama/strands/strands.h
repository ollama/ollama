#pragma once
#include "json.h"
#include <memory>
#include <string>
#include <vector>

class strands_head {
  public:
    explicit strands_head(const std::string &path);
    ~strands_head();
    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &hidden,
                                          const common_json &fields);
  private:
    struct impl;
    std::unique_ptr<impl> p;
};
