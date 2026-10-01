#pragma once
#include "json.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The backbone is evaluated by llama-server. This head consumes every final
// hidden state and the token spans produced by Cloudflare's record format.
class clef_head {
  public:
    explicit clef_head(const std::string &path);
    ~clef_head();
    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &hidden,
                                          const std::vector<int32_t> &tokens, const common_json &fields);

  private:
    struct impl;
    std::unique_ptr<impl> p;
};
