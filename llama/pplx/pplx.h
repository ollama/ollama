#pragma once
#include <string>
#include <vector>

// Perplexity's 255-option linear readout, evaluated after the Qwen backbone.
class pplx_head {
  public:
    explicit pplx_head(const std::string & path);
    std::vector<float> score(const std::vector<float> & hidden, int options) const;

  private:
    int width;
    float temperature;
    std::vector<float> weights;
};
