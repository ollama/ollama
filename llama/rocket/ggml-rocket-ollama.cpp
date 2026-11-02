// SPDX-License-Identifier: MIT
//
// Ollama wrapper around upstream ggml-rocket.cpp that probes for an available
// Rockchip NPU device (`rocket` or `rknpu`) in get_device_count() so non-Rockchip
// hosts report 0 devices without opening/failing device nodes at compute time.

#include "ggml-rocket.cpp"

extern "C" int rocket_npu_probe(void);

static size_t ggml_backend_rocket_reg_get_device_count_probed(ggml_backend_reg_t reg) {
    (void)reg;
    return rocket_npu_probe() != 0 ? 1 : 0;
}

namespace {
struct rocket_reg_probe_installer {
    rocket_reg_probe_installer() {
        ggml_backend_reg_t reg = ggml_backend_rocket_reg();
        reg->iface.get_device_count = ggml_backend_rocket_reg_get_device_count_probed;
    }
} g_rocket_reg_probe_installer;
} // namespace
