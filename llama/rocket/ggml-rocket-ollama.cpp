// SPDX-License-Identifier: MIT
//
// Ollama wrapper around upstream ggml-rocket.cpp that probes for an available
// Rockchip NPU device (`rocket` or `rknpu`) in get_device_count() so non-Rockchip
// hosts report 0 devices without opening/failing device nodes at compute time.

// GCC < 13 exposes `_Float16` in C mode (`rocket_matmul.c`), but in C++17/20 mode
// only exposes `__fp16` on ARM/AArch64 (or requires a 16-bit storage typedef).
// `rocket_matmul.h` provides `ROCKET_FLOAT16_T` for C++ callers whose compiler lacks
// `_Float16` in C++ mode.
#if defined(__cplusplus) && !defined(ROCKET_FLOAT16_T)
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)
#if defined(__aarch64__) || defined(__arm__)
#define ROCKET_FLOAT16_T __fp16
#else
#include <cstdint>
#define ROCKET_FLOAT16_T uint16_t
#endif
#endif
#endif

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
