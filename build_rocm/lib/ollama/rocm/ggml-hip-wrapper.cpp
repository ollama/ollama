// Dummy HIP wrapper library for Windows
#include <windows.h>

extern "C" {
    // Export the minimal set of functions necessary to satisfy the loader
    __declspec(dllexport) int ggml_hip_init(void) { return 0; }
    __declspec(dllexport) int ggml_hip_available(void) { return 0; }
}
