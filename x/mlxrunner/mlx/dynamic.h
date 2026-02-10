#ifndef MLX_DYNAMIC_H
#define MLX_DYNAMIC_H

#ifdef _WIN32
#include <windows.h>
#define DLSYM(handle, symbol) GetProcAddress((HMODULE)(handle), symbol)
#else
#include <dlfcn.h>
#define DLSYM(handle, symbol) dlsym(handle.ctx, symbol)
#endif

#include <stdint.h>

// Provide fallback typedefs for float16_t and bfloat16_t on non-ARM64
// platforms where arm_fp16.h and arm_bf16.h are not available. These are
// only used as function pointer signature placeholders since MLX requires
// Apple Silicon at runtime.
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
typedef uint16_t float16_t;
#endif

#if !defined(__aarch64__) && !defined(__ARM_FEATURE_BF16)
typedef uint16_t bfloat16_t;
#endif

#define ERROR(fmt, ...) fprintf(stderr, "%s %s - ERROR - %s:%d - " fmt "\n", __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__); return 1
#define CHECK(x) if (!(x)) { ERROR("CHECK failed: " #x); }
#define CHECK_LOAD(handle, x) x##_ = DLSYM(handle, #x); CHECK(x##_)

typedef struct {
    void* ctx;
} mlx_dynamic_handle;

int mlx_dynamic_load(
    mlx_dynamic_handle* handle,
    const char *path);

void mlx_dynamic_unload(
    mlx_dynamic_handle* handle);

#endif // MLX_DYNAMIC_H
