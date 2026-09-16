#ifndef MLX_DYNAMIC_H
#define MLX_DYNAMIC_H

#ifdef _WIN32
#include <windows.h>
#define DLSYM(handle, symbol) (void*)GetProcAddress((HMODULE)(handle.ctx), symbol)
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

// Symbol load failures must not print to stderr here: this loader runs at
// startup in every process (including machines where MLX is not applicable),
// so a missing symbol is recorded and surfaced through the Go side (see
// CheckInit in dynamic.go) only when MLX is actually requested.
void mlx_dynamic_record_load_error(const char* symbol);
const char* mlx_dynamic_load_error(void);
#define CHECK_LOAD(handle, x) *(void**)(&x##_) = DLSYM(handle, #x); if (!(x##_)) { mlx_dynamic_record_load_error(#x); return 1; }
// OPTIONAL_LOAD: load symbol if available, leave function pointer NULL otherwise
#define OPTIONAL_LOAD(handle, x) *(void**)(&x##_) = DLSYM(handle, #x)

typedef struct {
    void* ctx;
} mlx_dynamic_handle;

int mlx_dynamic_load(
    mlx_dynamic_handle* handle,
    const char *path);

void mlx_dynamic_unload(
    mlx_dynamic_handle* handle);

#endif // MLX_DYNAMIC_H
