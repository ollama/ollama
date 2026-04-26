// SPDX-License-Identifier: MIT
// level_zero_info.c — runtime shim that dlopen-loads libggml-level-zero and
// resolves the 8 ze_ollama_* symbols into the local lz_* API.
// Never statically linked to ze_loader or any Intel SDK library.
// On platforms where dlopen is unavailable (darwin) the file is excluded by
// the //go:build !darwin tag on gpu_level_zero.go.

#include "level_zero_info.h"

#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE dl_handle_t;
#define DL_OPEN(name)          LoadLibraryA(name)
#define DL_SYM(h, sym)        ((void*)GetProcAddress((h), (sym)))
#define DL_CLOSE(h)            FreeLibrary(h)
#else
#include <dlfcn.h>
typedef void *dl_handle_t;
#define DL_OPEN(name)          dlopen((name), RTLD_NOW | RTLD_LOCAL)
#define DL_SYM(h, sym)        dlsym((h), (sym))
#define DL_CLOSE(h)            dlclose(h)
#endif

// ---------------------------------------------------------------------------
// Function pointer typedefs matching ze_ollama.h signatures.
// Using the opaque pointer cast trick so we never include ze_ollama.h here.
// ---------------------------------------------------------------------------
typedef int  (*fn_init_t)(void);
typedef int  (*fn_enumerate_t)(void *out_buf, size_t buf_cap, size_t *out_count);
typedef int  (*fn_device_open_t)(uint32_t index, void **out);
typedef void (*fn_device_close_t)(void *handle);
typedef int  (*fn_device_free_mem_t)(void *handle, uint64_t *out_bytes);
typedef const char *(*fn_result_str_t)(int result);
typedef const char *(*fn_version_t)(void);
typedef int  (*fn_shutdown_t)(void);

static dl_handle_t        g_handle          = NULL;
static fn_init_t          g_init            = NULL;
static fn_enumerate_t     g_enumerate       = NULL;
static fn_device_open_t   g_device_open     = NULL;
static fn_device_close_t  g_device_close    = NULL;
static fn_device_free_mem_t g_free_mem      = NULL;
static fn_result_str_t    g_result_str      = NULL;
static fn_version_t       g_version         = NULL;
static fn_shutdown_t      g_shutdown        = NULL;
static int                g_loaded          = 0; /* 0 = not tried, 1 = ok, -1 = missing */

// ---------------------------------------------------------------------------
// Library names tried in order.  On Linux the .so.1 soname is preferred.
// ---------------------------------------------------------------------------
#if defined(_WIN32)
static const char *const LIB_NAMES[] = {
    "ggml-level-zero.dll",
    NULL
};
#else
static const char *const LIB_NAMES[] = {
    "libggml-level-zero.so.1",
    "libggml-level-zero.so",
    NULL
};
#endif

// ---------------------------------------------------------------------------
// lz_load_library — resolves all symbols from the shared library.
// Returns LZ_OK on success, LZ_ERR_LOADER_MISSING when absent.
// Thread safety: callers must serialise access (sync.Once on Go side).
// ---------------------------------------------------------------------------
static lz_result_t lz_load_library(void) {
    if (g_loaded == 1) { return LZ_OK; }
    if (g_loaded == -1) { return LZ_ERR_LOADER_MISSING; }

    for (int i = 0; LIB_NAMES[i] != NULL; i++) {
        g_handle = DL_OPEN(LIB_NAMES[i]);
        if (g_handle) { break; }
    }
    if (!g_handle) {
        g_loaded = -1;
        return LZ_ERR_LOADER_MISSING;
    }

#define LOAD_SYM(fp, name) \
    do { \
        (fp) = (void*)DL_SYM(g_handle, (name)); \
        if (!(fp)) { DL_CLOSE(g_handle); g_handle = NULL; g_loaded = -1; return LZ_ERR_LOADER_MISSING; } \
    } while (0)

    LOAD_SYM(g_init,        "ze_ollama_init");
    LOAD_SYM(g_enumerate,   "ze_ollama_enumerate_devices");
    LOAD_SYM(g_device_open, "ze_ollama_device_open");
    LOAD_SYM(g_device_close,"ze_ollama_device_close");
    LOAD_SYM(g_free_mem,    "ze_ollama_device_free_memory");
    LOAD_SYM(g_result_str,  "ze_ollama_result_str");
    LOAD_SYM(g_version,     "ze_ollama_version");
    LOAD_SYM(g_shutdown,    "ze_ollama_shutdown");

#undef LOAD_SYM

    g_loaded = 1;
    return LZ_OK;
}

// ---------------------------------------------------------------------------
// Public shim implementations
// ---------------------------------------------------------------------------

lz_result_t lz_init(void) {
    lz_result_t r = lz_load_library();
    if (r != LZ_OK) { return r; }
    return (lz_result_t)g_init();
}

lz_result_t lz_enumerate_devices(
    lz_device_info_t *out_buf,
    size_t            buf_cap,
    size_t           *out_count)
{
    if (g_loaded != 1) { return LZ_ERR_LOADER_MISSING; }
    return (lz_result_t)g_enumerate((void *)out_buf, buf_cap, out_count);
}

lz_result_t lz_device_open(
    uint32_t           index,
    lz_device_handle_t *out)
{
    if (g_loaded != 1) { return LZ_ERR_LOADER_MISSING; }
    return (lz_result_t)g_device_open(index, (void **)out);
}

void lz_device_close(lz_device_handle_t handle) {
    if (g_loaded != 1) { return; }
    g_device_close((void *)handle);
}

lz_result_t lz_device_free_memory(
    lz_device_handle_t handle,
    uint64_t          *out_bytes)
{
    if (g_loaded != 1) { return LZ_ERR_LOADER_MISSING; }
    return (lz_result_t)g_free_mem((void *)handle, out_bytes);
}

const char *lz_result_str(lz_result_t result) {
    if (g_loaded != 1) { return "loader-missing"; }
    return g_result_str((int)result);
}

const char *lz_version(void) {
    if (g_loaded != 1) { return "0.0.0-loader-missing"; }
    return g_version();
}

lz_result_t lz_shutdown(void) {
    if (g_loaded != 1) { return LZ_OK; }
    lz_result_t r = (lz_result_t)g_shutdown();
    if (g_handle) { DL_CLOSE(g_handle); g_handle = NULL; }
    g_loaded  = 0;
    g_init        = NULL;
    g_enumerate   = NULL;
    g_device_open = NULL;
    g_device_close = NULL;
    g_free_mem    = NULL;
    g_result_str  = NULL;
    g_version     = NULL;
    g_shutdown    = NULL;
    return r;
}
