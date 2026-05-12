// mlx_error_handler.c - Safe error handling for MLX initialization
// Provides a non-fatal error handler for use during init(), so that
// GPU failures are captured instead of calling exit(-1).

#include "mlx_error_handler.h"
#include "mlx.h"
#include <string.h>

static char mlx_init_error_msg[1024] = {0};
static int  mlx_init_error_flag = 0;

// Error handler that silently stores the error message.
// The error is surfaced on the Go side via mlxInitError / GetMLXInitError()
// only when MLX is actually needed.
static void mlx_silent_error_handler(const char* msg, void* data) {
    (void)data;
    strncpy(mlx_init_error_msg, msg, sizeof(mlx_init_error_msg) - 1);
    mlx_init_error_msg[sizeof(mlx_init_error_msg) - 1] = '\0';
    mlx_init_error_flag = 1;
}

void mlx_set_safe_init_mode(void) {
    mlx_init_error_flag = 0;
    mlx_init_error_msg[0] = '\0';
    mlx_set_error_handler(mlx_silent_error_handler, NULL, NULL);
}

void mlx_set_default_error_mode(void) {
    mlx_set_error_handler(NULL, NULL, NULL);
}

int mlx_had_init_error(void) {
    return mlx_init_error_flag;
}

const char* mlx_get_init_error(void) {
    return mlx_init_error_flag ? mlx_init_error_msg : NULL;
}
