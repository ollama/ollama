package mlx

//go:generate go run generator/main.go -output=. ./include/mlx/c/*.h

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include
// #cgo LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
// #include <string.h>
//
// static __thread char _mlx_last_error_msg[1024] = {0};
// static __thread int  _mlx_last_error_flag = 0;
//
// static void _mlx_capture_error_handler(const char* msg, void* data) {
//     (void)data;
//     strncpy(_mlx_last_error_msg, msg, sizeof(_mlx_last_error_msg) - 1);
//     _mlx_last_error_msg[sizeof(_mlx_last_error_msg) - 1] = '\0';
//     _mlx_last_error_flag = 1;
// }
//
// static void mlx_install_capture_handler(void) {
//     if (mlx_set_error_handler_) {
//         mlx_set_error_handler_(_mlx_capture_error_handler, NULL, NULL);
//     }
// }
//
// static void mlx_clear_last_error(void) {
//     _mlx_last_error_flag = 0;
//     _mlx_last_error_msg[0] = '\0';
// }
//
// static const char* mlx_get_last_error(void) {
//     return _mlx_last_error_flag ? _mlx_last_error_msg : "";
// }
//
// static __thread int _mlx_thread_bound = 0;
//
// static void mlx_bind_current_thread(void) {
//     _mlx_thread_bound = 1;
// }
//
// static void mlx_unbind_current_thread(void) {
//     _mlx_thread_bound = 0;
// }
//
// static int mlx_is_current_thread_bound(void) {
//     return _mlx_thread_bound;
// }
import "C"

import "runtime"

func init() {
	// Replace the default exit(-1) error handler with one that captures
	// the error message so we can surface it in Go.
	C.mlx_install_capture_handler()
}

// Version returns the MLX core library version string.
func Version() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_version(&str)
	return C.GoString(C.mlx_string_data(str))
}

// BindCurrentThread marks the current locked OS thread as allowed to execute
// MLX stream-bound operations. Callers should only use this on a goroutine that
// will remain locked to the same OS thread for the lifetime of its MLX work.
func BindCurrentThread() {
	C.mlx_bind_current_thread()
}

// UnbindCurrentThread clears the MLX thread-ownership marker from the current
// OS thread.
func UnbindCurrentThread() {
	C.mlx_unbind_current_thread()
}

func requireBoundThread(op string) {
	if C.mlx_is_current_thread_bound() == 0 {
		panic("mlx: " + op + " called outside a bound MLX thread; use x/internal/mlxthread or lock and bind the current OS thread first")
	}
}

// mlxCheck locks the goroutine to its OS thread, clears the captured error
// state, calls fn, and panics with the captured message if fn returns non-zero.
// The thread lock ensures the thread-local error state is read from the same
// thread that executed the call.
func mlxCheck(fallback string, fn func() C.int) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	requireBoundThread(fallback)

	C.mlx_clear_last_error()
	if fn() != 0 {
		msg := C.GoString(C.mlx_get_last_error())
		if msg == "" {
			msg = fallback
		}
		panic("mlx: " + msg)
	}
}

func doEval(outputs []*Array, async bool) {
	if len(outputs) == 0 {
		return
	}

	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			C.mlx_vector_array_append_value(vector, output.ctx)
		}
	}

	mlxCheck("eval failed", func() C.int {
		if async {
			return C.mlx_async_eval(vector)
		}
		return C.mlx_eval(vector)
	})
}

func AsyncEval(outputs ...*Array) {
	doEval(outputs, true)
}

func Eval(outputs ...*Array) {
	doEval(outputs, false)
}

// MetalIsAvailable returns true if a Metal GPU is available.
func MetalIsAvailable() bool {
	var available C._Bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}
