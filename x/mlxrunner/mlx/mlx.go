package mlx

//go:generate go run generator/main.go -output=. ./include/mlx/c/*.h

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include
// #cgo LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
// #include <string.h>
//
// static char _mlx_last_error_msg[1024] = {0};
// static int  _mlx_last_error_flag = 0;
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
// static int mlx_had_last_error(void) {
//     return _mlx_last_error_flag;
// }
//
// static const char* mlx_get_last_error(void) {
//     return _mlx_last_error_flag ? _mlx_last_error_msg : NULL;
// }
import "C"

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

	C.mlx_clear_last_error()
	var rc C.int
	if async {
		rc = C.mlx_async_eval(vector)
	} else {
		rc = C.mlx_eval(vector)
	}
	if rc != 0 {
		msg := "mlx eval failed"
		if C.mlx_had_last_error() != 0 {
			msg = C.GoString(C.mlx_get_last_error())
		}
		panic("mlx: " + msg)
	}
}

func AsyncEval(outputs ...*Array) {
	doEval(outputs, true)
}

func Eval(outputs ...*Array) {
	doEval(outputs, false)
}
