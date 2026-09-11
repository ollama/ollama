// Package mlx wraps the MLX C API.
//
// MLX keeps stream and backend state in thread-locals, so all calls into this
// package must come from a single goroutine locked to its OS thread (see
// x/internal/mlxthread).
package mlx

//go:generate go run generator/main.go -output=. ./include/mlx/c/*.h

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include
// #cgo LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
// #include <string.h>
//
// static char _mlx_last_error[1024];
//
// static void _mlx_capture_error(const char* msg, void* data) {
//     (void)data;
//     strncpy(_mlx_last_error, msg, sizeof(_mlx_last_error) - 1);
// }
//
// static void mlx_install_capture_handler(void) {
//     if (mlx_set_error_handler_) {
//         mlx_set_error_handler_(_mlx_capture_error, NULL, NULL);
//     }
// }
//
// static char* mlx_last_error(void) {
//     return _mlx_last_error;
// }
import "C"

import (
	"errors"
	"fmt"
)

func init() {
	// Replace the default exit(-1) error handler with one that captures
	// the error message so we can surface it in Go.
	C.mlx_install_capture_handler()
}

var errBuf = C.mlx_last_error()

// lastError consumes the captured MLX error, or returns nil when none is
// pending.
func lastError() error {
	if *errBuf == 0 {
		return nil
	}
	err := fmt.Errorf("mlx: %s", C.GoString(errBuf))
	*errBuf = 0
	return err
}

// mlxError returns the MLX error captured by the call that produced v. mlx-c
// signals failure with a non-zero int status; a message next to a zero
// status came from an earlier unchecked call.
func mlxError[T comparable](v T) error {
	var zero T
	var failed, signaled bool
	switch any(zero).(type) {
	case C.int:
		failed, signaled = v != zero, true
	default:
		// Only an int status signals failure. Handles, pointers, sizes, and
		// dtypes are all valid at zero: a null handle is what the out-param
		// constructors return, and an empty array has no data.
	}
	if *errBuf != 0 {
		err := lastError()
		if signaled && !failed {
			return fmt.Errorf("mlx: unchecked error from an earlier call: %w", err)
		}
		return err
	}
	if failed {
		return errors.New("mlx: call failed without an error message")
	}
	return nil
}

// mlxCheck panics on a failed call and otherwise passes its result through.
// Most array operations cannot recover from a failed graph construction or
// evaluation.
func mlxCheck[T comparable](v T) T {
	if err := mlxError(v); err != nil {
		panic(err)
	}
	return v
}

// Deferred frees go through these helpers: defer evaluates a call's
// arguments immediately, so defer mlxCheck(C.mlx_..._free(v)) would
// free v on the spot and defer only the check.
func freeArray(a C.mlx_array)              { mlxCheck(C.mlx_array_free(a)) }
func freeString(s C.mlx_string)            { mlxCheck(C.mlx_string_free(s)) }
func freeVectorArray(v C.mlx_vector_array) { mlxCheck(C.mlx_vector_array_free(v)) }
func freeClosure(c C.mlx_closure)          { mlxCheck(C.mlx_closure_free(c)) }
func freeStream(s C.mlx_stream)            { mlxCheck(C.mlx_stream_free(s)) }
func freeDevice(d C.mlx_device)            { mlxCheck(C.mlx_device_free(d)) }
func freeDeviceInfo(i C.mlx_device_info)   { mlxCheck(C.mlx_device_info_free(i)) }

func freeArrayMap(m C.mlx_map_string_to_array) {
	mlxCheck(C.mlx_map_string_to_array_free(m))
}

func freeStringMap(m C.mlx_map_string_to_string) {
	mlxCheck(C.mlx_map_string_to_string_free(m))
}

func freeArrayMapIterator(it C.mlx_map_string_to_array_iterator) {
	mlxCheck(C.mlx_map_string_to_array_iterator_free(it))
}

// Version returns the MLX core library version string.
func Version() string {
	str := mlxCheck(C.mlx_string_new())
	mlxCheck(C.mlx_version(&str))
	defer freeString(str)
	return C.GoString(mlxCheck(C.mlx_string_data(str)))
}

func doEval(outputs []*Array, async bool) {
	if len(outputs) == 0 {
		return
	}

	vector := mlxCheck(C.mlx_vector_array_new())
	defer freeVectorArray(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			mlxCheck(C.mlx_vector_array_append_value(vector, output.ctx))
		}
	}

	if async {
		mlxCheck(C.mlx_async_eval(vector))
	} else {
		mlxCheck(C.mlx_eval(vector))
	}
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
	mlxCheck(C.mlx_metal_is_available(&available))
	return bool(available)
}

// CUDAIsAvailable returns true if a CUDA GPU is available.
func CUDAIsAvailable() bool {
	var available C._Bool
	mlxCheck(C.mlx_cuda_is_available(&available))
	return bool(available)
}
