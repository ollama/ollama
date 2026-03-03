package mlx

// #include "generated.h"
import "C"

import (
	"iter"
	"runtime"
	"unsafe"
)

func Load(path string) iter.Seq2[string, *Array] {
	return func(yield func(string, *Array) bool) {
		string2array := C.mlx_map_string_to_array_new()
		defer C.mlx_map_string_to_array_free(string2array)

		string2string := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_string_free(string2string)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		// Use GPU stream so tensors load directly to GPU memory (CUDA has Load::eval_gpu).
		// macOS Metal doesn't implement eval_gpu for Load, so fall back to CPU stream.
		var stream C.mlx_stream
		if runtime.GOOS == "darwin" {
			stream = C.mlx_default_cpu_stream_new()
		} else {
			stream = C.mlx_default_gpu_stream_new()
		}
		defer C.mlx_stream_free(stream)

		C.mlx_load_safetensors(&string2array, &string2string, cPath, stream)

		it := C.mlx_map_string_to_array_iterator_new(string2array)
		defer C.mlx_map_string_to_array_iterator_free(it)

		for {
			var key *C.char
			value := C.mlx_array_new()
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}

			name := C.GoString(key)
			arr := New(name)
			arr.ctx = value
			if !yield(name, arr) {
				break
			}
		}
	}
}
