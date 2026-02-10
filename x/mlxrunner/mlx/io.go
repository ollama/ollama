//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"iter"
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

		cpu := C.mlx_default_cpu_stream_new()
		defer C.mlx_stream_free(cpu)

		C.mlx_load_safetensors(&string2array, &string2string, cPath, cpu)

		it := C.mlx_map_string_to_array_iterator_new(string2array)
		defer C.mlx_map_string_to_array_iterator_free(it)

		for {
			var key *C.char
			value := C.mlx_array_new()
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}

			name := C.GoString(key)
			if !yield(name, &Array{ctx: value, desc: tensorDesc{name: name, numRefs: 1000}}) {
				break
			}
		}
	}
}
