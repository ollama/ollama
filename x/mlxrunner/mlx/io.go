package mlx

// #include "generated.h"
import "C"

import (
	"iter"
	"log/slog"
	"maps"
	"slices"
	"unsafe"

	"github.com/ollama/ollama/types/model"
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

func LoadAll(root *model.Root, pattern string, states map[string]*Array, afterLoadFuncs []func(*model.Root) error) error {
	matches, err := root.Glob(pattern)
	if err != nil {
		return err
	}

	weights := make(map[string]*Array)
	for match := range matches {
		slog.Debug("Loading weights from", "file", match)
		maps.Copy(weights, maps.Collect(Load(root.JoinPath("blobs", match))))
	}

	var numBytes int
	for name, weight := range states {
		if _, ok := weights[name]; ok {
			slog.Debug("Loading weight", "name", name, "weight", weight)
			*weight = *weights[name]
			numBytes += weight.NumBytes()
		}
	}

	for _, afterLoadFunc := range afterLoadFuncs {
		if err := afterLoadFunc(root); err != nil {
			return err
		}
	}

	Eval(slices.Collect(maps.Values(states))...)
	ClearCache()
	slog.Info("Loaded weights", "count", len(states), "num_bytes", PrettyBytes(numBytes), "memory", Memory{})
	return nil
}

func UnloadAll(states map[string]*Array) {
	weights := slices.Collect(maps.Values(states))
	for _, weight := range weights {
		weight.desc.numRefs = 0
	}

	numBytes := Free(weights...)
	slog.Info("Unloaded weights", "count", len(states), "num_bytes", PrettyBytes(numBytes), "memory", Memory{})
}
