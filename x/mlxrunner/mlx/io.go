package mlx

// #include "generated.h"
import "C"

import (
	"fmt"
	"iter"
	"runtime"
	"unsafe"
)

// SafetensorsFile represents a loaded safetensors file.
type SafetensorsFile struct {
	arrays   C.mlx_map_string_to_array
	metadata C.mlx_map_string_to_string
}

func loadSafetensorsStream() C.mlx_stream {
	if runtime.GOOS == "darwin" {
		return C.mlx_default_cpu_stream_new()
	}
	return C.mlx_default_gpu_stream_new()
}

// LoadSafetensorsNative loads a safetensors file using MLX's native loader.
func LoadSafetensorsNative(path string) (*SafetensorsFile, error) {
	var arrays C.mlx_map_string_to_array
	var metadata C.mlx_map_string_to_string

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	stream := loadSafetensorsStream()
	defer C.mlx_stream_free(stream)

	if C.mlx_load_safetensors(&arrays, &metadata, cPath, stream) != 0 {
		return nil, fmt.Errorf("failed to load safetensors: %s", path)
	}

	return &SafetensorsFile{arrays: arrays, metadata: metadata}, nil
}

// Get retrieves a tensor by name.
func (s *SafetensorsFile) Get(name string) *Array {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	value := C.mlx_array_new()
	if C.mlx_map_string_to_array_get(&value, s.arrays, cName) != 0 {
		return nil
	}
	if value.ctx == nil {
		return nil
	}

	arr := New(name)
	arr.ctx = value
	return arr
}

// GetMetadata retrieves a metadata value by key.
func (s *SafetensorsFile) GetMetadata(key string) string {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var cValue *C.char
	if C.mlx_map_string_to_string_get(&cValue, s.metadata, cKey) != 0 {
		return ""
	}
	return C.GoString(cValue)
}

// Free releases the loaded safetensors maps.
func (s *SafetensorsFile) Free() {
	if s == nil {
		return
	}
	C.mlx_map_string_to_array_free(s.arrays)
	C.mlx_map_string_to_string_free(s.metadata)
}

func Load(path string) iter.Seq2[string, *Array] {
	return func(yield func(string, *Array) bool) {
		sf, err := LoadSafetensorsNative(path)
		if err != nil {
			return
		}
		defer sf.Free()

		it := C.mlx_map_string_to_array_iterator_new(sf.arrays)
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

// SaveSafetensors saves arrays to a safetensors file without metadata.
func SaveSafetensors(path string, arrays map[string]*Array) error {
	return SaveSafetensorsWithMetadata(path, arrays, nil)
}

// SaveSafetensorsWithMetadata saves arrays to a safetensors file with metadata.
func SaveSafetensorsWithMetadata(path string, arrays map[string]*Array, metadata map[string]string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cArrays := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(cArrays)

	for name, arr := range arrays {
		if arr == nil {
			continue
		}
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(cArrays, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	cMetadata := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(cMetadata)

	for key, value := range metadata {
		cKey := C.CString(key)
		cValue := C.CString(value)
		C.mlx_map_string_to_string_insert(cMetadata, cKey, cValue)
		C.free(unsafe.Pointer(cKey))
		C.free(unsafe.Pointer(cValue))
	}

	if C.mlx_save_safetensors(cPath, cArrays, cMetadata) != 0 {
		return fmt.Errorf("failed to save safetensors: %s", path)
	}

	return nil
}
