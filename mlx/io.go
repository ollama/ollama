package mlx

// #include "generated.h"
import "C"

import (
	"fmt"
	"iter"
	"runtime"
	"sort"
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
	if err := mlxError(stream); err != nil {
		return nil, err
	}
	defer freeStream(stream)

	if err := mlxError(C.mlx_load_safetensors(&arrays, &metadata, cPath, stream)); err != nil {
		return nil, fmt.Errorf("failed to load safetensors %s: %w", path, err)
	}

	return &SafetensorsFile{arrays: arrays, metadata: metadata}, nil
}

// Get retrieves a tensor by name.
func (s *SafetensorsFile) Get(name string) *Array {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	value := mlxCheck(C.mlx_array_new())
	rc := C.mlx_map_string_to_array_get(&value, s.arrays, cName)
	if err := lastError(); err != nil {
		panic(err)
	}
	if rc != 0 {
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
	rc := C.mlx_map_string_to_string_get(&cValue, s.metadata, cKey)
	if err := lastError(); err != nil {
		panic(err)
	}
	if rc != 0 {
		return ""
	}
	return C.GoString(cValue)
}

// Free releases the loaded safetensors maps.
func (s *SafetensorsFile) Free() {
	if s == nil {
		return
	}
	freeArrayMap(s.arrays)
	freeStringMap(s.metadata)
}

func Load(path string) iter.Seq2[string, *Array] {
	return func(yield func(string, *Array) bool) {
		sf, err := LoadSafetensorsNative(path)
		if err != nil {
			return
		}
		defer sf.Free()

		it := mlxCheck(C.mlx_map_string_to_array_iterator_new(sf.arrays))
		defer freeArrayMapIterator(it)

		for {
			var key *C.char
			value := mlxCheck(C.mlx_array_new())
			rc := C.mlx_map_string_to_array_iterator_next(&key, &value, it)
			if err := lastError(); err != nil {
				panic(err)
			}
			if rc != 0 {
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
	if err := mlxError(cArrays); err != nil {
		return err
	}
	defer freeArrayMap(cArrays)

	arrayNames := make([]string, 0, len(arrays))
	for name, arr := range arrays {
		if arr == nil {
			continue
		}
		arrayNames = append(arrayNames, name)
	}
	sort.Strings(arrayNames)

	for _, name := range arrayNames {
		arr := arrays[name]
		cName := C.CString(name)
		err := mlxError(C.mlx_map_string_to_array_insert(cArrays, cName, arr.ctx))
		C.free(unsafe.Pointer(cName))
		if err != nil {
			return err
		}
	}

	cMetadata := C.mlx_map_string_to_string_new()
	if err := mlxError(cMetadata); err != nil {
		return err
	}
	defer freeStringMap(cMetadata)

	metadataKeys := make([]string, 0, len(metadata))
	for key := range metadata {
		metadataKeys = append(metadataKeys, key)
	}
	sort.Strings(metadataKeys)

	for _, key := range metadataKeys {
		value := metadata[key]
		cKey := C.CString(key)
		cValue := C.CString(value)
		err := mlxError(C.mlx_map_string_to_string_insert(cMetadata, cKey, cValue))
		C.free(unsafe.Pointer(cKey))
		C.free(unsafe.Pointer(cValue))
		if err != nil {
			return err
		}
	}

	if err := mlxError(C.mlx_save_safetensors(cPath, cArrays, cMetadata)); err != nil {
		return fmt.Errorf("failed to save safetensors %s: %w", path, err)
	}

	return nil
}
