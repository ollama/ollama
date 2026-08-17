package mlx

// #include <stdbool.h>
// #include <stdint.h>
// #include <stddef.h>
// #include <stdlib.h>
// #include "generated.h"
//
// extern bool goMLXReaderIsOpen(void*);
// extern size_t goMLXReaderTell(void*);
// extern int goMLXReaderSeek(void*, int64_t, int);
// extern size_t goMLXReaderRead(void*, char*, size_t);
// extern size_t goMLXReaderReadAtOffset(void*, char*, size_t, size_t);
// extern void goMLXReaderFree(void*);
//
// static size_t go_mlx_reader_write(void* desc, const char* data, size_t n) {
// 	(void)desc;
// 	(void)data;
// 	(void)n;
// 	return 0;
// }
//
// static const char* go_mlx_reader_label(void* desc) {
// 	(void)desc;
// 	return "Go reader";
// }
//
// static mlx_io_vtable go_mlx_reader_vtable(void) {
// 	mlx_io_vtable vtable = {0};
// 	vtable.is_open = goMLXReaderIsOpen;
// 	vtable.good = goMLXReaderIsOpen;
// 	vtable.tell = goMLXReaderTell;
// 	vtable.seek = goMLXReaderSeek;
// 	vtable.read = goMLXReaderRead;
// 	vtable.read_at_offset = goMLXReaderReadAtOffset;
// 	vtable.write = go_mlx_reader_write;
// 	vtable.label = go_mlx_reader_label;
// 	vtable.free = goMLXReaderFree;
// 	return vtable;
// }
//
// static mlx_io_reader go_mlx_reader_new(void* desc) {
// 	return mlx_io_reader_new(desc, go_mlx_reader_vtable());
// }
import "C"

import (
	"errors"
	"fmt"
	"io"
	"iter"
	"runtime"
	"runtime/cgo"
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

// SafetensorsReader is the random-access source MLX reads lazily. Ownership is
// transferred to LoadSafetensors; Close is called after MLX releases its last
// reference to the reader.
type SafetensorsReader interface {
	io.ReaderAt
	Size() int64
	Close() error
}

// LoadSafetensors loads a safetensors file through MLX's reader API.
func LoadSafetensors(reader SafetensorsReader) (*SafetensorsFile, error) {
	var arrays C.mlx_map_string_to_array
	var metadata C.mlx_map_string_to_string
	if reader == nil {
		return nil, errors.New("mlx: nil safetensors reader")
	}

	stream := loadSafetensorsStream()
	if err := mlxError(stream); err != nil {
		_ = reader.Close()
		return nil, err
	}
	defer freeStream(stream)

	cReader, err := newIOReader(reader)
	if err != nil {
		return nil, err
	}
	defer freeIOReader(cReader)

	if err := mlxError(C.mlx_load_safetensors_reader(&arrays, &metadata, cReader, stream)); err != nil {
		return nil, fmt.Errorf("failed to load safetensors: %w", err)
	}

	return &SafetensorsFile{arrays: arrays, metadata: metadata}, nil
}

func newIOReader(reader SafetensorsReader) (C.mlx_io_reader, error) {
	var zero C.mlx_io_reader
	payload := (*cgo.Handle)(C.malloc(C.size_t(unsafe.Sizeof(cgo.Handle(0)))))
	if payload == nil {
		_ = reader.Close()
		return zero, errors.New("mlx: failed to allocate I/O reader handle")
	}
	handle := cgo.NewHandle(&ioReader{reader: reader})
	*payload = handle
	cReader := C.go_mlx_reader_new(unsafe.Pointer(payload))
	if err := mlxError(cReader); err != nil {
		handle.Delete()
		C.free(unsafe.Pointer(payload))
		_ = reader.Close()
		return zero, fmt.Errorf("mlx: failed to create I/O reader: %w", err)
	}
	return cReader, nil
}

func freeIOReader(reader C.mlx_io_reader) {
	mlxCheck(C.mlx_io_reader_free(reader))
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

func (s *SafetensorsFile) Arrays() iter.Seq2[string, *Array] {
	return func(yield func(string, *Array) bool) {
		it := mlxCheck(C.mlx_map_string_to_array_iterator_new(s.arrays))
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
