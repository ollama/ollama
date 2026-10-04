package gguf

import (
	"fmt"
	"io"
	"reflect"
)

// Readers and accessors used only by tests. Production code reads GGUF files
// through ReadFileMetadata and ReadModel.

func Open(path string) (*File, error) {
	return open(path, -1)
}

func (f *File) NumKeyValues() int {
	return int(f.keyValues.count)
}

func (f *File) NumTensors() int {
	return int(f.tensors.count)
}

func (f *File) TensorReader(name string) (TensorInfo, io.Reader, error) {
	t := f.TensorInfo(name)
	if err := f.Err(); err != nil {
		return TensorInfo{}, nil, err
	}
	if t.Name == "" {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}
	// fast forward through tensor info if we haven't already
	f.tensors.rest()
	if err := f.Err(); err != nil {
		return TensorInfo{}, nil, err
	}

	fileInfo, err := f.file.Stat()
	if err != nil {
		return TensorInfo{}, nil, err
	}
	offset, numBytes, err := f.tensorRange(t, fileInfo.Size())
	if err != nil {
		return TensorInfo{}, nil, err
	}

	return t, io.NewSectionReader(f.file, offset, numBytes), nil
}

func (ti TensorInfo) Valid() bool {
	return ti.Name != "" && ti.NumBytes() > 0
}

// Int returns Value as a signed integer. If it is not a signed integer, it returns 0.
func (v Value) Int() int64 {
	return value[int64](v, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64)
}

// Uint converts an unsigned integer value to uint64. If the value is not a unsigned integer, it returns 0.
func (v Value) Uint() uint64 {
	return value[uint64](v, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64)
}

// Float returns Value as a float. If it is not a float, it returns 0.
func (v Value) Float() float64 {
	return value[float64](v, reflect.Float32, reflect.Float64)
}

// Floats returns Value as a float slice. If it is not a float slice, it returns nil.
func (v Value) Floats() (f64s []float64) {
	return values[float64](v, reflect.Float32, reflect.Float64)
}

// Bools returns Value as a boolean slice. If it is not a boolean slice, it returns nil.
func (v Value) Bools() (bools []bool) {
	return values[bool](v, reflect.Bool)
}

// Strings returns Value as a string slice. If it is not a string slice, it returns nil.
func (v Value) Strings() (strings []string) {
	return values[string](v, reflect.String)
}
