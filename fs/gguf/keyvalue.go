package gguf

import (
	"reflect"
	"slices"
)

type KeyValue struct {
	Key string
	Value
}

func (kv KeyValue) Valid() bool {
	return kv.Key != "" && kv.Value.value != nil
}

type Value struct {
	value any
}

func value[T any](v Value, kinds ...reflect.Kind) (t T) {
	vv := reflect.ValueOf(v.value)
	if slices.Contains(kinds, vv.Kind()) {
		t = vv.Convert(reflect.TypeOf(t)).Interface().(T)
	}
	return
}

func values[T any](v Value, kinds ...reflect.Kind) (ts []T) {
	switch vv := reflect.ValueOf(v.value); vv.Kind() {
	case reflect.Slice:
		if slices.Contains(kinds, vv.Type().Elem().Kind()) {
			ts = make([]T, vv.Len())
			for i := range vv.Len() {
				ts[i] = vv.Index(i).Convert(reflect.TypeOf(ts[i])).Interface().(T)
			}
		}
	}
	return
}

// Int returns Value as a signed integer. If it is not a signed integer, it returns 0.
func (v Value) Int() int64 {
	return value[int64](v, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64)
}

// Ints returns Value as a signed integer slice. If it is not a signed integer slice, it returns nil.
func (v Value) Ints() (i64s []int64) {
	return values[int64](v, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64)
}

// Uint converts an unsigned integer value to uint64. If the value is not a unsigned integer, it returns 0.
func (v Value) Uint() uint64 {
	return value[uint64](v, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64)
}

// Uints returns Value as a unsigned integer slice. If it is not a unsigned integer slice, it returns nil.
func (v Value) Uints() (u64s []uint64) {
	return values[uint64](v, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64)
}

// Float returns Value as a float. If it is not a float, it returns 0.
func (v Value) Float() float64 {
	return value[float64](v, reflect.Float32, reflect.Float64)
}

// Floats returns Value as a float slice. If it is not a float slice, it returns nil.
func (v Value) Floats() (f64s []float64) {
	return values[float64](v, reflect.Float32, reflect.Float64)
}

// Bool returns Value as a boolean. If it is not a boolean, it returns false.
func (v Value) Bool() bool {
	return value[bool](v, reflect.Bool)
}

// Bools returns Value as a boolean slice. If it is not a boolean slice, it returns nil.
func (v Value) Bools() (bools []bool) {
	return values[bool](v, reflect.Bool)
}

// String returns Value as a string. If it is not a string, it returns an empty string.
func (v Value) String() string {
	return value[string](v, reflect.String)
}

// Strings returns Value as a string slice. If it is not a string slice, it returns nil.
func (v Value) Strings() (strings []string) {
	return values[string](v, reflect.String)
}
