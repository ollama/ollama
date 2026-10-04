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

// Any returns Value as stored, without conversion. If it is not set, it returns nil.
func (v Value) Any() any {
	return v.value
}

func value[T any](v Value, kinds ...reflect.Kind) (t T) {
	vv := reflect.ValueOf(v.value)
	if slices.Contains(kinds, vv.Kind()) {
		t = vv.Convert(reflect.TypeOf(t)).Interface().(T)
	}
	return
}

func valueOK[T any](v Value, kinds ...reflect.Kind) (t T, ok bool) {
	vv := reflect.ValueOf(v.value)
	if !vv.IsValid() || !slices.Contains(kinds, vv.Kind()) {
		return t, false
	}

	return vv.Convert(reflect.TypeOf(t)).Interface().(T), true
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

// IntOK converts a signed integer value to int64 and reports whether the
// underlying type was signed.
func (v Value) IntOK() (int64, bool) {
	return valueOK[int64](v, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64)
}

// Ints returns Value as a signed integer slice. If it is not a signed integer slice, it returns nil.
func (v Value) Ints() (i64s []int64) {
	return values[int64](v, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64)
}

// UintOK converts an unsigned integer value to uint64 and reports whether the
// underlying type was unsigned.
func (v Value) UintOK() (uint64, bool) {
	return valueOK[uint64](v, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64)
}

// Uints returns Value as a unsigned integer slice. If it is not a unsigned integer slice, it returns nil.
func (v Value) Uints() (u64s []uint64) {
	return values[uint64](v, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64)
}

// Bool returns Value as a boolean. If it is not a boolean, it returns false.
func (v Value) Bool() bool {
	return value[bool](v, reflect.Bool)
}

// String returns Value as a string. If it is not a string, it returns an empty string.
func (v Value) String() string {
	return value[string](v, reflect.String)
}
