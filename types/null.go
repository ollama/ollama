package types

import (
	"encoding/json"
)

// Null represents a value of any type T that may be null.
type Null[T any] struct {
	value T
	valid bool
}

// NullWithValue creates a new, valid Null[T].
func NullWithValue[T any](value T) Null[T] {
	return Null[T]{value: value, valid: true}
}

// Value returns the value of the Type[T] if set, otherwise it returns the provided default value or the zero value of T.
func (n Null[T]) Value(defaultValue ...T) T {
	if n.valid {
		return n.value
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	var zero T
	return zero
}

// SetValue sets the value of the Type[T].
func (n *Null[T]) SetValue(t T) {
	n.value = t
	n.valid = true
}

// MarshalJSON implements [json.Marshaler].
func (n Null[T]) MarshalJSON() ([]byte, error) {
	if n.valid {
		return json.Marshal(n.value)
	}
	return []byte("null"), nil
}

// UnmarshalJSON implements [json.Unmarshaler].
func (n *Null[T]) UnmarshalJSON(data []byte) error {
	if string(data) != "null" {
		if err := json.Unmarshal(data, &n.value); err != nil {
			return err
		}
		n.valid = true
	}
	return nil
}
