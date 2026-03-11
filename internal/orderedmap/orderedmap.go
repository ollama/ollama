// Package orderedmap provides a generic ordered map that maintains insertion order.
// It wraps github.com/wk8/go-ordered-map/v2 to encapsulate the dependency.
package orderedmap

import (
	"bytes"
	"encoding/json"
	"iter"

	"github.com/ollama/ollama/internal/jsonutil"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

// Map is a generic ordered map that maintains insertion order.
type Map[K comparable, V any] struct {
	om *orderedmap.OrderedMap[K, V]
}

// New creates a new empty ordered map.
func New[K comparable, V any]() *Map[K, V] {
	return &Map[K, V]{
		om: orderedmap.New[K, V](),
	}
}

// Get retrieves a value by key.
func (m *Map[K, V]) Get(key K) (V, bool) {
	if m == nil || m.om == nil {
		var zero V
		return zero, false
	}
	return m.om.Get(key)
}

// Set sets a key-value pair. If the key already exists, its value is updated
// but its position in the iteration order is preserved. If the key is new,
// it is appended to the end.
func (m *Map[K, V]) Set(key K, value V) {
	if m == nil {
		return
	}
	if m.om == nil {
		m.om = orderedmap.New[K, V]()
	}
	m.om.Set(key, value)
}

// Len returns the number of entries.
func (m *Map[K, V]) Len() int {
	if m == nil || m.om == nil {
		return 0
	}
	return m.om.Len()
}

// All returns an iterator over all key-value pairs in insertion order.
func (m *Map[K, V]) All() iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		if m == nil || m.om == nil {
			return
		}
		for pair := m.om.Oldest(); pair != nil; pair = pair.Next() {
			if !yield(pair.Key, pair.Value) {
				return
			}
		}
	}
}

// ToMap converts to a regular Go map.
// Note: The resulting map does not preserve order.
func (m *Map[K, V]) ToMap() map[K]V {
	if m == nil || m.om == nil {
		return nil
	}
	result := make(map[K]V, m.om.Len())
	for pair := m.om.Oldest(); pair != nil; pair = pair.Next() {
		result[pair.Key] = pair.Value
	}
	return result
}

// MarshalJSON implements json.Marshaler. The JSON output preserves key order.
func (m *Map[K, V]) MarshalJSON() ([]byte, error) {
	if m == nil || m.om == nil {
		return []byte("null"), nil
	}

	var buf bytes.Buffer
	buf.WriteByte('{')

	first := true
	for pair := m.om.Oldest(); pair != nil; pair = pair.Next() {
		if !first {
			buf.WriteByte(',')
		}
		first = false

		key, err := jsonutil.Marshal(pair.Key)
		if err != nil {
			return nil, err
		}
		value, err := jsonutil.Marshal(pair.Value)
		if err != nil {
			return nil, err
		}

		buf.Write(key)
		buf.WriteByte(':')
		buf.Write(value)
	}

	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// UnmarshalJSON implements json.Unmarshaler. The insertion order matches the
// order of keys in the JSON input.
func (m *Map[K, V]) UnmarshalJSON(data []byte) error {
	m.om = orderedmap.New[K, V]()
	return json.Unmarshal(data, &m.om)
}
