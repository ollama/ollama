package orderedmap

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMap_BasicOperations(t *testing.T) {
	m := New[string, int]()

	// Test empty map
	assert.Equal(t, 0, m.Len())
	v, ok := m.Get("a")
	assert.False(t, ok)
	assert.Equal(t, 0, v)

	// Test Set and Get
	m.Set("a", 1)
	m.Set("b", 2)
	m.Set("c", 3)

	assert.Equal(t, 3, m.Len())

	v, ok = m.Get("a")
	assert.True(t, ok)
	assert.Equal(t, 1, v)

	v, ok = m.Get("b")
	assert.True(t, ok)
	assert.Equal(t, 2, v)

	v, ok = m.Get("c")
	assert.True(t, ok)
	assert.Equal(t, 3, v)

	// Test updating existing key preserves position
	m.Set("a", 10)
	v, ok = m.Get("a")
	assert.True(t, ok)
	assert.Equal(t, 10, v)
	assert.Equal(t, 3, m.Len()) // Length unchanged
}

func TestMap_InsertionOrderPreserved(t *testing.T) {
	m := New[string, int]()

	// Insert in non-alphabetical order
	m.Set("z", 1)
	m.Set("a", 2)
	m.Set("m", 3)
	m.Set("b", 4)

	// Verify iteration order matches insertion order
	var keys []string
	var values []int
	for k, v := range m.All() {
		keys = append(keys, k)
		values = append(values, v)
	}

	assert.Equal(t, []string{"z", "a", "m", "b"}, keys)
	assert.Equal(t, []int{1, 2, 3, 4}, values)
}

func TestMap_UpdatePreservesPosition(t *testing.T) {
	m := New[string, int]()

	m.Set("first", 1)
	m.Set("second", 2)
	m.Set("third", 3)

	// Update middle element
	m.Set("second", 20)

	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
	}

	// Order should still be first, second, third
	assert.Equal(t, []string{"first", "second", "third"}, keys)
}

func TestMap_MarshalJSON_PreservesOrder(t *testing.T) {
	m := New[string, int]()

	// Insert in non-alphabetical order
	m.Set("z", 1)
	m.Set("a", 2)
	m.Set("m", 3)

	data, err := json.Marshal(m)
	require.NoError(t, err)

	// JSON should preserve insertion order, not alphabetical
	assert.Equal(t, `{"z":1,"a":2,"m":3}`, string(data))
}

func TestMap_UnmarshalJSON_PreservesOrder(t *testing.T) {
	// JSON with non-alphabetical key order
	jsonData := `{"z":1,"a":2,"m":3}`

	m := New[string, int]()
	err := json.Unmarshal([]byte(jsonData), m)
	require.NoError(t, err)

	// Verify iteration order matches JSON order
	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
	}

	assert.Equal(t, []string{"z", "a", "m"}, keys)
}

func TestMap_JSONRoundTrip(t *testing.T) {
	// Test that unmarshal -> marshal produces identical JSON
	original := `{"zebra":"z","apple":"a","mango":"m","banana":"b"}`

	m := New[string, string]()
	err := json.Unmarshal([]byte(original), m)
	require.NoError(t, err)

	data, err := json.Marshal(m)
	require.NoError(t, err)

	assert.Equal(t, original, string(data))
}

func TestMap_ToMap(t *testing.T) {
	m := New[string, int]()
	m.Set("a", 1)
	m.Set("b", 2)

	regular := m.ToMap()

	assert.Equal(t, 2, len(regular))
	assert.Equal(t, 1, regular["a"])
	assert.Equal(t, 2, regular["b"])
}

func TestMap_NilSafety(t *testing.T) {
	var m *Map[string, int]

	// All operations should be safe on nil
	assert.Equal(t, 0, m.Len())

	v, ok := m.Get("a")
	assert.False(t, ok)
	assert.Equal(t, 0, v)

	// Set on nil is a no-op
	m.Set("a", 1)
	assert.Equal(t, 0, m.Len())

	// All returns empty iterator
	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
	}
	assert.Empty(t, keys)

	// ToMap returns nil
	assert.Nil(t, m.ToMap())

	// MarshalJSON returns null
	data, err := json.Marshal(m)
	require.NoError(t, err)
	assert.Equal(t, "null", string(data))
}

func TestMap_EmptyMapMarshal(t *testing.T) {
	m := New[string, int]()

	data, err := json.Marshal(m)
	require.NoError(t, err)
	assert.Equal(t, "{}", string(data))
}

func TestMap_NestedValues(t *testing.T) {
	m := New[string, any]()
	m.Set("string", "hello")
	m.Set("number", 42)
	m.Set("bool", true)
	m.Set("nested", map[string]int{"x": 1})

	data, err := json.Marshal(m)
	require.NoError(t, err)

	expected := `{"string":"hello","number":42,"bool":true,"nested":{"x":1}}`
	assert.Equal(t, expected, string(data))
}

func TestMap_AllIteratorEarlyExit(t *testing.T) {
	m := New[string, int]()
	m.Set("a", 1)
	m.Set("b", 2)
	m.Set("c", 3)
	m.Set("d", 4)

	// Collect only first 2
	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
		if len(keys) == 2 {
			break
		}
	}

	assert.Equal(t, []string{"a", "b"}, keys)
}

func TestMap_IntegerKeys(t *testing.T) {
	m := New[int, string]()
	m.Set(3, "three")
	m.Set(1, "one")
	m.Set(2, "two")

	var keys []int
	for k := range m.All() {
		keys = append(keys, k)
	}

	// Should preserve insertion order, not numerical order
	assert.Equal(t, []int{3, 1, 2}, keys)
}

func TestMap_UnmarshalIntoExisting(t *testing.T) {
	m := New[string, int]()
	m.Set("existing", 999)

	// Unmarshal should replace contents
	err := json.Unmarshal([]byte(`{"new":1}`), m)
	require.NoError(t, err)

	_, ok := m.Get("existing")
	assert.False(t, ok, "existing key should be gone after unmarshal")

	v, ok := m.Get("new")
	assert.True(t, ok)
	assert.Equal(t, 1, v)
}

func TestMap_LargeOrderPreservation(t *testing.T) {
	m := New[string, int]()

	// Create many keys in specific order
	keys := make([]string, 100)
	for i := range 100 {
		keys[i] = string(rune('a' + (99 - i))) // reverse order: 'd', 'c', 'b', 'a' (extended)
		if i >= 26 {
			keys[i] = string(rune('A'+i-26)) + string(rune('a'+i%26))
		}
	}

	for i, k := range keys {
		m.Set(k, i)
	}

	// Verify order preserved
	var resultKeys []string
	for k := range m.All() {
		resultKeys = append(resultKeys, k)
	}

	assert.True(t, slices.Equal(keys, resultKeys), "large map should preserve insertion order")
}
