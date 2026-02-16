package orderedmap

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestMap_BasicOperations(t *testing.T) {
	m := New[string, int]()

	// Test empty map
	if m.Len() != 0 {
		t.Errorf("expected Len() = 0, got %d", m.Len())
	}
	v, ok := m.Get("a")
	if ok {
		t.Error("expected Get on empty map to return false")
	}
	if v != 0 {
		t.Errorf("expected zero value, got %d", v)
	}

	// Test Set and Get
	m.Set("a", 1)
	m.Set("b", 2)
	m.Set("c", 3)

	if m.Len() != 3 {
		t.Errorf("expected Len() = 3, got %d", m.Len())
	}

	v, ok = m.Get("a")
	if !ok || v != 1 {
		t.Errorf("expected Get(a) = (1, true), got (%d, %v)", v, ok)
	}

	v, ok = m.Get("b")
	if !ok || v != 2 {
		t.Errorf("expected Get(b) = (2, true), got (%d, %v)", v, ok)
	}

	v, ok = m.Get("c")
	if !ok || v != 3 {
		t.Errorf("expected Get(c) = (3, true), got (%d, %v)", v, ok)
	}

	// Test updating existing key preserves position
	m.Set("a", 10)
	v, ok = m.Get("a")
	if !ok || v != 10 {
		t.Errorf("expected Get(a) = (10, true), got (%d, %v)", v, ok)
	}
	if m.Len() != 3 {
		t.Errorf("expected Len() = 3 after update, got %d", m.Len())
	}
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

	expectedKeys := []string{"z", "a", "m", "b"}
	expectedValues := []int{1, 2, 3, 4}

	if !slices.Equal(keys, expectedKeys) {
		t.Errorf("expected keys %v, got %v", expectedKeys, keys)
	}
	if !slices.Equal(values, expectedValues) {
		t.Errorf("expected values %v, got %v", expectedValues, values)
	}
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
	expected := []string{"first", "second", "third"}
	if !slices.Equal(keys, expected) {
		t.Errorf("expected keys %v, got %v", expected, keys)
	}
}

func TestMap_MarshalJSON_PreservesOrder(t *testing.T) {
	m := New[string, int]()

	// Insert in non-alphabetical order
	m.Set("z", 1)
	m.Set("a", 2)
	m.Set("m", 3)

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// JSON should preserve insertion order, not alphabetical
	expected := `{"z":1,"a":2,"m":3}`
	if string(data) != expected {
		t.Errorf("expected %s, got %s", expected, string(data))
	}
}

func TestMap_UnmarshalJSON_PreservesOrder(t *testing.T) {
	// JSON with non-alphabetical key order
	jsonData := `{"z":1,"a":2,"m":3}`

	m := New[string, int]()
	if err := json.Unmarshal([]byte(jsonData), m); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify iteration order matches JSON order
	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
	}

	expected := []string{"z", "a", "m"}
	if !slices.Equal(keys, expected) {
		t.Errorf("expected keys %v, got %v", expected, keys)
	}
}

func TestMap_JSONRoundTrip(t *testing.T) {
	// Test that unmarshal -> marshal produces identical JSON
	original := `{"zebra":"z","apple":"a","mango":"m","banana":"b"}`

	m := New[string, string]()
	if err := json.Unmarshal([]byte(original), m); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	if string(data) != original {
		t.Errorf("round trip failed: expected %s, got %s", original, string(data))
	}
}

func TestMap_ToMap(t *testing.T) {
	m := New[string, int]()
	m.Set("a", 1)
	m.Set("b", 2)

	regular := m.ToMap()

	if len(regular) != 2 {
		t.Errorf("expected len 2, got %d", len(regular))
	}
	if regular["a"] != 1 {
		t.Errorf("expected regular[a] = 1, got %d", regular["a"])
	}
	if regular["b"] != 2 {
		t.Errorf("expected regular[b] = 2, got %d", regular["b"])
	}
}

func TestMap_NilSafety(t *testing.T) {
	var m *Map[string, int]

	// All operations should be safe on nil
	if m.Len() != 0 {
		t.Errorf("expected Len() = 0 on nil map, got %d", m.Len())
	}

	v, ok := m.Get("a")
	if ok {
		t.Error("expected Get on nil map to return false")
	}
	if v != 0 {
		t.Errorf("expected zero value from nil map, got %d", v)
	}

	// Set on nil is a no-op
	m.Set("a", 1)
	if m.Len() != 0 {
		t.Errorf("expected Len() = 0 after Set on nil, got %d", m.Len())
	}

	// All returns empty iterator
	var keys []string
	for k := range m.All() {
		keys = append(keys, k)
	}
	if len(keys) != 0 {
		t.Errorf("expected empty iteration on nil map, got %v", keys)
	}

	// ToMap returns nil
	if m.ToMap() != nil {
		t.Error("expected ToMap to return nil on nil map")
	}

	// MarshalJSON returns null
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(data) != "null" {
		t.Errorf("expected null, got %s", string(data))
	}
}

func TestMap_EmptyMapMarshal(t *testing.T) {
	m := New[string, int]()

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(data) != "{}" {
		t.Errorf("expected {}, got %s", string(data))
	}
}

func TestMap_NestedValues(t *testing.T) {
	m := New[string, any]()
	m.Set("string", "hello")
	m.Set("number", 42)
	m.Set("bool", true)
	m.Set("nested", map[string]int{"x": 1})

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	expected := `{"string":"hello","number":42,"bool":true,"nested":{"x":1}}`
	if string(data) != expected {
		t.Errorf("expected %s, got %s", expected, string(data))
	}
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

	expected := []string{"a", "b"}
	if !slices.Equal(keys, expected) {
		t.Errorf("expected %v, got %v", expected, keys)
	}
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
	expected := []int{3, 1, 2}
	if !slices.Equal(keys, expected) {
		t.Errorf("expected %v, got %v", expected, keys)
	}
}

func TestMap_UnmarshalIntoExisting(t *testing.T) {
	m := New[string, int]()
	m.Set("existing", 999)

	// Unmarshal should replace contents
	if err := json.Unmarshal([]byte(`{"new":1}`), m); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	_, ok := m.Get("existing")
	if ok {
		t.Error("existing key should be gone after unmarshal")
	}

	v, ok := m.Get("new")
	if !ok || v != 1 {
		t.Errorf("expected Get(new) = (1, true), got (%d, %v)", v, ok)
	}
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

	if !slices.Equal(keys, resultKeys) {
		t.Error("large map should preserve insertion order")
	}
}
