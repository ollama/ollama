package mapx

import (
	jsonv1 "encoding/json"
	"encoding/json/v2"
	"slices"
	"testing"
)

func TestOrderedMap_MarshalOrder(t *testing.T) {
	var m OrderedMap[string, int]
	m.Set("b", 2)
	m.Set("a", 1)
	m.Set("c", 3)

	bs, err := json.Marshal(&m)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(bs), `{"b":2,"a":1,"c":3}`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}

	// The value form must marshal identically, and a nil pointer must
	// marshal as null, in both encoding/json and encoding/json/v2.
	if bs2, err := json.Marshal(m); err != nil || string(bs2) != string(bs) {
		t.Errorf("value marshal: %s, %v", bs2, err)
	}
	var np *OrderedMap[string, int]
	if bs2, err := json.Marshal(np); err != nil || string(bs2) != "null" {
		t.Errorf("v2 nil marshal: %s, %v", bs2, err)
	}
	if bs2, err := jsonv1.Marshal(np); err != nil || string(bs2) != "null" {
		t.Errorf("v1 nil marshal: %s, %v", bs2, err)
	}
}

func TestOrderedMap_UnmarshalOrder(t *testing.T) {
	var m OrderedMap[string, int]
	if err := json.Unmarshal([]byte(`{"z":1,"a":2,"m":3}`), &m); err != nil {
		t.Fatal(err)
	}
	if got, want := slices.Collect(m.Keys()), []string{"z", "a", "m"}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if v, ok := m.GetOk("a"); !ok || v != 2 {
		t.Errorf("got (%d, %v), want (2, true)", v, ok)
	}
}

func TestOrderedMap_RoundTrip(t *testing.T) {
	var m OrderedMap[string, any]
	var inner OrderedMap[string, int]
	inner.Set("y", 2)
	inner.Set("x", 1)
	m.Set("inner", &inner)
	m.Set("n", nil)

	bs, err := json.Marshal(&m)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(bs), `{"inner":{"y":2,"x":1},"n":null}`; got != want {
		t.Errorf("got %s, want %s", got, want)
	}

	var m2 OrderedMap[string, *OrderedMap[string, int]]
	if err := json.Unmarshal(bs, &m2); err != nil {
		t.Fatal(err)
	}
	if bs2, err := json.Marshal(&m2); err != nil || string(bs2) != string(bs) {
		t.Errorf("round trip: %s, %v", bs2, err)
	}
}

// JSON null clears the map, like a Go map.
func TestOrderedMap_UnmarshalNull(t *testing.T) {
	var m OrderedMap[string, int]
	m.Set("a", 1)
	if err := json.Unmarshal([]byte(`null`), &m); err != nil {
		t.Fatal(err)
	}
	if got := slices.Collect(m.Keys()); len(got) != 0 {
		t.Errorf("got %v, want empty", got)
	}
}

// An empty object unmarshals to an empty (non-nil) map, like a Go map.
func TestOrderedMap_UnmarshalEmpty(t *testing.T) {
	var m OrderedMap[string, int]
	if err := json.Unmarshal([]byte(`{}`), &m); err != nil {
		t.Fatal(err)
	}
	if got := slices.Collect(m.Keys()); len(got) != 0 {
		t.Errorf("got %v, want empty", got)
	}
}

// A non-object is an error.
func TestOrderedMap_UnmarshalNonObject(t *testing.T) {
	var m OrderedMap[string, int]
	if err := json.Unmarshal([]byte(`[1]`), &m); err == nil {
		t.Error("expected an error unmarshaling an array")
	}
}

func TestOrderedMap_Len(t *testing.T) {
	var np *OrderedMap[string, any]
	if got := np.Len(); got != 0 {
		t.Errorf("Len(nil) = %d, want 0", got)
	}
	var m OrderedMap[string, any]
	if got := m.Len(); got != 0 {
		t.Errorf("Len(empty) = %d, want 0", got)
	}
	m.Set("a", 1)
	m.Set("b", 2)
	if got := m.Len(); got != 2 {
		t.Errorf("Len(2 entries) = %d, want 2", got)
	}
}
