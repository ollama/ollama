package main

import (
	"encoding/json"
	"slices"
	"testing"
)

func TestMetadataSemantics(t *testing.T) {
	a, err := decodeJSON([]byte(`{"number":18446744073709551615,"nullable":null,"tokens":["a ","b"],"object":{"x":1},"array":[1,2]}`))
	if err != nil {
		t.Fatal(err)
	}
	b, err := decodeJSON([]byte(`{"number":18446744073709551614,"tokens":["a","b"],"object":{"x":1.0},"array":[2,1]}`))
	if err != nil {
		t.Fatal(err)
	}
	var changes []MetadataChange
	metadataDiff("", a, b, true, true, &changes)
	var paths []string
	for _, c := range changes {
		paths = append(paths, c.Path)
	}
	if !slices.Equal(paths, []string{"/array/0", "/array/1", "/nullable", "/number", "/tokens/0"}) {
		t.Fatalf("wrong semantic changes: %+v", changes)
	}
	if c := changes[2]; !c.LeftPresent || c.RightPresent || c.Left != nil {
		t.Fatalf("lost null vs absent: %+v", c)
	}
	if n, ok := changes[3].Left.(json.Number); !ok || string(n) != "18446744073709551615" {
		t.Fatalf("number precision lost: %v", changes[3])
	}
	for _, bad := range []string{`{"x":1,"x":2}`, `{"a":{"x":1,"x":2}}`, `{} {}`, `[1,]`} {
		if _, err := decodeJSON([]byte(bad)); err == nil {
			t.Errorf("accepted ambiguous metadata %q", bad)
		}
	}
}

func FuzzMetadata(f *testing.F) {
	f.Add([]byte(`{"a":[1,2,null],"number":18446744073709551615}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		v, err := decodeJSON(data)
		if err != nil {
			return
		}
		var changes []MetadataChange
		metadataDiff("", v, v, true, true, &changes)
		if len(changes) != 0 {
			t.Fatalf("self diff: %v", changes)
		}
	})
}
