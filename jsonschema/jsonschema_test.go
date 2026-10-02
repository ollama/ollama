// Copyright 2025 The JSON Schema Go Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Derived from the schema_test.go file of github.com/google/jsonschema-go
// (v0.4.3), adapted for ordered map keyword fields.

package jsonschema

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"testing"

	"tailscale.com/util/must"

	"github.com/ollama/ollama/types/mapx"
)

// orderedSchema builds a *mapx.OrderedMap[string, *Schema] from alternating
// name, schema arguments, so that tests are deterministic.
func orderedSchema(kvs ...any) *mapx.OrderedMap[string, *Schema] {
	var m mapx.OrderedMap[string, *Schema]
	for i := 0; i+1 < len(kvs); i += 2 {
		m.Set(kvs[i].(string), kvs[i+1].(*Schema))
	}
	return &m
}

// orderedAny builds a *mapx.OrderedMap[string, any] from alternating key, value
// arguments, so that tests are deterministic.
func orderedAny(kvs ...any) *mapx.OrderedMap[string, any] {
	var m mapx.OrderedMap[string, any]
	for i := 0; i+1 < len(kvs); i += 2 {
		m.Set(kvs[i].(string), kvs[i+1])
	}
	return &m
}

// json returns the schema in json format.
func (s *Schema) json() string {
	data, err := json.Marshal(s)
	if err != nil {
		return fmt.Sprintf("<jsonschema.Schema:%v>", err)
	}
	return string(data)
}

func TestMarshalJSONConsistency(t *testing.T) {
	// Test that MarshalJSON with value receiver ensures consistent JSON encoding
	// regardless of how Schema is stored (fixes golang/go#22967, golang/go#33993, and golang/go#55890)

	// Create a test schema
	testSchema := Schema{
		Type:      "object",
		MinLength: new(10),
		Properties: orderedSchema(
			"name", &Schema{Type: "string"},
			"age", &Schema{Type: "integer"},
		),
		Required: []string{"name"},
	}

	// Expected JSON output
	expectedJSON, err := json.Marshal(testSchema)
	if err != nil {
		t.Fatalf("Failed to marshal expected schema: %v", err)
	}

	if !strings.Contains(string(expectedJSON), "object") {
		t.Fatalf("Expected JSON does not contain 'object': %s", string(expectedJSON))
	}

	t.Run("DirectValue", func(t *testing.T) {
		// Test direct value marshaling
		got, err := json.Marshal(testSchema)
		if err != nil {
			t.Fatalf("Failed to marshal direct value: %v", err)
		}
		if string(got) != string(expectedJSON) {
			t.Errorf("Direct value marshaling mismatch\ngot:  %s\nwant: %s", got, expectedJSON)
		}
	})

	t.Run("Pointer", func(t *testing.T) {
		// Test pointer marshaling
		schemaPtr := &testSchema
		got, err := json.Marshal(schemaPtr)
		if err != nil {
			t.Fatalf("Failed to marshal pointer: %v", err)
		}
		if string(got) != string(expectedJSON) {
			t.Errorf("Pointer marshaling mismatch\ngot:  %s\nwant: %s", got, expectedJSON)
		}
	})

	t.Run("MapValue", func(t *testing.T) {
		// Test marshaling when stored as map value (non-addressable)
		// This is a key case that fails with pointer receiver
		schemaMap := map[string]Schema{
			"test": testSchema,
		}
		got, err := json.Marshal(schemaMap["test"])
		if err != nil {
			t.Fatalf("Failed to marshal map value: %v", err)
		}
		if string(got) != string(expectedJSON) {
			t.Errorf("Map value marshaling mismatch\ngot:  %s\nwant: %s", got, expectedJSON)
		}
	})

	t.Run("MapWithSchemaValues", func(t *testing.T) {
		// Test marshaling a map containing Schema values
		schemaMap := map[string]Schema{
			"schema1": testSchema,
			"schema2": {Type: "string"},
		}
		got, err := json.Marshal(schemaMap)
		if err != nil {
			t.Fatalf("Failed to marshal map with Schema values: %v", err)
		}

		// Verify the map marshals correctly
		var result map[string]json.RawMessage
		if err := json.Unmarshal(got, &result); err != nil {
			t.Fatalf("Failed to unmarshal result: %v", err)
		}

		// Check that schema1 matches expected
		if string(result["schema1"]) != string(expectedJSON) {
			t.Errorf("Map schema1 marshaling mismatch\ngot:  %s\nwant: %s", result["schema1"], expectedJSON)
		}
	})

	t.Run("SliceElement", func(t *testing.T) {
		// Test marshaling when stored in a slice
		schemas := []Schema{testSchema}
		gotSlice, err := json.Marshal(schemas)
		if err != nil {
			t.Fatalf("Failed to marshal slice: %v", err)
		}

		var unmarshaledSlice []json.RawMessage
		if err := json.Unmarshal(gotSlice, &unmarshaledSlice); err != nil {
			t.Fatalf("Failed to unmarshal slice: %v", err)
		}

		if len(unmarshaledSlice) != 1 || string(unmarshaledSlice[0]) != string(expectedJSON) {
			t.Errorf("Slice element marshaling mismatch\ngot:  %s\nwant: %s",
				unmarshaledSlice[0], expectedJSON)
		}
	})

	t.Run("StructField", func(t *testing.T) {
		// Test marshaling when embedded in another struct
		type Container struct {
			Schema Schema `json:"schema"`
			Name   string `json:"name"`
		}

		container := Container{
			Schema: testSchema,
			Name:   "test",
		}

		got, err := json.Marshal(container)
		if err != nil {
			t.Fatalf("Failed to marshal struct with Schema field: %v", err)
		}

		var result map[string]json.RawMessage
		if err := json.Unmarshal(got, &result); err != nil {
			t.Fatalf("Failed to unmarshal struct result: %v", err)
		}

		if string(result["schema"]) != string(expectedJSON) {
			t.Errorf("Struct field marshaling mismatch\ngot:  %s\nwant: %s",
				result["schema"], expectedJSON)
		}
	})

	t.Run("InterfaceValue", func(t *testing.T) {
		// Test marshaling when stored as interface{}
		var iface any = testSchema
		got, err := json.Marshal(iface)
		if err != nil {
			t.Fatalf("Failed to marshal interface value: %v", err)
		}
		if string(got) != string(expectedJSON) {
			t.Errorf("Interface value marshaling mismatch\ngot:  %s\nwant: %s", got, expectedJSON)
		}
	})

	t.Run("EmptyPropertiesMap", func(t *testing.T) {
		// Test that an empty map in Properties marshals as "{}".
		s := &Schema{Type: "object", Properties: new(mapx.OrderedMap[string, *Schema])}
		got, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("Failed to marshal interface value: %v", err)
		}
		want := `{"type":"object","properties":{}}`
		if string(got) != want {
			t.Errorf("\ngot  %s\nwant %s", got, want)
		}
	})
}

func TestGoRoundTrip(t *testing.T) {
	// Verify that Go representations round-trip, by comparing the JSON
	// of the original schema with the JSON of the schema obtained by
	// unmarshaling the original's JSON.
	for _, s := range []*Schema{
		{Type: "null"},
		{Types: []string{"null", "number"}},
		{Type: "string", MinLength: new(20)},
		{Minimum: new(20.0)},
		{Items: &Schema{Type: "integer"}},
		{Const: new(any(0))},
		{Const: new(any(nil))},
		{Const: new(any([]int{}))},
		{Default: must.Get(json.Marshal(1))},
		{Default: must.Get(json.Marshal(nil))},
		{Extra: orderedAny("test", "value")},
		{Properties: orderedSchema(
			"z", &Schema{Type: "string"},
			"a", &Schema{Type: "integer"},
		)},
	} {
		data, err := json.Marshal(s)
		if err != nil {
			t.Fatal(err)
		}
		var got *Schema
		must.Do(json.Unmarshal(data, &got))
		if gotJSON, sJSON := got.json(), s.json(); gotJSON != sJSON {
			t.Errorf("got %s, want %s", gotJSON, sJSON)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	// Verify that JSON texts for schemas marshal into equivalent forms.
	// We don't expect everything to round-trip perfectly. For example, "true" and "false"
	// will turn into their object equivalents.
	// But most things should.
	// Some of these cases test Schema.{UnM,M}arshalJSON.
	// Most of others follow from the behavior of encoding/json, but they are still
	// valuable as regression tests of this package's behavior.
	for _, tt := range []struct {
		in, want string
	}{
		{`true`, `true`},
		{`false`, `false`},
		{`{"type":"", "enum":null}`, `true`}, // empty fields are omitted
		{`{"minimum":1}`, `{"minimum":1}`},
		{`{"minimum":1.0}`, `{"minimum":1}`},     // floating-point integers lose their fractional part
		{`{"minLength":1.0}`, `{"minLength":1}`}, // some floats are unmarshaled into ints, but you can't tell
		{
			// ordered map keys keep the order in which they appear
			`{"$vocabulary":{"b":true, "a":false}}`,
			`{"$vocabulary":{"b":true,"a":false}}`,
		},
		{`{"unk":0}`, `{"unk":0}`}, // unknown fields are not dropped
		{
			// known and unknown fields are not dropped
			// note that the order will be by the declaration order of the struct inside MarshalJSON
			`{"comment":"test","type":"example","unk":0}`,
			`{"type":"example","comment":"test","unk":0}`,
		},
		{`{"extra":0}`, `{"extra":0}`}, // extra is not a special keyword and should not be dropped
		{`{"Extra":0}`, `{"Extra":0}`}, // Extra is not a special keyword and should not be dropped
		{
			// property order is preserved on round-trip
			`{"type":"object","properties":{"z":true,"a":true,"m":true}}`,
			`{"type":"object","properties":{"z":true,"a":true,"m":true}}`,
		},
		{
			// $defs order is preserved on round-trip
			`{"$defs":{"z":true,"a":true},"$ref":"#"}`,
			`{"$ref":"#","$defs":{"z":true,"a":true}}`,
		},
		{
			// draft-07 dependencies round-trip as schemas or string arrays.
			// Dependencies that are schemas are written before dependencies
			// that are string arrays, because they are kept in separate maps.
			`{"dependencies":{"b":["c"],"a":{"type":"string"}}}`,
			`{"dependencies":{"a":{"type":"string"},"b":["c"]}}`,
		},
		// A null map-valued keyword is dropped, like a null Go map.
		{`{"properties":null}`, `true`},
		{`{"$vocabulary":null}`, `true`},
		// An empty map-valued keyword is kept, like an empty Go map.
		{`{"properties":{}}`, `{"properties":{}}`},
		{`{"$defs":{}}`, `{"$defs":{}}`},
		{`{"dependentRequired":{}}`, `{"dependentRequired":{}}`},
		// An empty or null "dependencies" keyword produces no
		// DependencySchemas or DependencyStrings, so it is dropped.
		{`{"dependencies":null}`, `true`},
		{`{"dependencies":{}}`, `true`},
	} {
		var s Schema
		must.Do(json.Unmarshal([]byte(tt.in), &s))
		data, err := json.Marshal(&s)
		if err != nil {
			t.Fatal(err)
		}
		if got := string(data); got != tt.want {
			t.Errorf("%s:\ngot  %s\nwant %s", tt.in, got, tt.want)
		}
	}
}

func TestUnmarshalErrors(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want string // error must match this regexp
	}{
		{`1`, "cannot unmarshal number"},
		{`{"type":1}`, `invalid value for "type"`},
		{`{"minLength":1.5}`, `not an integer value`},
		{`{"maxLength":1.5}`, `not an integer value`},
		{`{"minItems":1.5}`, `not an integer value`},
		{`{"maxItems":1.5}`, `not an integer value`},
		{`{"minProperties":1.5}`, `not an integer value`},
		{`{"maxProperties":1.5}`, `not an integer value`},
		{`{"minContains":1.5}`, `not an integer value`},
		{`{"maxContains":1.5}`, `not an integer value`},
		{fmt.Sprintf(`{"maxContains":%d}`, int64(math.MaxInt32+1)), `out of range`},
		{`{"minLength":9e99}`, `cannot be unmarshaled`},
		{`{"minLength":"1.5"}`, `not a number`},
	} {
		var s Schema
		err := json.Unmarshal([]byte(tt.in), &s)
		if err == nil {
			t.Fatalf("%s: no error but expected one", tt.in)
		}
		if !regexp.MustCompile(tt.want).MatchString(err.Error()) {
			t.Errorf("%s: error %q does not match %q", tt.in, err, tt.want)
		}
	}
}

func TestMarshalOrder(t *testing.T) {
	for _, tt := range []struct {
		order      []string
		want       string
		wantErr    bool
		errMessage string
	}{
		{
			[]string{"A", "B", "C", "D"},
			`{"type":"object","properties":{"A":{"type":"integer"},"B":{"type":"integer"},"C":{"type":"integer"},"D":{"type":"integer"},"E":{"type":"integer"}}}`,
			false,
			"",
		},
		{
			[]string{"A", "C", "B", "D"},
			`{"type":"object","properties":{"A":{"type":"integer"},"C":{"type":"integer"},"B":{"type":"integer"},"D":{"type":"integer"},"E":{"type":"integer"}}}`,
			false,
			"",
		},
		{
			[]string{"D", "C", "B", "A"},
			`{"type":"object","properties":{"D":{"type":"integer"},"C":{"type":"integer"},"B":{"type":"integer"},"A":{"type":"integer"},"E":{"type":"integer"}}}`,
			false,
			"",
		},
		{
			[]string{"A", "B", "C"},
			`{"type":"object","properties":{"A":{"type":"integer"},"B":{"type":"integer"},"C":{"type":"integer"},"D":{"type":"integer"},"E":{"type":"integer"}}}`,
			false,
			"",
		},
		{
			// properties not listed in PropertyOrder keep their insertion order
			[]string{"A", "E", "C"},
			`{"type":"object","properties":{"A":{"type":"integer"},"E":{"type":"integer"},"C":{"type":"integer"},"B":{"type":"integer"},"D":{"type":"integer"}}}`,
			false,
			"",
		},
		{
			[]string{"A", "B", "C", "D", "D"},
			"",
			true,
			`json: error calling MarshalJSON for type *jsonschema.Schema: property order slice cannot contain duplicate entries, found duplicate "D"`,
		},
	} {
		s := &Schema{
			Type: "object",
			Properties: orderedSchema(
				"A", &Schema{Type: "integer"},
				"B", &Schema{Type: "integer"},
				"C", &Schema{Type: "integer"},
				"D", &Schema{Type: "integer"},
				"E", &Schema{Type: "integer"},
			),
		}
		s.PropertyOrder = tt.order
		gotBytes, err := json.Marshal(s)
		if err != nil {
			if !tt.wantErr {
				t.Fatal(err)
			}
			if err.Error() != tt.errMessage {
				t.Fatalf("error message mismatch:\ngot  %q\nwant %q", err.Error(), tt.errMessage)
			}
			continue
		}
		if got := string(gotBytes); got != tt.want {
			t.Fatalf("ForType mismatch:\ngot  %s\nwant %s", got, tt.want)
		}
	}
}

// TestInsertionOrder checks that map-valued keywords marshal in insertion
// order, and that unmarshaling preserves the order of keys in the input, so
// that output is deterministic.
func TestInsertionOrder(t *testing.T) {
	// Insertion order, not alphabetical order.
	s := &Schema{
		Type: "object",
		Properties: orderedSchema(
			"zebra", &Schema{Type: "string"},
			"apple", &Schema{Type: "string"},
			"mango", &Schema{Type: "string"},
		),
		Extra: orderedAny("zzz", 1, "aaa", 2),
	}
	want := `{"type":"object","properties":{"zebra":{"type":"string"},"apple":{"type":"string"},"mango":{"type":"string"}},"zzz":1,"aaa":2}`
	if got := s.json(); got != want {
		t.Errorf("\ngot  %s\nwant %s", got, want)
	}

	// Marshaling is deterministic across repeated calls.
	for range 10 {
		if got := s.json(); got != want {
			t.Errorf("non-deterministic marshal:\ngot  %s\nwant %s", got, want)
		}
	}

	// Unmarshal preserves input order.
	in := `{"type":"object","properties":{"zebra":{"type":"string"},"apple":{"type":"string"},"mango":{"type":"string"}},"zzz":1,"aaa":2}`
	var got Schema
	must.Do(json.Unmarshal([]byte(in), &got))
	if got.json() != in {
		t.Errorf("\ngot  %s\nwant %s", got.json(), in)
	}
}

func TestCloneSchemas(t *testing.T) {
	ss1 := &Schema{Type: "string"}
	ss2 := &Schema{Type: "integer"}
	ss3 := &Schema{Type: "boolean"}
	ss4 := &Schema{Type: "number"}
	ss5 := &Schema{Contains: ss4}

	s1 := Schema{
		Contains:    ss1,
		PrefixItems: []*Schema{ss2, ss3},
		Properties:  orderedSchema("a", ss5),
	}
	s2 := s1.CloneSchemas()

	// The clones should appear identical.
	if g, w := s1.json(), s2.json(); g != w {
		t.Errorf("\ngot  %s\nwant %s", g, w)
	}
	// None of the subschemas should overlap with the originals.
	if s2.Contains == ss1 || s2.PrefixItems[0] == ss2 || s2.PrefixItems[1] == ss3 {
		t.Errorf("uncloned schema in %s", s2.json())
	}
	ga := s2.Properties.Get("a")
	if ga == ss5 || ga.Contains == ss4 {
		t.Errorf("uncloned schema in %s", s2.json())
	}
	// s1's original schemas should be intact.
	if s1.Contains != ss1 || s1.PrefixItems[0] != ss2 || s1.PrefixItems[1] != ss3 || ss5.Contains != ss4 {
		t.Errorf("s1 modified")
	}
	if pa := s1.Properties.Get("a"); pa != ss5 {
		t.Errorf("s1 modified")
	}
}

func TestBasicChecks(t *testing.T) {
	for _, tt := range []struct {
		s    *Schema
		want string // "" means no error
	}{
		{&Schema{}, ""},
		{&Schema{Type: "string", Types: []string{"string"}}, "both Type and Types are set"},
		{&Schema{Defs: new(mapx.OrderedMap[string, *Schema]), Definitions: new(mapx.OrderedMap[string, *Schema])}, "both Defs and Definitions are set"},
		{&Schema{Items: &Schema{}, ItemsArray: []*Schema{}}, "both Items and ItemsArray are set"},
		{&Schema{PropertyOrder: []string{"a", "a"}}, `found duplicate "a"`},
		{
			&Schema{
				DependencySchemas: orderedSchema("a", &Schema{}),
				DependencyStrings: orderedStrings("a", []string{"b"}),
			},
			`dependency key "a" cannot be defined as both a schema and a string array`,
		},
	} {
		err := tt.s.basicChecks()
		if tt.want == "" {
			if err != nil {
				t.Errorf("%v: unexpected error %v", tt.s, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%v: got error %v, want %q", tt.s, err, tt.want)
		}
	}
}

// orderedStrings builds a *mapx.OrderedMap[string, []string] from alternating
// key, value arguments.
func orderedStrings(kvs ...any) *mapx.OrderedMap[string, []string] {
	var m mapx.OrderedMap[string, []string]
	for i := 0; i+1 < len(kvs); i += 2 {
		m.Set(kvs[i].(string), kvs[i+1].([]string))
	}
	return &m
}
