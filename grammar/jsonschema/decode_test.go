package jsonschema

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

const testSchemaBasic = `
{
  "properties": {
    "tupleClosedEmpty":   { "prefixItems": [] },
    "tupleClosedMissing": { "prefixItems": [{}] },
    "tupleClosedNull":    { "prefixItems": [{}], "items": null },
    "tupleClosedFalse":   { "prefixItems": [{}], "items": false },
    "tupleOpenTrue":      { "prefixItems": [{}], "items": true },
    "tupleOpenEmpty":     { "prefixItems": [{}], "items": {} },
    "tupleOpenTyped":     { "prefixItems": [{}], "items": {"type": "boolean"} },
    "tupleOpenMax":       { "prefixItems": [{}], "items": true, "maxItems": 3},

    "array": { "items": {"type": "number"} },

    "null": { "type": "null" },
    "string": { "type": "string" },
    "boolean": { "type": "boolean" }
  }
}
`

func TestSchemaUnmarshal(t *testing.T) {
	var got *Schema
	if err := json.Unmarshal([]byte(testSchemaBasic), &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	want := &Schema{
		Properties: []*Schema{
			{Name: "tupleClosedEmpty", PrefixItems: []*Schema{}, Items: nil},
			{Name: "tupleClosedMissing", PrefixItems: []*Schema{{}}, Items: nil},
			{Name: "tupleClosedNull", PrefixItems: []*Schema{{}}, Items: nil},
			{Name: "tupleClosedFalse", PrefixItems: []*Schema{{}}, Items: nil},

			{Name: "tupleOpenTrue", PrefixItems: []*Schema{{}}, Items: &Schema{}},
			{Name: "tupleOpenEmpty", PrefixItems: []*Schema{{}}, Items: &Schema{}},
			{Name: "tupleOpenTyped", PrefixItems: []*Schema{{}}, Items: &Schema{Type: "boolean"}},
			{Name: "tupleOpenMax", PrefixItems: []*Schema{{}}, Items: &Schema{}, MaxItems: 3},

			{Name: "array", Items: &Schema{Type: "number"}},

			{Name: "null", Type: "null"},
			{Name: "string", Type: "string"},
			{Name: "boolean", Type: "boolean"},
		},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("(-want, +got)\n%s", diff)
	}
}

func TestEffectiveType(t *testing.T) {
	const schema = `
		{"properties": {
			"o": {"type": "object"},
			"a": {"type": "array"},
			"n": {"type": "number"},
			"s": {"type": "string"},
			"z": {"type": "null"},
			"b": {"type": "boolean"},

			"t0": {"prefixItems": [{}], "items": {"type": "number"}},
			"t1": {"items": {"type": "number"}, "maxItems": 3},

			"v": {"maxItems": 3}
		}}
	`

	var s *Schema
	if err := json.Unmarshal([]byte(schema), &s); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	var got []string
	for _, p := range s.Properties {
		got = append(got, p.EffectiveType())
	}

	want := strings.Fields(`
		object
		array
		number
		string
		null
		boolean
		array
		array
		value
	`)
	if !reflect.DeepEqual(want, got) {
		t.Errorf("\ngot:\n\t%v\nwant:\n\t%v", got, want)
	}
}
