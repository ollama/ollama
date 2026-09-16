package jsonschema

import (
	"testing"
)

func TestToTypeScriptType(t *testing.T) {
	tests := []struct {
		name   string
		schema *Schema
		want   string
	}{
		{"nil", nil, "any"},
		{"empty", &Schema{}, "any"},
		{"string", &Schema{Type: "string"}, "string"},
		{"integer", &Schema{Type: "integer"}, "number"},
		{"number", &Schema{Type: "number"}, "number"},
		{"boolean", &Schema{Type: "boolean"}, "boolean"},
		{"array", &Schema{Type: "array"}, "any[]"},
		{"object", &Schema{Type: "object"}, "Record<string, any>"},
		{"null", &Schema{Type: "null"}, "null"},
		{"unknown", &Schema{Type: "custom"}, "any"},
		{"multiple types", &Schema{Types: []string{"string", "null"}}, "string | null"},
		{"anyOf", &Schema{AnyOf: []*Schema{{Type: "string"}, {Type: "number"}}}, "string | number"},
		{"anyOf with nil branch", &Schema{AnyOf: []*Schema{{Type: "string"}, nil}}, "string | any"},
		{"anyOf overrides type", &Schema{Type: "string", AnyOf: []*Schema{{Type: "boolean"}}}, "boolean"},
		{"nested anyOf", &Schema{
			AnyOf: []*Schema{
				{Type: "string"},
				{AnyOf: []*Schema{{Type: "number"}, {Type: "null"}}},
			},
		}, "string | number | null"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.schema.ToTypeScriptType(); got != tt.want {
				t.Errorf("ToTypeScriptType() = %q, want %q", got, tt.want)
			}
		})
	}
}
