package api

import (
	"encoding/json"
	"testing"
)

func TestToolParameterToTypeScriptType(t *testing.T) {
	tests := []struct {
		name     string
		param    ToolProperty
		expected string
	}{
		{
			name: "single string type",
			param: ToolProperty{
				Type: PropertyType{"string"},
			},
			expected: "string",
		},
		{
			name: "single number type",
			param: ToolProperty{
				Type: PropertyType{"number"},
			},
			expected: "number",
		},
		{
			name: "integer maps to number",
			param: ToolProperty{
				Type: PropertyType{"integer"},
			},
			expected: "number",
		},
		{
			name: "boolean type",
			param: ToolProperty{
				Type: PropertyType{"boolean"},
			},
			expected: "boolean",
		},
		{
			name: "array type",
			param: ToolProperty{
				Type: PropertyType{"array"},
			},
			expected: "any[]",
		},
		{
			name: "object type",
			param: ToolProperty{
				Type: PropertyType{"object"},
			},
			expected: "Record<string, any>",
		},
		{
			name: "null type",
			param: ToolProperty{
				Type: PropertyType{"null"},
			},
			expected: "null",
		},
		{
			name: "multiple types as union",
			param: ToolProperty{
				Type: PropertyType{"string", "number"},
			},
			expected: "string | number",
		},
		{
			name: "string or null union",
			param: ToolProperty{
				Type: PropertyType{"string", "null"},
			},
			expected: "string | null",
		},
		{
			name: "anyOf with single types",
			param: ToolProperty{
				AnyOf: []ToolProperty{
					{Type: PropertyType{"string"}},
					{Type: PropertyType{"number"}},
				},
			},
			expected: "string | number",
		},
		{
			name: "anyOf with multiple types in each branch",
			param: ToolProperty{
				AnyOf: []ToolProperty{
					{Type: PropertyType{"string", "null"}},
					{Type: PropertyType{"number"}},
				},
			},
			expected: "string | null | number",
		},
		{
			name: "nested anyOf",
			param: ToolProperty{
				AnyOf: []ToolProperty{
					{Type: PropertyType{"boolean"}},
					{
						AnyOf: []ToolProperty{
							{Type: PropertyType{"string"}},
							{Type: PropertyType{"number"}},
						},
					},
				},
			},
			expected: "boolean | string | number",
		},
		{
			name: "empty type returns any",
			param: ToolProperty{
				Type: PropertyType{},
			},
			expected: "any",
		},
		{
			name: "unknown type maps to any",
			param: ToolProperty{
				Type: PropertyType{"unknown_type"},
			},
			expected: "any",
		},
		{
			name: "multiple types including array",
			param: ToolProperty{
				Type: PropertyType{"string", "array", "null"},
			},
			expected: "string | any[] | null",
		},
		{
			name: "string enum renders as literal union",
			param: ToolProperty{
				Type: PropertyType{"string"},
				Enum: []any{"celsius", "fahrenheit"},
			},
			expected: `"celsius" | "fahrenheit"`,
		},
		{
			name: "enum escapes quotes in string values",
			param: ToolProperty{
				Type: PropertyType{"string"},
				Enum: []any{`a"b`},
			},
			expected: `"a\"b"`,
		},
		{
			name: "number and boolean enum values are not quoted",
			param: ToolProperty{
				Enum: []any{float64(1), 2.5, true, nil},
			},
			expected: "1 | 2.5 | true | null",
		},
		{
			name: "enum with a non-scalar value falls back to the declared type",
			param: ToolProperty{
				Type: PropertyType{"object"},
				Enum: []any{map[string]any{"a": 1}},
			},
			expected: "Record<string, any>",
		},
		{
			name: "array of enum items renders as a parenthesized union array",
			param: ToolProperty{
				Type:  PropertyType{"array"},
				Items: map[string]any{"type": "string", "enum": []any{"a", "b"}},
			},
			expected: `("a" | "b")[]`,
		},
		{
			name: "array without enum items stays any[]",
			param: ToolProperty{
				Type:  PropertyType{"array"},
				Items: map[string]any{"type": "string"},
			},
			expected: "any[]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.param.ToTypeScriptType()
			if result != tt.expected {
				t.Errorf("ToTypeScriptType() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestToolPropertyEnumFromJSONToTypeScriptType(t *testing.T) {
	var tools Tools
	body := `[{"type":"function","function":{"name":"get_current_weather","parameters":{"type":"object","properties":{
		"format":{"type":"string","enum":["celsius","fahrenheit"]},
		"days":{"type":"array","items":{"type":"string","enum":["mon","tue"]}},
		"count":{"type":"integer","enum":[1,2]}}}}}]`
	if err := json.Unmarshal([]byte(body), &tools); err != nil {
		t.Fatal(err)
	}

	want := map[string]string{
		"format": `"celsius" | "fahrenheit"`,
		"days":   `("mon" | "tue")[]`,
		"count":  "1 | 2",
	}
	props := tools[0].Function.Parameters.Properties
	for name, expected := range want {
		prop, ok := props.Get(name)
		if !ok {
			t.Fatalf("missing property %q", name)
		}
		if got := prop.ToTypeScriptType(); got != expected {
			t.Errorf("%s: ToTypeScriptType() = %q, want %q", name, got, expected)
		}
	}
}
