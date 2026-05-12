package api

import (
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
