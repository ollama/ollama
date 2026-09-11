package jsonschema

import "strings"

// ToTypeScriptType converts a Schema to a TypeScript type string.
func (s *Schema) ToTypeScriptType() string {
	if s == nil {
		return "any"
	}

	if len(s.AnyOf) > 0 {
		types := make([]string, len(s.AnyOf))
		for i, anyOf := range s.AnyOf {
			types[i] = anyOf.ToTypeScriptType()
		}
		return strings.Join(types, " | ")
	}

	if s.Type != "" {
		return mapToTypeScriptType(s.Type)
	}

	if len(s.Types) == 0 {
		return "any"
	}

	var types []string
	for _, t := range s.Types {
		types = append(types, mapToTypeScriptType(t))
	}
	return strings.Join(types, " | ")
}

// mapToTypeScriptType maps JSON Schema types to TypeScript types
func mapToTypeScriptType(jsonType string) string {
	switch jsonType {
	case "string", "boolean", "null":
		return jsonType
	case "number", "integer":
		return "number"
	case "array":
		return "any[]"
	case "object":
		return "Record<string, any>"
	default:
		return "any"
	}
}
