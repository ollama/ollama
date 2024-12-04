package llama

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestJsonSchema(t *testing.T) {
	testCases := []struct {
		name     string
		schema   JsonSchema
		expected string
	}{
		{
			name: "empty schema",
			schema: JsonSchema{
				Type: "object",
			},
			expected: `array ::= "[" space ( value ("," space value)* )? "]" space
boolean ::= ("true" | "false") space
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
decimal-part ::= [0-9]{1,16}
integral-part ::= [0] | [1-9] [0-9]{0,15}
null ::= "null" space
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
root ::= object
space ::= | " " | "\n" [ \t]{0,20}
string ::= "\"" char* "\"" space
value ::= object | array | string | number | boolean | null`,
		},
		{
			name: "invalid schema with circular reference",
			schema: JsonSchema{
				Type: "object",
				Properties: map[string]any{
					"self": map[string]any{
						"$ref": "#", // Self reference
					},
				},
			},
			expected: "", // Should return empty string for invalid schema
		},
		{
			name: "schema with invalid type",
			schema: JsonSchema{
				Type: "invalid_type", // Invalid type
				Properties: map[string]any{
					"foo": map[string]any{
						"type": "string",
					},
				},
			},
			expected: "", // Should return empty string for invalid schema
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.schema.AsGrammar()
			if !strings.EqualFold(strings.TrimSpace(result), strings.TrimSpace(tc.expected)) {
				if diff := cmp.Diff(tc.expected, result); diff != "" {
					t.Fatalf("grammar mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
