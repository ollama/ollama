//go:build mlx

package schema

import (
	"testing"

	gram "github.com/ollama/ollama/x/grammar"
	"github.com/ollama/ollama/x/imagegen/mlx"
)

func TestJSONEBNF(t *testing.T) {
	tests := []struct {
		name   string
		schema string
	}{
		{
			name: "simple object",
			schema: `{
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"age": {"type": "integer"}
				},
				"required": ["name", "age"]
			}`,
		},
		{
			name: "with enum",
			schema: `{
				"type": "object",
				"properties": {
					"status": {"enum": ["active", "inactive", "pending"]}
				},
				"required": ["status"]
			}`,
		},
		{
			name: "array of objects",
			schema: `{
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"id": {"type": "integer"}
					},
					"required": ["id"]
				}
			}`,
		},
		{
			name: "nested object",
			schema: `{
				"type": "object",
				"properties": {
					"user": {
						"type": "object",
						"properties": {
							"email": {"type": "string"}
						},
						"required": ["email"]
					}
				},
				"required": ["user"]
			}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ebnf, err := EBNF(tc.schema)
			if err != nil {
				t.Fatalf("EBNF failed: %v", err)
			}

			// Try to compile it
			grammar, err := gram.ParseEBNF(ebnf, "root")
			if err != nil {
				t.Fatalf("ParseEBNF failed: %v", err)
			}

			if grammar == nil {
				t.Fatal("grammar is nil")
			}
		})
	}
}

func TestGrammarEngine(t *testing.T) {
	schema := `{
		"type": "object",
		"properties": {
			"name": {"type": "string"},
			"age": {"type": "integer"}
		},
		"required": ["name", "age"]
	}`

	grammar, err := Grammar(schema)
	if err != nil {
		t.Fatalf("Grammar failed: %v", err)
	}

	vocab := []string{
		"{", "}", "[", "]", ":", ",",
		"\"name\"", "\"age\"", "\"test\"",
		"\"", "a", "b", "c",
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		" ", "\n",
		"true", "false", "null",
	}

	engine, err := gram.NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("grammar.NewEngine failed: %v", err)
	}
	defer engine.Close()

	logits := mlx.Ones(int32(len(vocab)))
	mlx.Keep(logits)

	// Test that we can apply mask
	masked := engine.ApplyMask(logits)
	mlx.Eval(masked)
}

// TestOpenAIStructuredOutputs tests features required for OpenAI compatibility
func TestOpenAIStructuredOutputs(t *testing.T) {
	tests := []struct {
		name   string
		schema string
	}{
		{
			name: "anyOf union",
			schema: `{
				"type": "object",
				"properties": {
					"value": {
						"anyOf": [
							{"type": "string"},
							{"type": "integer"}
						]
					}
				},
				"required": ["value"]
			}`,
		},
		{
			name: "nullable string via type array",
			schema: `{
				"type": "object",
				"properties": {
					"name": {"type": ["string", "null"]}
				},
				"required": ["name"]
			}`,
		},
		{
			name: "$ref with $defs",
			schema: `{
				"type": "object",
				"properties": {
					"person": {"$ref": "#/$defs/Person"}
				},
				"required": ["person"],
				"$defs": {
					"Person": {
						"type": "object",
						"properties": {
							"name": {"type": "string"},
							"age": {"type": "integer"}
						},
						"required": ["name", "age"]
					}
				}
			}`,
		},
		{
			name: "const value",
			schema: `{
				"type": "object",
				"properties": {
					"type": {"const": "user"}
				},
				"required": ["type"]
			}`,
		},
		{
			name: "format date-time",
			schema: `{
				"type": "object",
				"properties": {
					"created": {"type": "string", "format": "date-time"}
				},
				"required": ["created"]
			}`,
		},
		{
			name: "format date",
			schema: `{
				"type": "object",
				"properties": {
					"birthday": {"type": "string", "format": "date"}
				},
				"required": ["birthday"]
			}`,
		},
		{
			name: "format email",
			schema: `{
				"type": "object",
				"properties": {
					"email": {"type": "string", "format": "email"}
				},
				"required": ["email"]
			}`,
		},
		{
			name: "format uuid",
			schema: `{
				"type": "object",
				"properties": {
					"id": {"type": "string", "format": "uuid"}
				},
				"required": ["id"]
			}`,
		},
		{
			name: "array with minItems maxItems",
			schema: `{
				"type": "object",
				"properties": {
					"tags": {
						"type": "array",
						"items": {"type": "string"},
						"minItems": 1,
						"maxItems": 3
					}
				},
				"required": ["tags"]
			}`,
		},
		{
			name: "deeply nested with refs",
			schema: `{
				"type": "object",
				"properties": {
					"company": {
						"type": "object",
						"properties": {
							"name": {"type": "string"},
							"employees": {
								"type": "array",
								"items": {"$ref": "#/$defs/Employee"}
							}
						},
						"required": ["name", "employees"]
					}
				},
				"required": ["company"],
				"$defs": {
					"Employee": {
						"type": "object",
						"properties": {
							"name": {"type": "string"},
							"role": {"enum": ["engineer", "manager", "intern"]}
						},
						"required": ["name", "role"]
					}
				}
			}`,
		},
		{
			name: "multiple refs same def",
			schema: `{
				"type": "object",
				"properties": {
					"from": {"$ref": "#/$defs/Address"},
					"to": {"$ref": "#/$defs/Address"}
				},
				"required": ["from", "to"],
				"$defs": {
					"Address": {
						"type": "object",
						"properties": {
							"city": {"type": "string"},
							"zip": {"type": "string"}
						},
						"required": ["city", "zip"]
					}
				}
			}`,
		},
		{
			name: "oneOf variant",
			schema: `{
				"type": "object",
				"properties": {
					"result": {
						"oneOf": [
							{
								"type": "object",
								"properties": {"success": {"type": "boolean"}},
								"required": ["success"]
							},
							{
								"type": "object",
								"properties": {"error": {"type": "string"}},
								"required": ["error"]
							}
						]
					}
				},
				"required": ["result"]
			}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ebnf, err := EBNF(tc.schema)
			if err != nil {
				t.Fatalf("EBNF failed: %v", err)
			}

			grammar, err := gram.ParseEBNF(ebnf, "root")
			if err != nil {
				t.Fatalf("ParseEBNF failed: %v", err)
			}

			if grammar == nil {
				t.Fatal("grammar is nil")
			}
		})
	}
}
