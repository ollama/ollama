package llama

import (
	"bufio"
	"bytes"
	"strings"
	"testing"
)

// https://github.com/ollama/ollama/issues/7978
const issue7978JSONSchema = `{
  "type": "object",
  "properties": {
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "explanation": { "type": "string" },
          "output": { "type": "string" },
          "nested": {
            "type": "object",
            "properties": {
              "deep": { "type": "string" }
            }
          }
        },
        "required": ["explanation", "output"],
        "additionalProperties": false
      }
    },
    "final_answer": { "type": "string" },
    "01_numbered_key": { "type": "string" },
    "numbers": {
      "type": "array",
      "items": { "type": "number" }
    },
    "booleans": {
      "type": "array", 
      "items": { "type": "boolean" }
    },
    "mixed": {
      "type": "array",
      "items": {
        "oneOf": [
          { "type": "string" },
          { "type": "number" },
          { "type": "boolean" }
        ]
      }
    }
  },
  "required": ["steps", "final_answer"],
  "additionalProperties": false
}`

func TestIssue7978(t *testing.T) {
	g := SchemaToGrammar([]byte(issue7978JSONSchema))
	if g == nil {
		t.Fatal("failed to convert JSON schema to grammar")
	}

	t.Logf("grammar:\n%s", g)
	t.Log()

	var got string
	s := bufio.NewScanner(bytes.NewReader(g))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		step, _, _ := strings.Cut(line, " ::= ")
		step = strings.TrimSpace(step)
		if step == "root" {
			got = line
		}
	}

	want := `root ::= "{" space steps-kv "," space final-answer-kv ( "," space ( 01-numbered-key-kv 01-numbered-key-rest | numbers-kv numbers-rest | booleans-kv booleans-rest | mixed-kv ) )? "}" space`
	if got != want {
		t.Errorf("root =\n%qwant:\n%q", got, want)
	}
}

func TestSchemaToGrammer(t *testing.T) {
	cases := []struct {
		schema string
		prefix []byte // nil is check as nil
	}{
		{`invalid`, nil},

		// Simple heuristic/smoke test
		{`{"type":"object"}`, []byte("root ::= object")},
	}

	for _, c := range cases {
		t.Run("x", func(t *testing.T) {
			g := SchemaToGrammar([]byte(c.schema))
			if c.prefix == nil && g != nil {
				t.Fatalf("grammar = %v, want nil", g)
			}
			if !bytes.HasPrefix(g, c.prefix) {
				t.Errorf("grammar = %q, want %q", g, c.prefix)
			}
		})
	}
}


const growingJSONSchema = `{
    "type": "object",
    "properties": {
        "execSchema": { "type": "object", "additionalProperties": true },
        "execInstructions": { "type": "string" },
        "combineSchema": { "type": "object", "additionalProperties": true },
        "combineInstructions": { "type": "string" },
        "question": { "type": "string" }
    }
}`
const growingJSONSchemaGrammar = `root ::= "{" space  (execSchema-kv execSchema-rest | execInstructions-kv execInstructions-rest | combineSchema-kv combineSchema-rest | combineInstructions-kv combineInstructions-rest | question-kv )? "}" space
execSchema-rest ::= ( "," space execInstructions-kv )? execInstructions-rest
execInstructions-rest ::= ( "," space combineSchema-kv )? combineSchema-rest
combineInstructions-rest ::= ( "," space question-kv )?
question-kv ::= "\"question\"" space ":" space string
combineInstructions-kv ::= "\"combineInstructions\"" space ":" space string
combineSchema-kv ::= "\"combineSchema\"" space ":" space combineSchema
execInstructions-kv ::= "\"execInstructions\"" space ":" space string
combineSchema-rest ::= ( "," space combineInstructions-kv )? combineInstructions-rest
space ::= | " " | "\n"{1,2} [ \t]{0,20}
number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
boolean ::= ("true" | "false") space
combineSchema ::= object
string ::= "\"" char* "\"" space
value ::= object | array | string | number | boolean | null
integral-part ::= [0] | [1-9] [0-9]{0,15}
null ::= "null" space
execSchema-kv ::= "\"execSchema\"" space ":" space execSchema
array ::= "[" space ( value ("," space value)* )? "]" space
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
decimal-part ::= [0-9]{1,16}
execSchema ::= object
`

func TestGrowingSchema(t *testing.T) {
	g := SchemaToGrammar([]byte(growingJSONSchema))
	if g == nil {
		t.Fatal("failed to convert JSON schema to grammar")
	}

	if string(g) != growingJSONSchemaGrammar {
		t.Errorf("Mismatch!\ngot =\n%q\nwant:\n%q", g, growingJSONSchemaGrammar)
	}
}
