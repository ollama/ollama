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
	  "output": { "type": "string" }
	},
	"required": ["explanation", "output"],
	"additionalProperties": false
      }
    },
    "final_answer": { "type": "string" }
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

	var sawSteps bool
	s := bufio.NewScanner(bytes.NewReader(g))
	for s.Scan() {
		line := s.Text()
		if strings.Contains(line, "steps") {
			sawSteps = true
		}
		if strings.Contains(line, "final-answer") && !sawSteps {
			t.Error("expected 'steps' before 'final-answer'")
		}
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
