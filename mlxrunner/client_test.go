package mlxrunner

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	jsonFormat := `{"type":"json_schema","json_schema":` + schema + `}`
	tag := `{"type":"structural_tag","format":` + jsonFormat + `}`
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
		want string
	}{
		{name: "unset"},
		{name: "null", req: llm.CompletionRequest{Format: json.RawMessage(`null`)}},
		{name: "empty", req: llm.CompletionRequest{Format: json.RawMessage(`""`)}},
		{
			name: "json",
			req:  llm.CompletionRequest{Format: json.RawMessage(`"json"`)},
			want: `{"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object"}}}`,
		},
		{name: "schema", req: llm.CompletionRequest{Format: json.RawMessage(schema)}, want: tag},
		{
			name: "schema after thinking",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"</think>"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["</think>"]},` +
				`{"type":"const_string","value":"</think>"},{"type":"json_schema","json_schema":` + schema + `}]}}`,
		},
		{
			name: "schema after thinking with two closings",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"<|final|>", "<|final|>json"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["<|final|>","<|final|>json"]},` +
				`{"type":"or","elements":[{"type":"const_string","value":"<|final|>"},{"type":"const_string","value":"<|final|>json"}]},{"type":"json_schema","json_schema":` + schema + `}]}}`,
		},
		{
			name: "direct schema or opened thinking",
			req: llm.CompletionRequest{
				Format:        json.RawMessage(schema),
				ThinkingOpen:  []string{"<|channel>"},
				ThinkingClose: []string{"<channel|>"},
			},
			want: `{"type":"structural_tag","format":{"type":"or","elements":[` + jsonFormat +
				`,{"type":"sequence","elements":[{"type":"const_string","value":"<|channel>"},{"type":"sequence","elements":[{"type":"any_text","excludes":["<channel|>"]},{"type":"const_string","value":"<channel|>"},{"type":"json_schema","json_schema":` + schema + `}]}]}]}}`,
		},
		{name: "thinking without a format", req: llm.CompletionRequest{ThinkingClose: []string{"</think>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(requestGrammar(tt.req)); got != tt.want {
				t.Fatalf("requestGrammar = %s, want %s", got, tt.want)
			}
		})
	}
}
