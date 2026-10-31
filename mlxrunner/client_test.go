package mlxrunner

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	tag := `{"type":"structural_tag","format":{"type":"json_schema","json_schema":` + schema + `}}`
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
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"const_string","value":"</think>"},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{
			name: "schema after thinking with two closings",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"<|final|>", "<|final|>json"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["<|final|>","<|final|>json"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"<|final|>"},{"type":"const_string","value":"<|final|>json"}]},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
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

func TestClientCompletionTokenRepeatLimit(t *testing.T) {
	for _, tt := range []struct {
		name    string
		token   string
		repeats int
		wantErr bool
	}{
		{name: "at limit", token: "ab", repeats: 101},
		{name: "past limit", token: "ab", repeats: 102, wantErr: true},
		// The final response carries no text, so it must not extend a whitespace run.
		{name: "whitespace before done", token: "\n", repeats: 100},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				enc := json.NewEncoder(w)
				for range tt.repeats {
					_ = enc.Encode(CompletionResponse{Content: tt.token})
				}
				_ = enc.Encode(CompletionResponse{Done: true})
			}))
			defer srv.Close()

			client := &Client{port: srv.Listener.Addr().(*net.TCPAddr).Port, client: srv.Client()}
			var streamed int
			err := client.Completion(t.Context(), llm.CompletionRequest{}, func(r llm.CompletionResponse) {
				if r.Content != "" {
					streamed++
				}
			})
			if tt.wantErr {
				if err == nil || err.Error() != "prediction aborted, token repeat limit reached" || streamed != 101 {
					t.Fatalf("streamed %d tokens, err %v; want 101 tokens then the repeat limit error", streamed, err)
				}
			} else if err != nil || streamed != tt.repeats {
				t.Fatalf("streamed %d tokens, err %v; want %d tokens and no error", streamed, err, tt.repeats)
			}
		})
	}
}
