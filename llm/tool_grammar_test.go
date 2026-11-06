package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestCompletionToolFormat(t *testing.T) {
	for _, tc := range []struct {
		name     string
		format   json.RawMessage
		thinking []string
		fail     bool
	}{
		{name: "unformatted tools"},
		{name: "JSON tools", format: json.RawMessage(`"json"`)},
		{name: "schema tools", format: json.RawMessage(`{"type":"object","title":"tool-format"}`)},
		{name: "thinking and tools", format: json.RawMessage(`"json"`), thinking: []string{"</think>"}},
		{name: "invalid schema fails closed", format: json.RawMessage(`{"type":"invalid-tool-format"}`), fail: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var completions int
			var grammar string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				var body struct {
					Grammar        string          `json:"grammar"`
					Schema         json.RawMessage `json:"json_schema"`
					ResponseFields []string        `json:"response_fields"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if len(body.ResponseFields) > 0 {
					if string(body.Schema) != string(tc.format) {
						t.Errorf("schema changed: %s", body.Schema)
					}
					if tc.fail {
						http.Error(w, "invalid schema", http.StatusBadRequest)
						return
					}
					json.NewEncoder(w).Encode(map[string]string{"generation_settings/grammar": "root ::= \"{}\"\n"})
					return
				}
				completions++
				grammar = body.Grammar
				if body.Schema != nil {
					t.Error("uncomposed schema would exclude tool syntax")
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos"}`)
			}))
			defer srv.Close()
			var port int
			fmt.Sscanf(srv.URL, "http://127.0.0.1:%d", &port)
			cmd := fakeRunningCmd()
			t.Cleanup(func() {
				cmd.Process.Kill()
				cmd.Wait()
			})
			runner := &llamaServerRunner{port: port, cmd: cmd, sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
			opts := api.DefaultOptions()
			err := runner.Completion(t.Context(), CompletionRequest{
				Prompt: "inspect before deciding", Format: tc.format, Options: &opts,
				ToolCallOpen: "<tool_call>", ToolCallClose: "</tool_call>", ThinkingClose: tc.thinking,
			}, func(CompletionResponse) {})
			if tc.fail {
				if err == nil || completions != 0 {
					t.Fatalf("invalid schema must not generate unconstrained output: err=%v completions=%d", err, completions)
				}
				return
			}
			if err != nil || completions != 1 {
				t.Fatalf("completion: err=%v count=%d", err, completions)
			}
			if len(tc.format) == 0 {
				if grammar != "" {
					t.Error("unformatted tools acquired an output constraint")
				}
				return
			}
			if !strings.Contains(grammar, `"<tool_call>"`) || !strings.Contains(grammar, "format") {
				t.Errorf("missing tool or text alternative: %s", grammar)
			}
			if len(tc.thinking) > 0 && !strings.HasPrefix(grammar, "root ::= ollama-ollama-thinking-0\n") {
				t.Error("thinking must precede both tool calls and structured text")
			}
		})
	}
}
