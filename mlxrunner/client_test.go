package mlxrunner

import (
	"encoding/json"
	"flag"
	"io"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
)

func TestClientCloseHelper(t *testing.T) {
	args := flag.Args()
	if len(args) != 1 {
		return
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
	if args[0] == "stuck" {
		// Model a native operation which does not return after shutdown.
		time.Sleep(2 * runnerShutdownTimeout)
	}
}

func TestClientClose(t *testing.T) {
	for _, mode := range []string{"graceful", "stuck"} {
		t.Run(mode, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestClientCloseHelper$", "--", mode)
			stdin, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			defer stdin.Close()
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			c := &Client{cmd: cmd, stdin: stdin, done: make(chan struct{})}
			go func() { c.doneErr = cmd.Wait(); close(c.done) }()
			defer func() { _ = cmd.Process.Kill(); <-c.done }()
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if (c.doneErr == nil) != (mode == "graceful") {
				t.Fatalf("unexpected process exit: %v", c.doneErr)
			}
			if !c.closed || c.cmd != nil || c.stdin != nil {
				t.Fatalf("incomplete close: closed=%v, command=%v, stdin=%v", c.closed, c.cmd, c.stdin)
			}
			if err := c.Close(); err != nil {
				t.Fatalf("repeat close: %v", err)
			}
		})
	}
}

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
