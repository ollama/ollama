package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// newTestRequestLog redirects the request log into a temp dir and resets the global
// slot list, so each test starts at ollama_requests-0.log.
func newTestRequestLog(t *testing.T) (*requestLogger, func() string) {
	t.Helper()

	dir := t.TempDir()
	t.Setenv("OLLAMA_REQUEST_LOG_DIR", dir)

	requestLogsMu.Lock()
	saved := requestLogSlots
	requestLogSlots = nil
	requestLogsMu.Unlock()
	t.Cleanup(func() {
		requestLogsMu.Lock()
		requestLogSlots = saved
		requestLogsMu.Unlock()
	})

	l, err := openRequestLog()
	if err != nil {
		t.Fatalf("openRequestLog: %v", err)
	}

	read := func() string {
		b, err := os.ReadFile(filepath.Join(dir, "ollama_requests-0.log"))
		if err != nil {
			t.Fatalf("read log: %v", err)
		}
		return string(b)
	}
	return l, read
}

func testTool() api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("lines", api.ToolProperty{Type: api.PropertyType{"integer"}})
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "read_file",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   []string{"path"},
				Properties: props,
			},
		},
	}
}

func testToolCall(name string) api.ToolCall {
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "/etc/hosts")
	args.Set("lines", 10)
	return api.ToolCall{
		ID:       "call_abc",
		Function: api.ToolCallFunction{Name: name, Arguments: args},
	}
}

// TestRequestLogToolCalls is the regression test for the actual bug: a response that
// consists only of tool calls was logged empty, because tool calls never go through
// writeChunk.
func TestRequestLogToolCalls(t *testing.T) {
	l, read := newTestRequestLog(t)

	l.writeRequestHeader("test-model", nil, requestMeta{}, "[user]: read it\n")
	l.writeToolCalls([]api.ToolCall{testToolCall("read_file")})
	l.writeDone(time.Now(), "stop", api.Metrics{})
	l.close()

	got := read()
	for _, want := range []string{
		"TOOL CALLS (1):",
		`[0] read_file({"path":"/etc/hosts","lines":10})`,
		"id=call_abc",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("log is missing %q\n--- log ---\n%s", want, got)
		}
	}

	// The tool calls must stand before the DONE block, not after it.
	if i, j := strings.Index(got, "TOOL CALLS"), strings.Index(got, "DONE "); i < 0 || j < 0 || i > j {
		t.Errorf("TOOL CALLS at %d, DONE at %d — expected tool calls first\n--- log ---\n%s", i, j, got)
	}
}

func TestRequestLogToolCallsEmpty(t *testing.T) {
	l, read := newTestRequestLog(t)

	l.writeRequestHeader("test-model", nil, requestMeta{}, "[user]: hi\n")
	l.writeChunk("hello")
	l.writeToolCalls(nil)
	l.writeDone(time.Now(), "stop", api.Metrics{})
	l.close()

	if got := read(); strings.Contains(got, "TOOL CALLS") {
		t.Errorf("no tool calls, but a TOOL CALLS block was written\n--- log ---\n%s", got)
	}
}

// TestRequestLogHeaderMeta covers the fields a tool-call bug needs: a model that
// ignored a tool and one that was never offered one must be distinguishable.
func TestRequestLogHeaderMeta(t *testing.T) {
	l, read := newTestRequestLog(t)

	l.writeRequestHeader("test-model", map[string]any{"temperature": 0.5}, requestMeta{
		Tools:  api.Tools{testTool()},
		Format: json.RawMessage(`{ "type" : "object" }`),
		Think:  &api.ThinkValue{Value: "high"},
		Images: 2,
	}, "[user]: hi\n")
	l.close()

	got := read()
	for _, want := range []string{
		"REQUEST ",
		"model=test-model",
		"temperature: 0.5",
		"TOOLS (1):",
		"read_file(path string*, lines integer)",
		`FORMAT: {"type":"object"}`,
		`THINK: "high"`,
		"IMAGES: 2",
		"PROMPT/MESSAGES:",
		"RESPONSE ",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("header is missing %q\n--- log ---\n%s", want, got)
		}
	}
}

func TestRequestLogHeaderMetaOmitted(t *testing.T) {
	l, read := newTestRequestLog(t)

	l.writeRequestHeader("test-model", nil, requestMeta{}, "[user]: hi\n")
	l.close()

	got := read()
	for _, unwanted := range []string{"TOOLS", "FORMAT", "THINK", "IMAGES", "OPTIONS"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("empty meta, but %q appears in the header\n--- log ---\n%s", unwanted, got)
		}
	}
}

func TestRequestLogError(t *testing.T) {
	l, read := newTestRequestLog(t)

	l.writeRequestHeader("test-model", nil, requestMeta{}, "[user]: hi\n")
	l.writeChunk("half a sent")
	l.writeError(os.ErrDeadlineExceeded)
	l.close()

	got := read()
	if !strings.Contains(got, "ERROR ") || !strings.Contains(got, os.ErrDeadlineExceeded.Error()) {
		t.Errorf("broken-off request was not marked\n--- log ---\n%s", got)
	}
}

// TestFormatRequestMessages: the first line of each message must keep the
// "[role]: content" shape — ollama_server.sh's trace --short filters on it.
func TestFormatRequestMessages(t *testing.T) {
	msgs := []api.Message{
		{Role: "system", Content: "be brief"},
		{Role: "user", Content: "read /etc/hosts"},
		{
			Role:      "assistant",
			Thinking:  "needs the file",
			ToolCalls: []api.ToolCall{testToolCall("read_file")},
		},
		{Role: "tool", Content: "127.0.0.1 localhost", ToolName: "read_file", ToolCallID: "call_abc"},
		{Role: "user", Content: "and this picture?", Images: []api.ImageData{{1, 2}, {3}}},
	}

	got := formatRequestMessages(msgs)

	for _, want := range []string{
		"[system]: be brief\n",
		"[user]: read /etc/hosts\n",
		"[assistant]: \n",
		"   thinking: needs the file\n",
		`   tool_call[0]: read_file({"path":"/etc/hosts","lines":10})  id=call_abc` + "\n",
		"[tool]: 127.0.0.1 localhost\n",
		"   tool_name: read_file\n",
		"   tool_call_id: call_abc\n",
		"   images: 2\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q\n--- got ---\n%s", want, got)
		}
	}

	// Every extra line must be indented, otherwise it looks like a new message to the
	// trace filter.
	for _, line := range strings.Split(strings.TrimRight(got, "\n"), "\n") {
		if strings.HasPrefix(line, "[") {
			continue
		}
		if !strings.HasPrefix(line, "   ") {
			t.Errorf("line is neither a role marker nor indented: %q", line)
		}
	}
}

func TestFormatToolSignature(t *testing.T) {
	if got, want := formatToolSignature(testTool()), "read_file(path string*, lines integer)"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// No parameters at all, and a nil Properties map, must not panic.
	bare := api.Tool{Function: api.ToolFunction{Name: "now"}}
	if got, want := formatToolSignature(bare), "now()"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
