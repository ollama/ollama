package applefm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// newFakeFM returns a client connected to handler over a unix socket, the way
// it talks to `fm serve`.
func newFakeFM(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()

	dir, err := os.MkdirTemp("", "fm") //nolint:usetesting // t.TempDir can exceed the 104 byte unix socket path limit on macOS
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })

	sock := filepath.Join(dir, "fm.sock")
	ln, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = ln
	srv.Start()
	t.Cleanup(srv.Close)

	return &Client{client: newSocketClient(sock), done: make(chan struct{})}
}

func writeSSE(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, e := range events {
		fmt.Fprintf(w, "%s\n\n", e)
	}
}

func collectChat(t *testing.T, c *Client, req llm.ChatRequest) (string, llm.ChatResponse, error) {
	t.Helper()
	var content strings.Builder
	var final llm.ChatResponse
	err := c.Chat(t.Context(), req, func(r llm.ChatResponse) {
		content.WriteString(r.Message.Content)
		if r.Done {
			final = r
		}
	})
	return content.String(), final, err
}

func TestChat(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n")

	var got chatCompletionRequest
	var gotRaw map[string]any
	c := newFakeFM(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		var raw json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			t.Fatal(err)
		}
		json.Unmarshal(raw, &got)
		json.Unmarshal(raw, &gotRaw)
		writeSSE(w,
			`data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}`,
			`data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}`,
			`data: {"choices":[{"delta":{"content":" world"},"index":0}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}`,
			`data: [DONE]`,
		)
	})

	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
	content, final, err := collectChat(t, c, llm.ChatRequest{
		Messages: []api.Message{
			{Role: "system", Content: "be terse"},
			{Role: "user", Content: "what is this?", Images: []api.ImageData{png}},
		},
		Format:  schema,
		Options: &api.Options{Temperature: 0.5, TopK: 40, TopP: 0.9, Seed: -1, NumPredict: -1},
	})
	if err != nil {
		t.Fatal(err)
	}

	if content != "Hello world" {
		t.Errorf("content = %q", content)
	}
	if !final.Done || final.DoneReason != llm.DoneReasonStop || final.PromptEvalCount != 12 || final.EvalCount != 3 {
		t.Errorf("final = %+v", final)
	}

	if got.Model != "system" || !got.Stream || !got.StreamOptions["include_usage"] {
		t.Errorf("request = %+v", got)
	}
	if *got.Temperature != 0.5 || *got.TopK != 40 || *got.TopP != 0.9 {
		t.Errorf("sampling = %v %v %v", *got.Temperature, *got.TopK, *got.TopP)
	}
	if got.Seed != nil || got.MaxCompletionTokens != nil {
		t.Errorf("unset options were sent: seed=%v max=%v", got.Seed, got.MaxCompletionTokens)
	}
	if got.ResponseFormat == nil || got.ResponseFormat.Type != "json_schema" || string(got.ResponseFormat.JSONSchema.Schema) != string(schema) {
		t.Errorf("response_format = %+v", got.ResponseFormat)
	}

	wantMessages := []any{
		map[string]any{"role": "system", "content": "be terse"},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "what is this?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,iVBORw0KGgo="}},
		}},
	}
	if diff := cmp.Diff(wantMessages, gotRaw["messages"]); diff != "" {
		t.Errorf("messages mismatch (-want +got):\n%s", diff)
	}
}

func TestChatTokenLimit(t *testing.T) {
	var got chatCompletionRequest
	c := newFakeFM(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		// fm reports "stop" even when it hits max_completion_tokens
		writeSSE(w,
			`data: {"choices":[{"delta":{"content":"Once upon"},"index":0}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}`,
			`data: [DONE]`,
		)
	})

	_, final, err := collectChat(t, c, llm.ChatRequest{
		Messages: []api.Message{{Role: "user", Content: "tell a story"}},
		Options:  &api.Options{NumPredict: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.MaxCompletionTokens == nil || *got.MaxCompletionTokens != 2 {
		t.Errorf("max_completion_tokens = %v", got.MaxCompletionTokens)
	}
	if final.DoneReason != llm.DoneReasonLength {
		t.Errorf("done reason = %v, want length", final.DoneReason)
	}
}

func TestChatErrors(t *testing.T) {
	cases := []struct {
		name    string
		handler http.HandlerFunc
		format  json.RawMessage
		status  int
		message string
	}{
		{
			name: "request rejected",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprint(w, `{"error":{"type":"invalid_request_error","message":"Invalid request: bad image","code":"400"}}`)
			},
			status:  http.StatusBadRequest,
			message: "Invalid request: bad image",
		},
		{
			name: "error mid stream",
			handler: func(w http.ResponseWriter, r *http.Request) {
				writeSSE(w,
					`data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}`,
					"event: error\ndata: {\"error\":{\"type\":\"server_error\",\"message\":\"The model's safety guardrails were triggered.\",\"code\":\"500\"}}",
				)
			},
			message: "The model's safety guardrails were triggered.",
		},
		{
			name: "json format",
			handler: func(w http.ResponseWriter, r *http.Request) {
				t.Error("fm should not be called")
			},
			format:  json.RawMessage(`"json"`),
			status:  http.StatusBadRequest,
			message: `format "json" is not supported`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newFakeFM(t, tc.handler)
			_, _, err := collectChat(t, c, llm.ChatRequest{
				Messages: []api.Message{{Role: "user", Content: "hi"}},
				Format:   tc.format,
			})
			if err == nil || !strings.Contains(err.Error(), tc.message) {
				t.Fatalf("error = %v, want %q", err, tc.message)
			}
			var serr api.StatusError
			if tc.status != 0 && (!errors.As(err, &serr) || serr.StatusCode != tc.status) {
				t.Fatalf("error = %#v, want status %d", err, tc.status)
			}
		})
	}
}

func TestCompletionSendsPromptAsUserMessage(t *testing.T) {
	var got chatCompletionRequest
	c := newFakeFM(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		writeSSE(w,
			`data: {"choices":[{"delta":{"content":"hi"},"index":0}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}`,
			`data: [DONE]`,
		)
	})

	var content strings.Builder
	var final llm.CompletionResponse
	err := c.Completion(t.Context(), llm.CompletionRequest{Prompt: "say hi", Options: &api.Options{Seed: 7}}, func(r llm.CompletionResponse) {
		content.WriteString(r.Content)
		if r.Done {
			final = r
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || got.Messages[0].Content != "say hi" {
		t.Errorf("messages = %+v", got.Messages)
	}
	if got.Seed == nil || *got.Seed != 7 {
		t.Errorf("seed = %v", got.Seed)
	}
	if content.String() != "hi" || !final.Done || final.PromptEvalCount != 4 || final.EvalCount != 1 {
		t.Errorf("content = %q final = %+v", content.String(), final)
	}
}
