package applefm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// newFakeFM returns a client connected to handler in place of `fm serve`.
func newFakeFM(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return &Client{baseURL: srv.URL, done: make(chan struct{})}
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
	var got map[string]any
	c := newFakeFM(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&got)
		writeSSE(w,
			`data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}`,
			`data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}`,
			`data: {"choices":[{"delta":{"content":" world"},"index":0}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}`,
			`data: [DONE]`,
		)
	})

	schema := `{"type":"object","properties":{"city":{"type":"string"}}}`
	content, final, err := collectChat(t, c, llm.ChatRequest{
		Messages: []api.Message{
			{Role: "system", Content: "be terse"},
			{Role: "user", Content: "what is this?", Images: []api.ImageData{[]byte("\x89PNG\r\n\x1a\n")}},
		},
		Format:  json.RawMessage(schema),
		Options: &api.Options{Temperature: 0.5, TopK: 40, TopP: 0.5, Seed: -1, NumPredict: -1},
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

	// The unset seed and num_predict are left out.
	var want map[string]any
	if err := json.Unmarshal([]byte(`{
		"model": "system",
		"stream": true,
		"stream_options": {"include_usage": true},
		"temperature": 0.5,
		"top_k": 40,
		"top_p": 0.5,
		"response_format": {"type": "json_schema", "json_schema": {"name": "response", "schema": `+schema+`}},
		"messages": [
			{"role": "system", "content": "be terse"},
			{"role": "user", "content": [
				{"type": "text", "text": "what is this?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
			]}
		]
	}`), &want); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("request mismatch (-want +got):\n%s", diff)
	}
}

func TestCompletion(t *testing.T) {
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

	var content strings.Builder
	var final llm.CompletionResponse
	err := c.Completion(t.Context(), llm.CompletionRequest{
		Prompt:  "tell a story",
		Options: &api.Options{Seed: 7, NumPredict: 2},
	}, func(r llm.CompletionResponse) {
		content.WriteString(r.Content)
		if r.Done {
			final = r
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || got.Messages[0].Content != "tell a story" {
		t.Errorf("messages = %+v", got.Messages)
	}
	if got.Seed == nil || *got.Seed != 7 || got.MaxCompletionTokens == nil || *got.MaxCompletionTokens != 2 {
		t.Errorf("seed = %v, max_completion_tokens = %v", got.Seed, got.MaxCompletionTokens)
	}
	if content.String() != "Once upon" || final.DoneReason != llm.DoneReasonLength || final.PromptEvalCount != 5 || final.EvalCount != 2 {
		t.Errorf("content = %q final = %+v", content.String(), final)
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
			name: "stream ends early",
			handler: func(w http.ResponseWriter, r *http.Request) {
				writeSSE(w, `data: {"choices":[{"delta":{"content":"Once upon"},"index":0}]}`)
			},
			message: "ended the response early",
		},
		{
			name: "malformed chunk",
			handler: func(w http.ResponseWriter, r *http.Request) {
				writeSSE(w, `data: {"choices":[`)
			},
			message: "invalid stream chunk",
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
