package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestStreamFollowUpChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		var request api.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Stream == nil || !*request.Stream {
			t.Fatalf("stream = %#v, want true", request.Stream)
		}
		if string(request.Format) != `{"type":"object"}` || request.Think == nil || request.Think.Value != "high" {
			t.Fatalf("follow-up controls were not preserved: format=%s think=%#v", request.Format, request.Think)
		}
		encoder := json.NewEncoder(w)
		if err := encoder.Encode(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "one"}}); err != nil {
			t.Fatal(err)
		}
		if err := encoder.Encode(api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "two"}}); err != nil {
			t.Fatal(err)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)

	var chunks []string
	base := api.ChatRequest{Model: "test-model", Format: json.RawMessage(`{"type":"object"}`), Think: &api.ThinkValue{Value: "high"}}
	if err := streamFollowUpChat(context.Background(), base, nil, nil, func(response api.ChatResponse) error {
		chunks = append(chunks, response.Message.Content)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 || chunks[0] != "one" || chunks[1] != "two" {
		t.Fatalf("chunks = %#v", chunks)
	}
}

func TestFindWebSearchToolCall(t *testing.T) {
	first := api.ToolCall{ID: "search_1", Function: api.ToolCallFunction{Name: "web_search"}}
	calls := []api.ToolCall{
		{ID: "client_1", Function: api.ToolCallFunction{Name: "get_weather"}},
		first,
		{ID: "search_2", Function: api.ToolCallFunction{Name: "web_search"}},
	}

	got, found, mixed := findWebSearchToolCall(calls)
	if !found || !mixed || got.ID != first.ID {
		t.Fatalf("call = %#v, found = %v, mixed = %v", got, found, mixed)
	}
}

func TestExtractQueryFromToolCall(t *testing.T) {
	tests := []struct {
		name string
		args api.ToolCallFunctionArguments
		want string
	}{
		{name: "valid", args: webSearchTestArgs("query", "test search"), want: "test search"},
		{name: "missing"},
		{name: "wrong key", args: webSearchTestArgs("other", "value")},
		{name: "wrong type", args: webSearchTestArgs("query", 42)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			call := api.ToolCall{Function: api.ToolCallFunction{Name: "web_search", Arguments: test.args}}
			if got := extractQueryFromToolCall(&call); got != test.want {
				t.Fatalf("query = %q, want %q", got, test.want)
			}
		})
	}
}

func webSearchTestArgs(key string, value any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	args.Set(key, value)
	return args
}

func TestBuildWebSearchAssistantMessage(t *testing.T) {
	call := api.ToolCall{ID: "search_1", Function: api.ToolCallFunction{Name: "web_search"}}
	response := api.ChatResponse{Message: api.Message{Content: "searching", Thinking: "need current data"}}

	message := buildWebSearchAssistantMessage(response, call)
	if message.Role != "assistant" || message.Content != response.Message.Content || message.Thinking != response.Message.Thinking || len(message.ToolCalls) != 1 || message.ToolCalls[0].ID != call.ID {
		t.Fatalf("message = %#v", message)
	}
}

func TestDoFollowUpChatPreservesHTTPErrorTypes(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		check  func(*testing.T, error)
	}{
		{
			name:   "authorization",
			status: http.StatusUnauthorized,
			body:   `{"error":"unauthorized","signin_url":"https://ollama.com/signin/followup"}`,
			check: func(t *testing.T, err error) {
				var authorizationError api.AuthorizationError
				if !errors.As(err, &authorizationError) || authorizationError.SigninURL != "https://ollama.com/signin/followup" {
					t.Fatalf("error = %#v, want AuthorizationError with sign-in URL", err)
				}
			},
		},
		{
			name:   "rate limit",
			status: http.StatusTooManyRequests,
			body:   `{"error":"slow down"}`,
			check: func(t *testing.T, err error) {
				var statusError api.StatusError
				if !errors.As(err, &statusError) || statusError.StatusCode != http.StatusTooManyRequests {
					t.Fatalf("error = %#v, want 429 StatusError", err)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)

			_, err := doFollowUpChat(context.Background(), api.ChatRequest{Model: "test-model"}, nil, nil)
			if err == nil {
				t.Fatal("expected error")
			}
			test.check(t, err)
		})
	}
}
