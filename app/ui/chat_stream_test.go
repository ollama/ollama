//go:build windows || darwin

package ui

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestChatEventFromApiChatResponseUsesThinkingEvent(t *testing.T) {
	start := time.Now()

	event := chatEventFromApiChatResponse(api.ChatResponse{
		Message: api.Message{
			Thinking: "Let me think",
		},
	}, &start, nil)

	if event.EventName != "thinking" {
		t.Fatalf("event.EventName = %q, want %q", event.EventName, "thinking")
	}

	if event.Thinking == nil || *event.Thinking != "Let me think" {
		t.Fatalf("event.Thinking = %v, want %q", event.Thinking, "Let me think")
	}
}

func TestChatStreamEmitsLoadingKeepaliveDuringIdlePhases(t *testing.T) {
	oldInterval := chatKeepAliveInterval
	chatKeepAliveInterval = 10 * time.Millisecond
	defer func() {
		chatKeepAliveInterval = oldInterval
	}()

	oldNewInferenceClient := newInferenceClient
	defer func() {
		newInferenceClient = oldNewInferenceClient
	}()

	baseURL, err := url.Parse("http://ollama.test")
	if err != nil {
		t.Fatalf("failed to parse fake ollama URL: %v", err)
	}

	newInferenceClient = func() *api.Client {
		return api.NewClient(baseURL, &http.Client{
			Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				switch r.URL.Path {
				case "/api/show":
					return &http.Response{
						StatusCode: http.StatusOK,
						Status:     "200 OK",
						Header: http.Header{
							"Content-Type": []string{"application/json"},
						},
						Body:    io.NopCloser(strings.NewReader(`{}`)),
						Request: r,
					}, nil
				case "/api/chat":
					pr, pw := io.Pipe()
					go func() {
						time.Sleep(35 * time.Millisecond)
						err := json.NewEncoder(pw).Encode(api.ChatResponse{
							Message: api.Message{Content: "hello"},
						})
						if err != nil {
							pw.CloseWithError(err)
							return
						}
						pw.Close()
					}()

					return &http.Response{
						StatusCode: http.StatusOK,
						Status:     "200 OK",
						Header: http.Header{
							"Content-Type": []string{"application/x-ndjson"},
						},
						Body:    pr,
						Request: r,
					}, nil
				default:
					return &http.Response{
						StatusCode: http.StatusNotFound,
						Status:     "404 Not Found",
						Header: http.Header{
							"Content-Type": []string{"application/json"},
						},
						Body:    io.NopCloser(strings.NewReader(`{"error":"not found"}`)),
						Request: r,
					}, nil
				}
			}),
		})
	}

	testStore := &store.Store{
		DBPath: filepath.Join(t.TempDir(), "db.sqlite"),
	}
	defer testStore.Close()

	server := &Server{Store: testStore}

	req := httptest.NewRequest("POST", "/api/v1/chat/chat-1", bytes.NewBufferString(`{"model":"test-model","prompt":"hello"}`))
	req.SetPathValue("id", "chat-1")
	rr := httptest.NewRecorder()

	if err := server.chat(rr, req); err != nil {
		t.Fatalf("chat() error = %v", err)
	}

	lines := strings.Split(strings.TrimSpace(rr.Body.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected at least 3 streamed events, got %d: %q", len(lines), rr.Body.String())
	}

	var eventNames []string
	for _, line := range lines {
		var event struct {
			EventName string `json:"eventName"`
		}
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("failed to parse streamed event %q: %v", line, err)
		}
		eventNames = append(eventNames, event.EventName)
	}

	loadingIndex := -1
	chatIndex := -1
	doneIndex := -1
	for i, eventName := range eventNames {
		switch eventName {
		case "loading":
			if loadingIndex == -1 {
				loadingIndex = i
			}
		case "chat":
			if chatIndex == -1 {
				chatIndex = i
			}
		case "done":
			if doneIndex == -1 {
				doneIndex = i
			}
		}
	}

	if loadingIndex == -1 {
		t.Fatalf("expected a loading keepalive event, got %v", eventNames)
	}
	if chatIndex == -1 {
		t.Fatalf("expected a chat event, got %v", eventNames)
	}
	if doneIndex == -1 {
		t.Fatalf("expected a done event, got %v", eventNames)
	}
	if eventNames[len(eventNames)-1] != "done" {
		t.Fatalf("expected done to be the final event, got %v", eventNames)
	}
	if !(loadingIndex < chatIndex && chatIndex < doneIndex) {
		t.Fatalf("expected loading before chat before done, got %v", eventNames)
	}
}
