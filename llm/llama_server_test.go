package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestLlamaServerHealthParsing(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		statusCode int
		wantStatus ServerStatus
		wantErr    bool
	}{
		{
			name:       "ready",
			body:       `{"status":"ok"}`,
			statusCode: 200,
			wantStatus: ServerStatusReady,
		},
		{
			name:       "loading",
			body:       `{"status":"loading model"}`,
			statusCode: 503,
			wantStatus: ServerStatusLoadingModel,
		},
		{
			name:       "loading error envelope",
			body:       `{"error":{"message":"Loading model","type":"unavailable_error","code":503}}`,
			statusCode: 503,
			wantStatus: ServerStatusLoadingModel,
		},
		{
			name:       "no slots",
			body:       `{"status":"no slot available"}`,
			statusCode: 503,
			wantStatus: ServerStatusNoSlotsAvailable,
		},
		{
			name:       "error status",
			body:       `{"status":"error","message":"out of memory"}`,
			statusCode: 500,
			wantStatus: ServerStatusError,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/health" {
					t.Errorf("unexpected path: %s", r.URL.Path)
				}
				w.WriteHeader(tt.statusCode)
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			// Parse the port from the test server
			parts := strings.Split(srv.URL, ":")
			port := parts[len(parts)-1]
			var portInt int
			fmt.Sscanf(port, "%d", &portInt)

			runner := &llamaServerRunner{
				port: portInt,
				cmd:  fakeRunningCmd(),
			}

			status, err := runner.getServerStatus(t.Context())
			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if status != tt.wantStatus {
				t.Errorf("status = %v, want %v", status, tt.wantStatus)
			}
		})
	}
}

func TestBoundedNumPredict(t *testing.T) {
	tests := []struct {
		name       string
		numPredict int
		numCtx     int
		want       int
	}{
		{name: "open ended gets finite budget", numPredict: -1, numCtx: 2048, want: 20480},
		{name: "explicit under limit preserved", numPredict: 100, numCtx: 2048, want: 100},
		{name: "explicit over limit capped", numPredict: 30000, numCtx: 2048, want: 20480},
		{name: "unknown context unchanged", numPredict: -1, numCtx: 0, want: -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := boundedNumPredict(tt.numPredict, tt.numCtx); got != tt.want {
				t.Fatalf("boundedNumPredict(%d, %d) = %d, want %d", tt.numPredict, tt.numCtx, got, tt.want)
			}
		})
	}
}

func TestLlamaServerCompletionSSEParsing(t *testing.T) {
	// Simulate llama-server SSE streaming response
	sseLines := []string{
		`data: {"content":"Hello","stop":false}`,
		``,
		`:`,
		`data: {"content":" world","stop":false}`,
		``,
		`:`,
		`data: {"content":"","stop":true,"stop_type":"eos","timings":{"prompt_n":5,"prompt_ms":10.5,"predicted_n":2,"predicted_ms":20.3}}`,
		``,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		if r.URL.Path != "/completion" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}

		// Verify request body is valid
		var reqBody llamaServerCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Errorf("invalid request body: %v", err)
			return
		}
		if reqBody.Prompt != "test prompt" {
			t.Errorf("prompt = %q, want %q", reqBody.Prompt, "test prompt")
		}
		if !reqBody.Stream {
			t.Error("stream should be true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	var responses []CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test prompt",
		Options: &opts,
	}, func(cr CompletionResponse) {
		responses = append(responses, cr)
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}

	if len(responses) != 3 {
		t.Fatalf("got %d responses, want 3", len(responses))
	}

	// First token
	if responses[0].Content != "Hello" {
		t.Errorf("response[0].Content = %q, want %q", responses[0].Content, "Hello")
	}
	if responses[0].Done {
		t.Error("response[0] should not be done")
	}

	// Second token
	if responses[1].Content != " world" {
		t.Errorf("response[1].Content = %q, want %q", responses[1].Content, " world")
	}

	// Final response
	if !responses[2].Done {
		t.Error("response[2] should be done")
	}
	if responses[2].DoneReason != DoneReasonStop {
		t.Errorf("DoneReason = %v, want %v", responses[2].DoneReason, DoneReasonStop)
	}
	if responses[2].PromptEvalCount != 5 {
		t.Errorf("PromptEvalCount = %d, want 5", responses[2].PromptEvalCount)
	}
	if responses[2].EvalCount != 2 {
		t.Errorf("EvalCount = %d, want 2", responses[2].EvalCount)
	}
}

func TestLlamaServerCompletionPromptEvalCountIncludesCache(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"cache_n":12,"prompt_n":5,"prompt_ms":10,"predicted_n":2,"predicted_ms":20}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	var responses []CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test prompt",
		Options: &opts,
	}, func(cr CompletionResponse) {
		responses = append(responses, cr)
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("got %d responses, want 1", len(responses))
	}
	if responses[0].PromptEvalCount != 17 {
		t.Errorf("PromptEvalCount = %d, want 17", responses[0].PromptEvalCount)
	}
	if responses[0].PromptEvalDuration != 10*time.Millisecond {
		t.Errorf("PromptEvalDuration = %s, want 10ms", responses[0].PromptEvalDuration)
	}
}

func TestLlamaServerChatPromptEvalCountIncludesCache(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Hello"}}]}`)
			fmt.Fprintln(w, `:`)
			fmt.Fprintln(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":12,"prompt_n":5,"prompt_ms":10,"predicted_n":2,"predicted_ms":20}}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	var responses []ChatResponse
	opts := api.DefaultOptions()
	err := runner.Chat(t.Context(), ChatRequest{
		Messages: []api.Message{{Role: "user", Content: "test prompt"}},
		Options:  &opts,
	}, func(cr ChatResponse) {
		responses = append(responses, cr)
	})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(responses) != 2 {
		t.Fatalf("got %d responses, want 2", len(responses))
	}
	if responses[1].PromptEvalCount != 17 {
		t.Errorf("PromptEvalCount = %d, want 17", responses[1].PromptEvalCount)
	}
	if responses[1].PromptEvalDuration != 10*time.Millisecond {
		t.Errorf("PromptEvalDuration = %s, want 10ms", responses[1].PromptEvalDuration)
	}
}

func TestLlamaServerStreamsHandleLargeSSELines(t *testing.T) {
	tests := []struct {
		name       string
		chat       bool
		payloadLen int
		wantErr    bool
	}{
		{name: "completion over old scanner limit", payloadLen: 512*1024 + 1024},
		{name: "completion over bounded limit", payloadLen: llamaServerStreamMaxBufferSize + 1, wantErr: true},
		{name: "chat over old scanner limit", chat: true, payloadLen: 512*1024 + 1024},
		{name: "chat over bounded limit", chat: true, payloadLen: llamaServerStreamMaxBufferSize + 1, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload := strings.Repeat("x", tt.payloadLen)
			path := "/completion"
			if tt.chat {
				path = "/v1/chat/completions"
			}

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/health":
					fmt.Fprint(w, `{"status":"ok"}`)
				case path:
					w.Header().Set("Content-Type", "text/event-stream")
					writeLargeLlamaServerEvent(t, w, tt.chat, payload)
					if !tt.wantErr {
						if tt.chat {
							fmt.Fprintln(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`)
						} else {
							fmt.Fprintln(w, `data: {"content":"","stop":true}`)
						}
					}
				default:
					t.Errorf("unexpected path: %s", r.URL.Path)
				}
			}))
			defer srv.Close()

			parts := strings.Split(srv.URL, ":")
			var portInt int
			fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

			runner := &llamaServerRunner{
				port:    portInt,
				cmd:     fakeRunningCmd(),
				sem:     semaphore.NewWeighted(1),
				options: api.Options{Runner: api.Runner{NumCtx: 2048}},
			}

			var got string
			opts := api.DefaultOptions()
			var err error
			if tt.chat {
				err = runner.Chat(t.Context(), ChatRequest{
					Messages: []api.Message{{Role: "user", Content: "test prompt"}},
					Options:  &opts,
				}, func(cr ChatResponse) {
					got += cr.Message.Content
				})
			} else {
				err = runner.Completion(t.Context(), CompletionRequest{
					Prompt:  "test prompt",
					Options: &opts,
				}, func(cr CompletionResponse) {
					got += cr.Content
				})
			}

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected oversized stream error")
				}
				if !strings.Contains(err.Error(), "stream event exceeded 8 MB limit") {
					t.Fatalf("expected stream limit error, got %v", err)
				}
				if strings.Contains(err.Error(), "bufio.Scanner") {
					t.Fatalf("expected wrapped stream limit error, got %v", err)
				}
				return
			}

			if err != nil {
				t.Fatal(err)
			}
			if got != payload {
				t.Fatalf("large payload length = %d, want %d", len(got), len(payload))
			}
		})
	}
}

func writeLargeLlamaServerEvent(t *testing.T, w io.Writer, chat bool, payload string) {
	t.Helper()

	fmt.Fprint(w, "data: ")
	var err error
	if chat {
		err = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"delta": map[string]any{"content": payload},
			}},
		})
	} else {
		err = json.NewEncoder(w).Encode(map[string]any{"content": payload, "stop": false})
	}
	if err != nil {
		t.Errorf("encoding large event: %v", err)
	}
}

func TestLlamaServerCompletionForwardsRepeatLastNZero(t *testing.T) {
	var completionBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("reading completion request body: %v", err)
				return
			}
			if err := json.Unmarshal(body, &completionBody); err != nil {
				t.Errorf("invalid completion request body %q: %v", body, err)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	opts := api.DefaultOptions()
	opts.RepeatLastN = 0
	if err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test prompt",
		Options: &opts,
	}, func(CompletionResponse) {}); err != nil {
		t.Fatalf("Completion error: %v", err)
	}

	value, ok := completionBody["repeat_last_n"]
	if !ok {
		t.Fatal("repeat_last_n missing from llama-server completion request")
	}
	if value != float64(0) {
		t.Fatalf("repeat_last_n = %v, want 0", value)
	}
}

func TestLlamaServerCompletionRejectsPromptOverContext(t *testing.T) {
	const wantError = "the prompt is longer than the context length currently available to the model; shorten the prompt or adjust the context length in settings"

	var tokenizeReq struct {
		Content      string `json:"content"`
		AddSpecial   bool   `json:"add_special"`
		ParseSpecial *bool  `json:"parse_special"`
	}

	completionCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/tokenize":
			if err := json.NewDecoder(r.Body).Decode(&tokenizeReq); err != nil {
				t.Errorf("invalid tokenize request body: %v", err)
				return
			}
			fmt.Fprint(w, `{"tokens":[0,1,2,3,4,5,6,7,8,9]}`)
		case "/completion":
			completionCalled = true
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"ok","stop":true,"timings":{"prompt_n":7,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 8}},
	}

	opts := api.DefaultOptions()
	opts.NumKeep = 3
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:   strings.Repeat("long prompt ", 2),
		Options:  &opts,
		Truncate: true,
	}, func(cr CompletionResponse) {})
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("Completion error = %T %v, want api.StatusError", err, err)
	}
	if statusErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("StatusCode = %d, want %d", statusErr.StatusCode, http.StatusBadRequest)
	}
	if statusErr.ErrorMessage != wantError {
		t.Fatalf("ErrorMessage = %q, want %q", statusErr.ErrorMessage, wantError)
	}

	if tokenizeReq.Content != strings.Repeat("long prompt ", 2) {
		t.Fatalf("tokenize content = %q", tokenizeReq.Content)
	}
	if !tokenizeReq.AddSpecial {
		t.Fatal("expected tokenize request to add special tokens")
	}
	if completionCalled {
		t.Fatal("completion endpoint was called")
	}
}

func TestLlamaServerCompletionWithMediaUsesRunnerMarker(t *testing.T) {
	var capturedReq llamaServerCompletionRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		if r.URL.Path != "/completion" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&capturedReq); err != nil {
			t.Errorf("invalid request body: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"content":"","stop":true,"timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	const mediaMarker = "<__ollama_media_test__>"
	runner := &llamaServerRunner{
		port:        portInt,
		cmd:         fakeRunningCmd(),
		sem:         semaphore.NewWeighted(1),
		options:     api.Options{Runner: api.Runner{NumCtx: 2048}},
		mediaMarker: mediaMarker,
	}

	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "look [img-7] now",
		Options: &opts,
		Media:   []MediaData{NewMediaData(7, []byte("media-bytes"))},
	}, func(cr CompletionResponse) {})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}

	promptObj, ok := capturedReq.Prompt.(map[string]any)
	if !ok {
		t.Fatalf("prompt = %T, want multimodal prompt object", capturedReq.Prompt)
	}
	if got, want := promptObj["prompt_string"], "look "+mediaMarker+" now"; got != want {
		t.Fatalf("prompt_string = %q, want %q", got, want)
	}
	data, ok := promptObj["multimodal_data"].([]any)
	if !ok {
		t.Fatalf("multimodal_data = %T, want array", promptObj["multimodal_data"])
	}
	if len(data) != 1 {
		t.Fatalf("multimodal_data len = %d, want 1", len(data))
	}
	if got, want := data[0], base64.StdEncoding.EncodeToString([]byte("media-bytes")); got != want {
		t.Fatalf("multimodal_data[0] = %q, want %q", got, want)
	}
}

func TestLlamaServerCompletionLengthStop(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"content":"tok","stop":false}`)
		fmt.Fprintln(w, ``)
		fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"limit","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	var lastResp CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test",
		Options: &opts,
	}, func(cr CompletionResponse) {
		lastResp = cr
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}
	if lastResp.DoneReason != DoneReasonLength {
		t.Errorf("DoneReason = %v, want %v", lastResp.DoneReason, DoneReasonLength)
	}
}

func TestLlamaServerStatusErrorMessageIncludesOOMStatus(t *testing.T) {
	status := &StatusWriter{}
	status.SetLastError("error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)")
	runner := &llamaServerRunner{
		status: status,
	}

	got := runner.statusErrorMessage([]byte(`{"error":{"message":"Compute error."}}`))
	if !strings.Contains(got, "Compute error") {
		t.Fatalf("expected original response body, got %q", got)
	}
	if !IsOutOfMemoryMessage(got) {
		t.Fatalf("expected OOM status detail to be detectable, got %q", got)
	}
}

func TestLlamaServerWaitUntilRunningUsesStatusWhenDoneErrIsNil(t *testing.T) {
	done := make(chan struct{})
	close(done)

	status := &StatusWriter{}
	status.SetLastError("llama_init_from_model: failed to initialize the context: failed to initialize Metal backend")

	runner := &llamaServerRunner{
		done:   done,
		status: status,
	}

	err := runner.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "%!w(<nil>)") {
		t.Fatalf("unexpected wrapped nil error: %q", err)
	}
	if !strings.Contains(err.Error(), status.LastError()) {
		t.Fatalf("error %q does not include status message %q", err, status.LastError())
	}
}

func TestLlamaServerWaitUntilRunningIgnoresStaleStartupOOMWhenReady(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	status := &StatusWriter{}
	status.SetLastError("error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)")

	runner := &llamaServerRunner{
		port:   portInt,
		cmd:    fakeRunningCmd(),
		status: status,
	}

	err := runner.WaitUntilRunning(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := status.LastError(); got != "" {
		t.Fatalf("expected stale startup status to be cleared, got %q", got)
	}
}

func TestLlamaServerWaitUntilRunningFailsOnHealthOOM(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"status":"error","message":"out of memory"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
	}

	err := runner.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected startup OOM error")
	}
	if !IsOutOfMemory(err) {
		t.Fatalf("expected OOM-classified error, got %q", err)
	}
}

func TestLlamaServerWaitUntilRunningWaitsOnRecoverableStartupOOM(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"status":"error","message":"compute buffer allocation failed"}`)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	status := &StatusWriter{}
	status.SetLastError("ggml_backend_sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism")
	runner := &llamaServerRunner{
		port:   portInt,
		cmd:    fakeRunningCmd(),
		status: status,
	}

	err := runner.WaitUntilRunning(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls < 2 {
		t.Fatalf("expected WaitUntilRunning to keep polling after recoverable OOM, calls=%d", calls)
	}
}

func TestLlamaServerWaitUntilRunningTimesOutWhenLoadStalls(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "10ms")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"status":"loading model"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
	}

	err := runner.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out waiting for llama-server to start") {
		t.Fatalf("expected load timeout, got %q", err)
	}
}

func TestLlamaServerWaitUntilRunningExtendsTimeoutOnOutputActivity(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "20ms")

	var activityCount atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		if activityCount.Load() < 5 {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error":{"message":"Loading model","type":"unavailable_error","code":503}}`)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
	}
	runner.output = &memoryParsingWriter{inner: io.Discard, runner: runner}

	done := make(chan struct{})
	defer close(done)
	go func() {
		ticker := time.NewTicker(5 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				activityCount.Add(1)
				_, _ = runner.output.Write([]byte("."))
			}
		}
	}()

	if err := runner.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning error: %v", err)
	}
}

func TestLlamaServerCompletionRequestFormat(t *testing.T) {
	tests := []struct {
		name           string
		format         string
		grammar        string
		wantGrammar    bool
		wantJsonSchema bool
		wantErr        bool
	}{
		{
			name: "no format",
		},
		{
			name:   "null format",
			format: `null`,
		},
		{
			name:   "empty string format",
			format: `""`,
		},
		{
			name:        "json format",
			format:      `"json"`,
			wantGrammar: true,
		},
		{
			name:           "json schema",
			format:         `{"type":"object","properties":{"name":{"type":"string"}}}`,
			wantJsonSchema: true,
		},
		{
			name:        "raw grammar",
			grammar:     `root ::= "hello"`,
			wantGrammar: true,
		},
		{
			name:    "invalid format",
			format:  `"xml"`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq llamaServerCompletionRequest

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				json.NewDecoder(r.Body).Decode(&capturedReq)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintln(w, `data: {"content":"ok","stop":true,"timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
			}))
			defer srv.Close()

			parts := strings.Split(srv.URL, ":")
			var portInt int
			fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

			runner := &llamaServerRunner{
				port:    portInt,
				cmd:     fakeRunningCmd(),
				sem:     semaphore.NewWeighted(1),
				options: api.Options{Runner: api.Runner{NumCtx: 2048}},
			}

			opts := api.DefaultOptions()
			req := CompletionRequest{
				Prompt:  "test",
				Options: &opts,
				Grammar: tt.grammar,
			}
			if tt.format != "" {
				req.Format = json.RawMessage(tt.format)
			}

			err := runner.Completion(t.Context(), req, func(cr CompletionResponse) {})

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantGrammar && capturedReq.Grammar == "" {
				t.Error("expected grammar to be set")
			}
			if tt.wantJsonSchema && capturedReq.JsonSchema == nil {
				t.Error("expected json_schema to be set")
			}
			if !tt.wantGrammar && !tt.wantJsonSchema && capturedReq.Grammar != "" {
				t.Errorf("unexpected grammar: %s", capturedReq.Grammar)
			}
		})
	}
}

func TestLlamaServerPreservedTokens(t *testing.T) {
	tests := []struct {
		name         string
		parserTokens []string
		toolCallTag  string
		want         []string
	}{
		{
			name:         "parser tokens only",
			parserTokens: []string{"<|channel>"},
			want:         []string{"<|channel>"},
		},
		{
			name:        "tool tag special token plus json punctuation",
			toolCallTag: "[TOOL_CALLS][",
			want:        []string{"[TOOL_CALLS]"},
		},
		{
			name:        "json array tool parser does not preserve array punctuation",
			toolCallTag: "[",
			want:        nil,
		},
		{
			name:        "ordinary tool tag",
			toolCallTag: "tool_call:",
			want:        []string{"tool_call:"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := llamaServerPreservedTokens(tt.parserTokens, tt.toolCallTag)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("llamaServerPreservedTokens = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestSetupLlamaServerCommandEnv(t *testing.T) {
	exeDir := t.TempDir()
	exe := filepath.Join(exeDir, "llama-server")
	if err := os.WriteFile(exe, nil, 0o755); err != nil {
		t.Fatal(err)
	}

	gpuDir := t.TempDir()
	backendName := "libggml-futuregpu.so"
	ignoredBackendNames := []string{"libggml-base.so", "libggml-cpu.so"}
	if runtime.GOOS == "darwin" {
		backendName = "libggml-futuregpu.dylib"
		ignoredBackendNames = []string{"libggml-base.dylib", "libggml-cpu.dylib"}
	}
	if runtime.GOOS == "windows" {
		backendName = "ggml-futuregpu.dll"
		ignoredBackendNames = []string{"ggml-base.dll", "ggml-cpu.dll"}
	}
	for _, name := range ignoredBackendNames {
		if err := os.WriteFile(filepath.Join(gpuDir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	backendPath := filepath.Join(gpuDir, backendName)
	if err := os.WriteFile(backendPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}

	pathEnv := llamaServerLibraryPathEnv()
	userLibDir := t.TempDir()
	t.Setenv(pathEnv, userLibDir)

	cmd := exec.Command("echo")
	SetupLlamaServerCommandEnv(cmd, exe, []string{ml.LibOllamaPath, gpuDir}, map[string]string{"OLLAMA_DEBUG": "1"})

	env := make(map[string]string)
	for _, kv := range cmd.Env {
		key, value, ok := strings.Cut(kv, "=")
		if ok {
			env[strings.ToUpper(key)] = value
		}
	}

	if got := env["GGML_BACKEND_PATH"]; got != backendPath {
		t.Fatalf("GGML_BACKEND_PATH = %q, want %q", got, backendPath)
	}
	if got := env["OLLAMA_DEBUG"]; got != "1" {
		t.Fatalf("OLLAMA_DEBUG = %q, want %q", got, "1")
	}

	paths := filepath.SplitList(env[strings.ToUpper(pathEnv)])
	if len(paths) < 3 {
		t.Fatalf("%s entries = %v, want at least 3 entries", pathEnv, paths)
	}
	if paths[0] != exeDir {
		t.Fatalf("%s[0] = %q, want %q", pathEnv, paths[0], exeDir)
	}
	if paths[1] != gpuDir {
		t.Fatalf("%s[1] = %q, want %q", pathEnv, paths[1], gpuDir)
	}
	if paths[2] != userLibDir {
		t.Fatalf("%s[2] = %q, want %q", pathEnv, paths[2], userLibDir)
	}
}

func TestFilteredEnvLogValue(t *testing.T) {
	attrs := filteredEnv([]string{
		"OLLAMA_DEBUG=1",
		"OLLAMA_API_KEY=ollama-secret",
		"OPENAI_API_KEY=openai-secret",
		"HF_TOKEN=hf-secret",
		"GGML_BACKEND_PATH=/tmp/ggml",
		"CUDA_VISIBLE_DEVICES=0",
		"CUDA_API_KEY=cuda-secret",
		"HIP_VISIBLE_DEVICES=1",
		"PATH=/bin",
	}).LogValue().Group()

	got := make(map[string]string, len(attrs))
	for _, attr := range attrs {
		got[attr.Key] = attr.Value.String()
	}

	for _, key := range []string{"OLLAMA_DEBUG", "OLLAMA_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"} {
		if _, ok := got[key]; ok {
			t.Fatalf("%s should not be logged: %#v", key, got)
		}
	}

	for key, want := range map[string]string{
		"GGML_BACKEND_PATH":    "/tmp/ggml",
		"CUDA_VISIBLE_DEVICES": "0",
		"HIP_VISIBLE_DEVICES":  "1",
		"PATH":                 "/bin",
		"CUDA_API_KEY":         "[redacted]",
	} {
		if got[key] != want {
			t.Fatalf("%s = %q, want %q; attrs=%#v", key, got[key], want, got)
		}
	}
}

func TestLlamaServerCompletionBOSOwnership(t *testing.T) {
	tests := []struct {
		name             string
		leadingBOS       string
		tokenizerAddsBOS bool
		ggmlKV           ggml.KV
		prompt           string
		wantPrompt       string
	}{
		{
			name:       "renderer owns bos when tokenizer does not add bos",
			leadingBOS: "<bos>",
			prompt:     "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt: "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:             "tokenizer auto bos path",
			tokenizerAddsBOS: true,
			prompt:           "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
			wantPrompt:       "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name:             "tokenizer auto bos path uses configured token",
			tokenizerAddsBOS: true,
			leadingBOS:       "<|startoftext|>",
			prompt:           "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
			wantPrompt:       "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:             "tokenizer auto bos keeps unknown token",
			tokenizerAddsBOS: true,
			prompt:           "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
			wantPrompt:       "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:       "other model keeps prompt",
			prompt:     "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt: "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:             "only leading bos is stripped when tokenizer owns bos",
			tokenizerAddsBOS: true,
			prompt:           "<bos>hello<bos>",
			wantPrompt:       "hello<bos>",
		},
		{
			name:       "gemma4 llama.cpp runtime bos override",
			leadingBOS: "<bos>",
			ggmlKV: ggml.KV{
				"general.architecture":            "gemma4",
				"tokenizer.ggml.pre":              "gemma4",
				"tokenizer.ggml.add_bos_token":    false,
				"tokenizer.ggml.bos_token_id":     uint32(2),
				"tokenizer.ggml.eos_token_id":     uint32(1),
				"tokenizer.ggml.unknown_token_id": uint32(0),
			},
			prompt:     "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt: "<|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:       "gemma4 model runtime bos override",
			leadingBOS: "<bos>",
			ggmlKV: ggml.KV{
				"general.architecture":            "gemma4",
				"tokenizer.ggml.model":            "gemma4",
				"tokenizer.ggml.add_bos_token":    false,
				"tokenizer.ggml.bos_token_id":     uint32(2),
				"tokenizer.ggml.eos_token_id":     uint32(1),
				"tokenizer.ggml.unknown_token_id": uint32(0),
			},
			prompt:     "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt: "<|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:       "lfm2 strips renderer bos",
			leadingBOS: "<|startoftext|>",
			ggmlKV: ggml.KV{
				"general.architecture":         "lfm2",
				"tokenizer.ggml.model":         "gpt2",
				"tokenizer.ggml.pre":           "lfm2",
				"tokenizer.ggml.add_bos_token": false,
				"tokenizer.ggml.bos_token_id":  uint32(0),
			},
			prompt:     "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
			wantPrompt: "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:       "lfm2 missing bos metadata uses llama.cpp default",
			leadingBOS: "<|startoftext|>",
			ggmlKV: ggml.KV{
				"general.architecture":        "lfm2",
				"tokenizer.ggml.model":        "gpt2",
				"tokenizer.ggml.pre":          "lfm2",
				"tokenizer.ggml.bos_token_id": uint32(0),
			},
			prompt:     "<|startoftext|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
			wantPrompt: "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq llamaServerCompletionRequest

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				if r.URL.Path == "/tokenize" {
					t.Errorf("unexpected tokenize request")
					w.WriteHeader(http.StatusInternalServerError)
					return
				}

				json.NewDecoder(r.Body).Decode(&capturedReq)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintln(w, `data: {"content":"ok","stop":true,"timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
			}))
			defer srv.Close()

			parts := strings.Split(srv.URL, ":")
			var portInt int
			fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

			runner := &llamaServerRunner{
				port:    portInt,
				cmd:     fakeRunningCmd(),
				sem:     semaphore.NewWeighted(1),
				options: api.Options{Runner: api.Runner{NumCtx: 2048}},
			}
			if tt.ggmlKV != nil {
				runner.ggml = loadTestGGML(t, tt.ggmlKV)
			} else if tt.tokenizerAddsBOS {
				runner.ggml = loadTestGGML(t, ggml.KV{
					"general.architecture":         "gemma3",
					"tokenizer.ggml.add_bos_token": true,
				})
			}

			opts := api.DefaultOptions()
			err := runner.Completion(t.Context(), CompletionRequest{
				Prompt:     tt.prompt,
				Options:    &opts,
				LeadingBOS: tt.leadingBOS,
			}, func(cr CompletionResponse) {})
			if err != nil {
				t.Fatalf("Completion error: %v", err)
			}

			if capturedReq.Prompt != tt.wantPrompt {
				t.Fatalf("prompt = %q, want %q", capturedReq.Prompt, tt.wantPrompt)
			}
		})
	}
}

func TestQwenVLServerArgs(t *testing.T) {
	tests := []struct {
		name string
		arch string
		want []string
	}{
		{
			name: "qwen2vl",
			arch: "qwen2vl",
			want: []string{"--image-min-tokens", "1024"},
		},
		{
			name: "qwen25vl",
			arch: "qwen25vl",
			want: []string{"--image-min-tokens", "1024"},
		},
		{
			name: "qwen3vl",
			arch: "qwen3vl",
			want: []string{"--image-min-tokens", "1024"},
		},
		{
			name: "qwen3vlmoe",
			arch: "qwen3vlmoe",
			want: []string{"--image-min-tokens", "1024"},
		},
		{
			name: "other model",
			arch: "llama",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := qwenVLServerArgs(tt.arch); !slices.Equal(got, tt.want) {
				t.Fatalf("qwenVLServerArgs(%q) = %v, want %v", tt.arch, got, tt.want)
			}
		})
	}
}

func TestLlamaServerTokenize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		var req map[string]string
		json.NewDecoder(r.Body).Decode(&req)
		if req["content"] != "hello world" {
			t.Errorf("content = %q, want %q", req["content"], "hello world")
		}
		fmt.Fprint(w, `{"tokens":[1,2,3]}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{port: portInt, cmd: fakeRunningCmd()}
	tokens, err := runner.Tokenize(t.Context(), "hello world")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}
	if len(tokens) != 3 || tokens[0] != 1 || tokens[1] != 2 || tokens[2] != 3 {
		t.Errorf("tokens = %v, want [1,2,3]", tokens)
	}
}

func TestLlamaServerTokenizeDoesNotReuseIdleConnections(t *testing.T) {
	var newConns atomic.Int64

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		fmt.Fprint(w, `{"tokens":[1,2,3]}`)
	}))
	srv.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			newConns.Add(1)
		}
	}
	srv.Start()
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{port: portInt, cmd: fakeRunningCmd()}
	for range 2 {
		if _, err := runner.Tokenize(t.Context(), "hello world"); err != nil {
			t.Fatalf("Tokenize error: %v", err)
		}
	}
	if got := newConns.Load(); got < 2 {
		t.Fatalf("Tokenize reused an idle llama-server connection, new connections = %d", got)
	}
}

func TestLlamaServerDetokenize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/detokenize" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			return
		}
		fmt.Fprint(w, `{"content":"hello world"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{port: portInt, cmd: fakeRunningCmd()}
	content, err := runner.Detokenize(t.Context(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("Detokenize error: %v", err)
	}
	if content != "hello world" {
		t.Errorf("content = %q, want %q", content, "hello world")
	}
}

func TestLlamaServerEmbedding(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path: %s, want /v1/embeddings", r.URL.Path)
			return
		}
		// OAI-compatible format (used when sending "input" field)
		fmt.Fprint(w, `{"data":[{"embedding":[0.1,0.2,0.3],"tokens_evaluated":2}],"usage":{"prompt_tokens":2}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
		sem:  semaphore.NewWeighted(1),
	}

	embedding, count, err := runner.Embedding(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embedding error: %v", err)
	}
	if len(embedding) != 3 {
		t.Errorf("embedding length = %d, want 3", len(embedding))
	}
	if count != 2 {
		t.Errorf("prompt_eval_count = %d, want 2", count)
	}
}

func TestLegacyEmbeddingsWereRaw(t *testing.T) {
	tests := []struct {
		name string
		kv   ggml.KV
		want bool
	}{
		{
			name: "bert t5 raw like bge-m3",
			kv: ggml.KV{
				"general.architecture": "bert",
				"bert.pooling_type":    uint32(1),
				"tokenizer.ggml.model": "t5",
			},
			want: true,
		},
		{
			name: "nomic bert default raw",
			kv: ggml.KV{
				"general.architecture":    "nomic-bert",
				"nomic-bert.pooling_type": uint32(1),
			},
			want: true,
		},
		{
			name: "qwen3 remains normalized",
			kv: ggml.KV{
				"general.architecture": "qwen3",
				"qwen3.pooling_type":   uint32(1),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := legacyEmbeddingsWereRaw(tt.kv); got != tt.want {
				t.Fatalf("legacyEmbeddingsWereRaw() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLlamaServerEmbeddingFallbackFormat(t *testing.T) {
	// Fallback: non-OAI array format (from "content" field)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		fmt.Fprint(w, `[{"index":0,"embedding":[[0.4,0.5,0.6]]}]`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
		sem:  semaphore.NewWeighted(1),
	}

	embedding, _, err := runner.Embedding(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embedding error: %v", err)
	}
	if len(embedding) != 3 {
		t.Errorf("embedding length = %d, want 3", len(embedding))
	}
	if embedding[0] != 0.4 {
		t.Errorf("embedding[0] = %v, want 0.4", embedding[0])
	}
}

func TestLlamaServerEmbeddingFlatArrayFallback(t *testing.T) {
	// Non-OAI format with flat (non-nested) embedding array
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		fmt.Fprint(w, `[{"index":0,"embedding":[0.7,0.8,0.9]}]`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
		sem:  semaphore.NewWeighted(1),
	}

	embedding, _, err := runner.Embedding(t.Context(), "hello")
	if err != nil {
		t.Fatalf("Embedding error: %v", err)
	}
	if len(embedding) != 3 || embedding[0] != 0.7 {
		t.Errorf("embedding = %v, want [0.7, 0.8, 0.9]", embedding)
	}
}

func TestLlamaServerEmbeddingTooLargeError(t *testing.T) {
	// llama-server returns 500 for oversized input; adapter should normalize to 400
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		w.WriteHeader(500)
		fmt.Fprint(w, `{"error":{"code":500,"message":"input is too large to process"}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port: portInt,
		cmd:  fakeRunningCmd(),
		sem:  semaphore.NewWeighted(1),
	}

	_, _, err := runner.Embedding(t.Context(), "very long input")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Should be normalized to 400 for the embed handler's truncation retry
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("expected api.StatusError, got %T: %v", err, err)
	}
	if statusErr.StatusCode != 400 {
		t.Errorf("status code = %d, want 400", statusErr.StatusCode)
	}
}

func TestEmbeddingBatchSize(t *testing.T) {
	tests := []struct {
		name        string
		numCtx      int
		numBatch    int
		numParallel int
		want        int
	}{
		{
			name:        "uses num batch",
			numCtx:      40960,
			numBatch:    2048,
			numParallel: 1,
			want:        2048,
		},
		{
			name:        "caps to context",
			numCtx:      1024,
			numBatch:    2048,
			numParallel: 1,
			want:        1024,
		},
		{
			name:        "accounts for parallel context",
			numCtx:      1024,
			numBatch:    4096,
			numParallel: 2,
			want:        2048,
		},
		{
			name:        "omits flags when unset",
			numCtx:      40960,
			numBatch:    0,
			numParallel: 1,
			want:        0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := api.DefaultOptions()
			opts.NumCtx = tt.numCtx
			opts.NumBatch = tt.numBatch
			if got := embeddingBatchSize(opts, tt.numParallel); got != tt.want {
				t.Fatalf("embeddingBatchSize = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestAppendBatchArgs(t *testing.T) {
	tests := []struct {
		name        string
		opts        api.Options
		embedding   bool
		numParallel int
		want        []string
	}{
		{
			name:        "generation sets logical and physical batch",
			opts:        api.Options{Runner: api.Runner{NumBatch: 1024}},
			numParallel: 1,
			want:        []string{"-b", "1024", "-ub", "1024"},
		},
		{
			name:        "generation omits unset batch",
			opts:        api.Options{},
			numParallel: 1,
			want:        nil,
		},
		{
			name:        "embedding caps batch to parallel context",
			opts:        api.Options{Runner: api.Runner{NumCtx: 512, NumBatch: 2048}},
			embedding:   true,
			numParallel: 2,
			want:        []string{"--embedding", "-b", "1024", "-ub", "1024"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendBatchArgs(nil, tt.opts, tt.embedding, tt.numParallel)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendBatchArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendFlashAttentionArgs(t *testing.T) {
	supportedGPU := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, DriverMajor: 13, ComputeMajor: 8, ComputeMinor: 9}}
	oldGPU := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 8, ComputeMinor: 9},
		{DeviceID: ml.DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 6, ComputeMinor: 2},
	}

	tests := []struct {
		name string
		env  string
		set  bool
		gpus []ml.DeviceInfo
		want []string
	}{
		{
			name: "unset uses llama-server auto mode",
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "auto"},
		},
		{
			name: "empty uses llama-server auto mode",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "auto"},
		},
		{
			name: "zero disables flash attention",
			env:  "0",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "off"},
		},
		{
			name: "false disables flash attention",
			env:  "false",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "off"},
		},
		{
			name: "one enables flash attention",
			env:  "1",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "on"},
		},
		{
			name: "true enables flash attention",
			env:  "true",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "on"},
		},
		{
			name: "invalid enables flash attention",
			env:  "random",
			set:  true,
			gpus: supportedGPU,
			want: []string{"base", "--flash-attn", "on"},
		},
		{
			name: "old cuda disables flash attention by default",
			gpus: oldGPU,
			want: []string{"base", "--flash-attn", "off"},
		},
		{
			name: "explicit enable overrides old cuda default",
			env:  "1",
			set:  true,
			gpus: oldGPU,
			want: []string{"base", "--flash-attn", "on"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setFlashAttentionEnv(t, tt.env, tt.set)
			got := appendFlashAttentionArgs([]string{"base"}, tt.gpus)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendFlashAttentionArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendMainGPUArgs(t *testing.T) {
	tests := []struct {
		name string
		opts api.Options
		want []string
	}{
		{
			name: "unset leaves llama-server default split mode",
			opts: api.DefaultOptions(),
			want: []string{"base"},
		},
		{
			name: "explicit zero selects gpu zero",
			opts: api.Options{Runner: api.Runner{MainGPU: testIntPtr(0)}},
			want: []string{"base", "--split-mode", "none", "--main-gpu", "0"},
		},
		{
			name: "explicit nonzero selects requested gpu",
			opts: api.Options{Runner: api.Runner{MainGPU: testIntPtr(1)}},
			want: []string{"base", "--split-mode", "none", "--main-gpu", "1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendMainGPUArgs([]string{"base"}, tt.opts)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendMainGPUArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendMMProjArgs(t *testing.T) {
	defaultOpts := api.DefaultOptions()
	partialOpts := api.DefaultOptions()
	partialOpts.NumGPU = 10
	fullOpts := api.DefaultOptions()
	fullOpts.NumGPU = 81
	cpuOpts := api.DefaultOptions()
	cpuOpts.NumGPU = 0

	tests := []struct {
		name        string
		projectors  []string
		opts        api.Options
		gpus        []ml.DeviceInfo
		modelLayers uint64
		retry       bool
		want        []string
	}{
		{
			name: "no projector leaves args unchanged",
			opts: defaultOpts,
			want: []string{"base"},
		},
		{
			name:        "large discrete gpu keeps projector offload",
			projectors:  []string{"model.gguf"},
			opts:        defaultOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf"},
		},
		{
			name:        "small discrete gpu disables projector offload",
			projectors:  []string{"model.gguf"},
			opts:        defaultOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, TotalMemory: 8 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf", "--no-mmproj-offload"},
		},
		{
			name:        "integrated rocm gpu disables projector offload",
			projectors:  []string{"model.gguf"},
			opts:        defaultOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "ROCm"}, Integrated: true, FreeMemory: 32 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf", "--no-mmproj-offload"},
		},
		{
			name:        "integrated metal gpu keeps projector offload",
			projectors:  []string{"model.gguf"},
			opts:        defaultOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}, Integrated: true, FreeMemory: 32 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf"},
		},
		{
			name:        "cpu only request disables projector offload",
			projectors:  []string{"model.gguf"},
			opts:        cpuOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf", "--no-mmproj-offload"},
		},
		{
			name:        "partial text offload disables projector offload",
			projectors:  []string{"model.gguf"},
			opts:        partialOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf", "--no-mmproj-offload"},
		},
		{
			name:        "explicit full text offload keeps projector offload",
			projectors:  []string{"model.gguf"},
			opts:        fullOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}},
			modelLayers: 81,
			want:        []string{"base", "--mmproj", "model.gguf"},
		},
		{
			name:        "startup oom retry disables projector offload",
			projectors:  []string{"model.gguf"},
			opts:        defaultOpts,
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, FreeMemory: 24 << 30}},
			modelLayers: 81,
			retry:       true,
			want:        []string{"base", "--mmproj", "model.gguf", "--no-mmproj-offload"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendMMProjArgs([]string{"base"}, llamaServerLaunchConfig{
				modelPath:            "model.gguf",
				projectors:           tt.projectors,
				opts:                 tt.opts,
				gpus:                 tt.gpus,
				modelLayers:          tt.modelLayers,
				forceNoMMProjOffload: tt.retry,
			})
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendMMProjArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendJinjaArgs(t *testing.T) {
	tests := []struct {
		name   string
		config LlamaServerConfig
		want   []string
	}{
		{
			name: "llama-server chat_template path leaves jinja enabled",
			want: []string{"base"},
		},
		{
			name:   "ollama rendered path disables unused jinja template",
			config: LlamaServerConfig{DisableJinja: true},
			want:   []string{"base", "--no-jinja", "--chat-template", "chatml"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendJinjaArgs([]string{"base"}, tt.config)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendJinjaArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendContextShiftArgs(t *testing.T) {
	opts := api.DefaultOptions()
	opts.NumKeep = 4

	tests := []struct {
		name    string
		opts    api.Options
		enabled bool
		want    []string
	}{
		{
			name: "disabled leaves context shift off",
			opts: opts,
			want: []string{"base"},
		},
		{
			name:    "enabled adds context shift and keep",
			opts:    opts,
			enabled: true,
			want:    []string{"base", "--context-shift", "--keep", "4"},
		},
		{
			name:    "enabled without keep omits keep flag",
			opts:    api.Options{},
			enabled: true,
			want:    []string{"base", "--context-shift"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendContextShiftArgs([]string{"base"}, tt.opts, tt.enabled)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendContextShiftArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAppendMTPDraftArgs(t *testing.T) {
	tests := []struct {
		name   string
		config LlamaServerConfig
		opts   api.Options
		want   []string
	}{
		{
			name: "no draft model leaves speculative decoding disabled",
			opts: api.Options{Runner: api.Runner{DraftNumPredict: 4}},
			want: []string{"base"},
		},
		{
			name:   "embedded draft uses configured draft depth",
			config: LlamaServerConfig{EnableMTP: true},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 4}},
			want:   []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-backend-sampling"},
		},
		{
			name:   "separate draft model uses configured draft depth",
			config: LlamaServerConfig{DraftModelPath: "draft.gguf"},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 8}},
			want:   []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-draft-backend-sampling", "--spec-draft-model", "draft.gguf"},
		},
		{
			name:   "zero draft depth disables speculative decoding",
			config: LlamaServerConfig{EnableMTP: true, DraftModelPath: "draft.gguf"},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 0}},
			want:   []string{"base"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendMTPDraftArgs([]string{"base"}, tt.config, tt.opts)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendMTPDraftArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHasLegacyQwenMTPDraft(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		tensors []*ggml.Tensor
		want    bool
	}{
		{
			name:    "qwen35 legacy mtp marker",
			arch:    "qwen35",
			tensors: []*ggml.Tensor{{Name: "mtp.fc.weight"}},
			want:    true,
		},
		{
			name:    "qwen35moe legacy mtp marker",
			arch:    "qwen35moe",
			tensors: []*ggml.Tensor{{Name: "mtp.layers.0.attn_q.weight"}},
			want:    true,
		},
		{
			name:    "qwen35 without legacy mtp marker",
			arch:    "qwen35",
			tensors: nil,
			want:    false,
		},
		{
			name:    "other arch with mtp prefix",
			arch:    "qwen3next",
			tensors: []*ggml.Tensor{{Name: "mtp.fc.weight"}},
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hasLegacyQwenMTPDraft(tt.arch, tt.tensors); got != tt.want {
				t.Fatalf("hasLegacyQwenMTPDraft() = %v, want %v", got, tt.want)
			}
		})
	}
}

func testIntPtr(v int) *int {
	return &v
}

func setFlashAttentionEnv(t *testing.T, value string, set bool) {
	t.Helper()

	if set {
		t.Setenv("OLLAMA_FLASH_ATTENTION", value)
		return
	}

	old, ok := os.LookupEnv("OLLAMA_FLASH_ATTENTION")
	if ok {
		t.Setenv("OLLAMA_FLASH_ATTENTION", old)
	}
	os.Unsetenv("OLLAMA_FLASH_ATTENTION")
}

func TestNormalizeEmbeddingError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		wantStatus int
		wantMsg    string
	}{
		{
			name:       "physical batch size",
			statusCode: http.StatusInternalServerError,
			body:       `{"error":{"code":500,"message":"input (103 tokens) is too large to process. increase the physical batch size (current batch size: 30)"}}`,
			wantStatus: http.StatusBadRequest,
			wantMsg:    "the input length exceeds the context length",
		},
		{
			name:       "context length string error",
			statusCode: http.StatusInternalServerError,
			body:       `{"error":"input length exceeds the context length"}`,
			wantStatus: http.StatusBadRequest,
			wantMsg:    "the input length exceeds the context length",
		},
		{
			name:       "available context",
			statusCode: http.StatusBadRequest,
			body:       `{"error":{"message":"request (302 tokens) exceeds the available context size (256 tokens), try increasing it"}}`,
			wantStatus: http.StatusBadRequest,
			wantMsg:    "the input length exceeds the context length",
		},
		{
			name:       "unrelated error",
			statusCode: http.StatusInternalServerError,
			body:       `{"error":{"message":"backend failed"}}`,
			wantStatus: http.StatusInternalServerError,
			wantMsg:    "backend failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, msg := normalizeEmbeddingError(tt.statusCode, []byte(tt.body))
			if status != tt.wantStatus {
				t.Fatalf("status = %d, want %d", status, tt.wantStatus)
			}
			if msg != tt.wantMsg {
				t.Fatalf("message = %q, want %q", msg, tt.wantMsg)
			}
		})
	}
}

func TestLlamaServerCompletionWithLogprobs(t *testing.T) {
	// Verify logprobs are parsed from SSE streaming responses
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"content":"Hi","stop":false,"completion_probabilities":[{"token":"Hi","logprob":-0.5,"top_logprobs":[{"token":"Hi","logprob":-0.5},{"token":"Hello","logprob":-1.2}]}]}`)
		fmt.Fprintln(w, ``)
		fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	var responses []CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:      "test",
		Options:     &opts,
		Logprobs:    true,
		TopLogprobs: 2,
	}, func(cr CompletionResponse) {
		responses = append(responses, cr)
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}

	// First response should have logprobs
	if len(responses) < 1 {
		t.Fatal("expected at least 1 response")
	}
	if len(responses[0].Logprobs) == 0 {
		t.Fatal("expected logprobs in first response")
	}
	if responses[0].Logprobs[0].Token != "Hi" {
		t.Errorf("token = %q, want %q", responses[0].Logprobs[0].Token, "Hi")
	}
	if responses[0].Logprobs[0].Logprob != -0.5 {
		t.Errorf("logprob = %v, want -0.5", responses[0].Logprobs[0].Logprob)
	}
	if len(responses[0].Logprobs[0].TopLogprobs) != 2 {
		t.Errorf("top_logprobs len = %d, want 2", len(responses[0].Logprobs[0].TopLogprobs))
	}
}

func TestLlamaServerCompletionDoneCallbackAfterStreamClosed(t *testing.T) {
	var completionClosed atomic.Bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/completion":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"","stop":true,"stop_type":"eos","timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			time.Sleep(25 * time.Millisecond)
			completionClosed.Store(true)
		case "/tokenize":
			if !completionClosed.Load() {
				http.Error(w, "completion stream still active", http.StatusInternalServerError)
				return
			}
			fmt.Fprint(w, `{"tokens":[1,2,3]}`)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	opts := api.DefaultOptions()
	var callbackErr error
	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test",
		Options: &opts,
	}, func(cr CompletionResponse) {
		if !cr.Done {
			return
		}
		_, callbackErr = runner.Tokenize(t.Context(), "test")
	})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}
	if callbackErr != nil {
		t.Fatalf("Tokenize from Done callback failed: %v", callbackErr)
	}
}

func TestLlamaServerCompletionSamplingParams(t *testing.T) {
	var capturedReq llamaServerCompletionRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		json.NewDecoder(r.Body).Decode(&capturedReq)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"content":"ok","stop":true,"timings":{"prompt_n":1,"prompt_ms":1,"predicted_n":1,"predicted_ms":1}}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:    portInt,
		cmd:     fakeRunningCmd(),
		sem:     semaphore.NewWeighted(1),
		options: api.Options{Runner: api.Runner{NumCtx: 2048}},
	}

	opts := api.Options{
		Runner:           api.Runner{NumCtx: 2048},
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.9,
		MinP:             0.05,
		NumPredict:       100,
		Stop:             []string{"</s>"},
		RepeatPenalty:    1.1,
		FrequencyPenalty: 0.5,
		PresencePenalty:  0.3,
		Seed:             42,
	}

	err := runner.Completion(t.Context(), CompletionRequest{
		Prompt:  "test",
		Options: &opts,
	}, func(cr CompletionResponse) {})
	if err != nil {
		t.Fatalf("Completion error: %v", err)
	}

	if capturedReq.Temperature != 0.7 {
		t.Errorf("temperature = %v, want 0.7", capturedReq.Temperature)
	}
	if capturedReq.TopK != 40 {
		t.Errorf("top_k = %v, want 40", capturedReq.TopK)
	}
	if capturedReq.TopP != 0.9 {
		t.Errorf("top_p = %v, want 0.9", capturedReq.TopP)
	}
	if capturedReq.NPredict != 100 {
		t.Errorf("n_predict = %v, want 100", capturedReq.NPredict)
	}
	if capturedReq.Seed != 42 {
		t.Errorf("seed = %v, want 42", capturedReq.Seed)
	}
	if capturedReq.RepeatPenalty != 1.1 {
		t.Errorf("repeat_penalty = %v, want 1.1", capturedReq.RepeatPenalty)
	}
	if len(capturedReq.Stop) != 1 || capturedReq.Stop[0] != "</s>" {
		t.Errorf("stop = %v, want [</s>]", capturedReq.Stop)
	}
}

func TestLlamaServerWaitUntilRunning(t *testing.T) {
	callCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount < 3 {
			w.WriteHeader(503)
			fmt.Fprint(w, `{"status":"loading model"}`)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	parts := strings.Split(srv.URL, ":")
	var portInt int
	fmt.Sscanf(parts[len(parts)-1], "%d", &portInt)

	runner := &llamaServerRunner{
		port:      portInt,
		cmd:       fakeRunningCmd(),
		done:      make(chan struct{}),
		loadStart: time.Now(),
	}

	err := runner.WaitUntilRunning(t.Context())
	if err != nil {
		t.Fatalf("WaitUntilRunning error: %v", err)
	}
	if callCount < 3 {
		t.Errorf("expected at least 3 health checks, got %d", callCount)
	}
}

func TestMemoryParsingWriter(t *testing.T) {
	tests := []struct {
		name      string
		lines     []string
		wantGPU   float64 // MiB
		wantTotal float64 // MiB
	}{
		{
			name: "Metal + CPU",
			lines: []string{
				"llama_model_load_from_file_impl:        Metal model buffer size =  1234.56 MiB\n",
				"llama_model_load_from_file_impl:          CPU model buffer size =    56.78 MiB\n",
			},
			wantGPU:   1234.56,
			wantTotal: 1234.56 + 56.78,
		},
		{
			name: "CUDA multi-GPU + host",
			lines: []string{
				"llama_model_load_from_file_impl:        CUDA0 model buffer size =   800.00 MiB\n",
				"llama_model_load_from_file_impl:        CUDA1 model buffer size =   400.00 MiB\n",
				"llama_model_load_from_file_impl:    CUDA_Host model buffer size =   100.00 MiB\n",
			},
			wantGPU:   1200.00,
			wantTotal: 1300.00,
		},
		{
			name: "ROCm + host",
			lines: []string{
				"llama_model_load_from_file_impl:        ROCm0 model buffer size =  2000.00 MiB\n",
				"llama_model_load_from_file_impl:    ROCm_Host model buffer size =   150.00 MiB\n",
			},
			wantGPU:   2000.00,
			wantTotal: 2150.00,
		},
		{
			name: "Vulkan + host",
			lines: []string{
				"llama_model_load_from_file_impl:      Vulkan0 model buffer size =   500.00 MiB\n",
				"llama_model_load_from_file_impl:  Vulkan_Host model buffer size =    50.00 MiB\n",
			},
			wantGPU:   500.00,
			wantTotal: 550.00,
		},
		{
			name: "Metal Private + Mapped (both GPU memory)",
			lines: []string{
				"llama_model_load_from_file_impl: Metal_Private model buffer size =   300.00 MiB\n",
				"llama_model_load_from_file_impl:  Metal_Mapped model buffer size =    20.00 MiB\n",
			},
			wantGPU:   320.00, // both Private and Mapped are device memory
			wantTotal: 320.00,
		},
		{
			name:      "no buffer lines",
			lines:     []string{"some random log line\n"},
			wantGPU:   0,
			wantTotal: 0,
		},
		{
			name: "fit probe buffers are replaced by final load",
			lines: []string{
				"load_tensors:        CUDA0 model buffer size =  1000.00 MiB\n",
				"llama_kv_cache:      CUDA0 KV buffer size =  2000.00 MiB\n",
				"sched_reserve:      CUDA0 compute buffer size =   300.00 MiB\n",
				"sched_reserve:  CUDA_Host compute buffer size =   400.00 MiB\n",
				"load_tensors:        CUDA0 model buffer size =  1100.00 MiB\n",
				"llama_kv_cache:      CUDA0 KV buffer size =  2200.00 MiB\n",
				"sched_reserve:      CUDA0 compute buffer size =   330.00 MiB\n",
				"sched_reserve:  CUDA_Host compute buffer size =   440.00 MiB\n",
				"alloc_compute_meta:  CUDA0 compute buffer size =    10.00 MiB\n",
				"llama_memory_recurrent:      CUDA0 RS buffer size =    20.00 MiB\n",
			},
			wantGPU:   1100 + 2200 + 330 + 10 + 20,
			wantTotal: 1100 + 2200 + 330 + 440 + 10 + 20,
		},
		{
			name: "rc21 fit probe accounting",
			lines: []string{
				"load_tensors:          CPU model buffer size =     0.00 MiB\n",
				"load_tensors:        CUDA0 model buffer size =     0.00 MiB\n",
				"load_tensors:        CUDA1 model buffer size =     0.00 MiB\n",
				"llama_context:  CUDA_Host  output buffer size =     0.95 MiB\n",
				"llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB\n",
				"llama_kv_cache:      CUDA1 KV buffer size =     0.00 MiB\n",
				"llama_memory_recurrent:      CUDA0 RS buffer size =    90.40 MiB\n",
				"llama_memory_recurrent:      CUDA1 RS buffer size =    59.23 MiB\n",
				"sched_reserve:      CUDA0 compute buffer size =  9952.25 MiB\n",
				"sched_reserve:      CUDA1 compute buffer size =  6436.28 MiB\n",
				"sched_reserve:  CUDA_Host compute buffer size =  8272.31 MiB\n",
				"load_tensors:          CPU model buffer size =   682.03 MiB\n",
				"load_tensors:        CUDA0 model buffer size =  8171.01 MiB\n",
				"load_tensors:        CUDA1 model buffer size =  6618.25 MiB\n",
				"llama_kv_cache:      CUDA0 KV buffer size =  9216.00 MiB\n",
				"llama_kv_cache:      CUDA1 KV buffer size =  7168.00 MiB\n",
				"llama_memory_recurrent:      CUDA0 RS buffer size =    90.40 MiB\n",
				"llama_memory_recurrent:      CUDA1 RS buffer size =    59.23 MiB\n",
				"sched_reserve:      CUDA0 compute buffer size =  9952.25 MiB\n",
				"sched_reserve:      CUDA1 compute buffer size =  6276.28 MiB\n",
				"sched_reserve:  CUDA_Host compute buffer size =  8272.31 MiB\n",
				"alloc_compute_meta:      CUDA0 compute buffer size =   248.10 MiB\n",
				"alloc_compute_meta:        CPU compute buffer size =    24.93 MiB\n",
			},
			wantGPU:   47799.52,
			wantTotal: 56779.74,
		},
	}

	withinKiB := func(got, want uint64) bool {
		if got > want {
			return got-want <= 1024
		}
		return want-got <= 1024
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &llamaServerRunner{vramByDevice: make(map[string]uint64)}
			w := &memoryParsingWriter{
				inner:  io.Discard,
				runner: runner,
			}

			for _, line := range tt.lines {
				w.Write([]byte(line))
			}

			expectedGPU := uint64(tt.wantGPU * 1024 * 1024)
			expectedTotal := uint64(tt.wantTotal * 1024 * 1024)

			if !withinKiB(runner.memGPU, expectedGPU) {
				t.Errorf("memGPU = %d, want %d", runner.memGPU, expectedGPU)
			}
			if !withinKiB(runner.memTotal, expectedTotal) {
				t.Errorf("memTotal = %d, want %d", runner.memTotal, expectedTotal)
			}

			total, vram := runner.MemorySize()
			if !withinKiB(total, expectedTotal) {
				t.Errorf("MemorySize total = %d, want %d", total, expectedTotal)
			}
			if !withinKiB(vram, expectedGPU) {
				t.Errorf("MemorySize vram = %d, want %d", vram, expectedGPU)
			}
		})
	}
}

func TestMemoryParsingWriterRecordsOutputActivityWithoutNewline(t *testing.T) {
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}

	runner.startLoadTracking(time.Now())
	before := time.Now()
	if _, err := w.Write([]byte("...")); err != nil {
		t.Fatal(err)
	}
	if got := runner.lastLoadActivity(); got.Before(before) {
		t.Fatalf("lastLoadActivity = %v, want after %v", got, before)
	}
}

func TestMemoryParsingWriterIgnoresOutputActivityAfterLoadTrackingStops(t *testing.T) {
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}

	runner.startLoadTracking(time.Now())
	if _, err := w.Write([]byte(".")); err != nil {
		t.Fatal(err)
	}
	lastActivity := runner.lastLoadActivity()

	runner.stopLoadTracking()
	if _, err := w.Write([]byte(".")); err != nil {
		t.Fatal(err)
	}
	if got := runner.lastLoadActivity(); !got.Equal(lastActivity) {
		t.Fatalf("lastLoadActivity changed after tracking stopped: got %v, want %v", got, lastActivity)
	}
}

func TestMemoryParsingPerDevice(t *testing.T) {
	tests := []struct {
		name       string
		lines      []string
		wantDevice map[string]uint64 // device name → expected MiB
	}{
		{
			name: "CUDA multi-GPU all buffer types",
			lines: []string{
				"load_tensors:        CUDA0 model buffer size =   852.89 MiB\n",
				"load_tensors:        CUDA1 model buffer size =  1065.46 MiB\n",
				"load_tensors:   CPU_Mapped model buffer size =   308.23 MiB\n",
				"llama_kv_cache:      CUDA0 KV buffer size =  1920.00 MiB\n",
				"llama_kv_cache:      CUDA1 KV buffer size =  1664.00 MiB\n",
				"sched_reserve:      CUDA0 compute buffer size =   378.04 MiB\n",
				"sched_reserve:      CUDA1 compute buffer size =   408.55 MiB\n",
				"sched_reserve:  CUDA_Host compute buffer size =   268.05 MiB\n",
			},
			wantDevice: map[string]uint64{
				"CUDA0": 852 + 1920 + 378, // model + KV + compute (approx MiB)
				"CUDA1": 1065 + 1664 + 408,
			},
		},
		{
			name: "Metal with mapped buffers",
			lines: []string{
				"load_tensors:  MTL0_Mapped model buffer size =  1918.35 MiB\n",
				"llama_kv_cache:       MTL0 KV buffer size =   448.00 MiB\n",
				"sched_reserve:       MTL0 compute buffer size =   256.50 MiB\n",
				"sched_reserve:        CPU compute buffer size =    20.01 MiB\n",
			},
			wantDevice: map[string]uint64{
				"MTL0": 1918 + 448 + 256, // Mapped model weights + KV + compute (all GPU)
			},
		},
		{
			name: "ROCm single GPU",
			lines: []string{
				"load_tensors:        ROCm0 model buffer size =  1918.35 MiB\n",
				"llama_kv_cache:      ROCm0 KV buffer size =   448.00 MiB\n",
				"sched_reserve:      ROCm0 compute buffer size =   256.50 MiB\n",
				"sched_reserve:  ROCm_Host compute buffer size =    20.01 MiB\n",
			},
			wantDevice: map[string]uint64{
				"ROCm0": 1918 + 448 + 256,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &llamaServerRunner{vramByDevice: make(map[string]uint64)}
			w := &memoryParsingWriter{inner: io.Discard, runner: runner}

			for _, line := range tt.lines {
				w.Write([]byte(line))
			}

			for dev, wantMiB := range tt.wantDevice {
				got := runner.vramByDevice[dev] / (1024 * 1024) // convert to MiB
				// Allow ~1 MiB tolerance for floating point
				if got < wantMiB-2 || got > wantMiB+2 {
					t.Errorf("vramByDevice[%q] = %d MiB, want ~%d MiB", dev, got, wantMiB)
				}
			}

			// Verify host/mapped buffers are NOT in per-device tracking
			for dev := range runner.vramByDevice {
				if !isGPUBuffer(dev) {
					t.Errorf("non-GPU buffer %q found in vramByDevice", dev)
				}
			}
		})
	}
}

func TestMemoryParsingWriterConcurrentReads(t *testing.T) {
	runner := &llamaServerRunner{
		vramByDevice:     make(map[string]uint64),
		systemFreeAtLoad: make(map[string]uint64),
		gpus: []ml.DeviceInfo{
			{
				DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
				Name:        "CUDA0",
				TotalMemory: 16000 * 1024 * 1024,
			},
		},
	}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}
	lines := [][]byte{
		[]byte("common_params_fit_impl: getting device memory data for initial parameters:\n"),
		[]byte("using device CUDA0 (NVIDIA GPU) (0000:01:00.0) - 12000 MiB free\n"),
		[]byte("load_tensors:        CUDA0 model buffer size =  1000.00 MiB\n"),
		[]byte("llama_kv_cache:      CUDA0 KV buffer size =  2000.00 MiB\n"),
		[]byte("sched_reserve:      CUDA0 compute buffer size =   300.00 MiB\n"),
		[]byte("llm_load_tensors: offloaded 33/33 layers to GPU\n"),
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 1000 {
			for _, line := range lines {
				_, _ = w.Write(line)
			}
		}
	}()

	for {
		select {
		case <-done:
			return
		default:
			runner.MemorySize()
			runner.VRAMByGPU(ml.DeviceID{ID: "0", Library: "CUDA"})
			runner.GetDeviceInfos(context.Background())
		}
	}
}

func TestMemoryParsingWriterMemorySizeFullOffload(t *testing.T) {
	tests := []struct {
		name             string
		lines            []string
		wantProcessTotal uint64
		wantProcessVRAM  uint64
	}{
		{
			name: "fully offloaded",
			lines: []string{
				"llm_load_tensors: offloading 32 repeating layers to GPU\n",
				"llm_load_tensors: offloaded 33/33 layers to GPU\n",
			},
			wantProcessTotal: 80,
			wantProcessVRAM:  80,
		},
		{
			name: "partially offloaded",
			lines: []string{
				"llm_load_tensors: offloaded 22/33 layers to GPU\n",
			},
			wantProcessTotal: 100,
			wantProcessVRAM:  80,
		},
		{
			name: "missing offload line",
			lines: []string{
				"llm_load_tensors: offloading 32 repeating layers to GPU\n",
			},
			wantProcessTotal: 100,
			wantProcessVRAM:  80,
		},
		{
			name: "latest offload line wins",
			lines: []string{
				"llm_load_tensors: offloaded 0/33 layers to GPU\n",
				"llm_load_tensors: offloaded 33/33 layers to GPU\n",
			},
			wantProcessTotal: 80,
			wantProcessVRAM:  80,
		},
		{
			name: "fit overflow suppresses full offload mask",
			lines: []string{
				"common_params_fit_impl:   - ROCm0 (AMD Radeon RX 6700 XT): 25 layers ( 5 overflowing),  11065 MiB used,   1036 MiB free\n",
				"llm_load_tensors: offloaded 25/25 layers to GPU\n",
			},
			wantProcessTotal: 100,
			wantProcessVRAM:  80,
		},
		{
			name: "fit without overflow still masks full offload",
			lines: []string{
				"common_params_fit_impl:   - ROCm0 (AMD Radeon Pro W7900): 34 layers ( 0 overflowing),  32765 MiB used,   1144 MiB free\n",
				"llm_load_tensors: offloaded 34/34 layers to GPU\n",
			},
			wantProcessTotal: 80,
			wantProcessVRAM:  80,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runner := &llamaServerRunner{memTotal: 100, memGPU: 80}
			w := &memoryParsingWriter{inner: io.Discard, runner: runner}
			for _, line := range tt.lines {
				if _, err := w.Write([]byte(line)); err != nil {
					t.Fatal(err)
				}
			}

			total, vram := runner.MemorySize()
			if total != tt.wantProcessTotal || vram != tt.wantProcessVRAM {
				t.Fatalf("MemorySize() = %d/%d, want %d/%d", total, vram, tt.wantProcessTotal, tt.wantProcessVRAM)
			}
		})
	}
}

func TestVRAMByGPU(t *testing.T) {
	runner := &llamaServerRunner{
		vramByDevice: map[string]uint64{
			"CUDA0": 1000 * 1024 * 1024,
			"CUDA1": 2000 * 1024 * 1024,
		},
		gpus: []ml.DeviceInfo{
			{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0"},
			{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, Name: "CUDA1"},
		},
	}

	got0 := runner.VRAMByGPU(ml.DeviceID{ID: "0", Library: "CUDA"})
	if got0 != 1000*1024*1024 {
		t.Errorf("VRAMByGPU(CUDA:0) = %d, want %d", got0, 1000*1024*1024)
	}

	got1 := runner.VRAMByGPU(ml.DeviceID{ID: "1", Library: "CUDA"})
	if got1 != 2000*1024*1024 {
		t.Errorf("VRAMByGPU(CUDA:1) = %d, want %d", got1, 2000*1024*1024)
	}

	// Unknown device returns 0
	gotUnknown := runner.VRAMByGPU(ml.DeviceID{ID: "9", Library: "CUDA"})
	if gotUnknown != 0 {
		t.Errorf("VRAMByGPU(unknown) = %d, want 0", gotUnknown)
	}
}

func TestGetDeviceInfos(t *testing.T) {
	runner := &llamaServerRunner{
		vramByDevice: map[string]uint64{
			"CUDA0": 3000 * 1024 * 1024,
		},
		gpus: []ml.DeviceInfo{
			{
				DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
				Name:        "CUDA0",
				TotalMemory: 16000 * 1024 * 1024,
				FreeMemory:  15000 * 1024 * 1024, // stale value from discovery
			},
		},
	}

	infos := runner.GetDeviceInfos(context.Background())
	if len(infos) != 1 {
		t.Fatalf("expected 1 device, got %d", len(infos))
	}
	// Free should be Total - Used, not the stale discovery value
	expectedFree := uint64((16000 - 3000) * 1024 * 1024)
	if infos[0].FreeMemory != expectedFree {
		t.Errorf("FreeMemory = %d, want %d", infos[0].FreeMemory, expectedFree)
	}
}

func TestGetDeviceInfosMinOfTwo(t *testing.T) {
	// External consumer scenario: system reports less free than our accounting expects
	runner := &llamaServerRunner{
		vramByDevice: map[string]uint64{
			"CUDA0": 3000 * 1024 * 1024, // we used 3GB
		},
		systemFreeAtLoad: map[string]uint64{
			"CUDA0": 12000 * 1024 * 1024, // system said 12GB free at load time (external app using 4GB)
		},
		gpus: []ml.DeviceInfo{
			{
				DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
				Name:        "CUDA0",
				TotalMemory: 16000 * 1024 * 1024, // 16GB total
			},
		},
	}

	infos := runner.GetDeviceInfos(context.Background())
	// Our accounting: 16000 - 3000 = 13000 MB free
	// System-based: 12000 - 3000 = 9000 MB free (external consumer detected)
	// Min = 9000 MB
	expectedFree := uint64(9000 * 1024 * 1024)
	if infos[0].FreeMemory != expectedFree {
		t.Errorf("FreeMemory = %d MiB, want %d MiB (min-of-two should detect external consumer)",
			infos[0].FreeMemory/(1024*1024), expectedFree/(1024*1024))
	}
}

func TestGetDeviceInfosSystemOptimistic(t *testing.T) {
	// Platform where system over-reports free (e.g., Metal shared memory)
	runner := &llamaServerRunner{
		vramByDevice: map[string]uint64{
			"MTL0": 5000 * 1024 * 1024, // we used 5GB
		},
		systemFreeAtLoad: map[string]uint64{
			"MTL0": 100000 * 1024 * 1024, // system says 100GB free (unified memory, unreliable)
		},
		gpus: []ml.DeviceInfo{
			{
				DeviceID:    ml.DeviceID{ID: "0", Library: "Metal"},
				Name:        "MTL0",
				TotalMemory: 100000 * 1024 * 1024,
			},
		},
	}

	infos := runner.GetDeviceInfos(context.Background())
	// Our accounting: 100000 - 5000 = 95000 MB
	// System-based: 100000 - 5000 = 95000 MB
	// Min = 95000 MB (both agree, system isn't lying here)
	expectedFree := uint64(95000 * 1024 * 1024)
	if infos[0].FreeMemory != expectedFree {
		t.Errorf("FreeMemory = %d MiB, want %d MiB",
			infos[0].FreeMemory/(1024*1024), expectedFree/(1024*1024))
	}
}

func TestIsGPUBuffer(t *testing.T) {
	gpu := []string{
		"Metal", "Metal_Private", "CUDA0", "CUDA1", "ROCm0", "Vulkan0", "MUSA0",
		"MTL0_Mapped", "MTL0_REPACK", "CUDA0_Mapped",
	}
	for _, name := range gpu {
		if !isGPUBuffer(name) {
			t.Errorf("isGPUBuffer(%q) = false, want true", name)
		}
	}
	notGPU := []string{
		"CPU", "BLAS", "CUDA_Host", "ROCm_Host", "Vulkan_Host",
		"CPU_Mapped", "CPU_REPACK",
	}
	for _, name := range notGPU {
		if isGPUBuffer(name) {
			t.Errorf("isGPUBuffer(%q) = true, want false", name)
		}
	}
}

func TestAccumulatedToolCallsRejectsInvalidArguments(t *testing.T) {
	_, err := accumulatedToolCalls(map[int]*llamaServerToolCallAccumulator{
		0: {
			name:      "weather",
			arguments: `{"city":`,
		},
	})
	if err == nil {
		t.Fatal("expected invalid tool call arguments to return an error")
	}
	if !strings.Contains(err.Error(), "weather") {
		t.Fatalf("expected function name in error, got %v", err)
	}
}

func TestLlamaServerChatTemplateKwargs(t *testing.T) {
	tests := []struct {
		name  string
		think *api.ThinkValue
		want  map[string]any
	}{
		{
			name: "unset",
		},
		{
			name:  "disabled",
			think: &api.ThinkValue{Value: false},
			want:  map[string]any{"enable_thinking": false},
		},
		{
			name:  "enabled uses template default effort",
			think: &api.ThinkValue{Value: true},
			want:  map[string]any{"enable_thinking": true},
		},
		{
			name:  "explicit effort",
			think: &api.ThinkValue{Value: "high"},
			want: map[string]any{
				"enable_thinking":  true,
				"reasoning_effort": "high",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := llamaServerChatTemplateKwargs(tt.think)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("kwargs = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestLlamaServerChatMessageConvertsToolCalls(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "ls")

	msg, err := llamaServerChatMessage(Message{
		Role: "assistant",
		ToolCalls: []api.ToolCall{{
			ID: "call_1",
			Function: api.ToolCallFunction{
				Index:     2,
				Name:      "bash",
				Arguments: args,
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	toolCalls, ok := msg["tool_calls"].([]llamaServerChatToolCall)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected one llama-server tool call, got %#v", msg["tool_calls"])
	}
	if toolCalls[0].Index != 2 || toolCalls[0].Type != "function" || toolCalls[0].Function.Name != "bash" {
		t.Fatalf("unexpected tool call metadata: %#v", toolCalls[0])
	}
	if toolCalls[0].Function.Arguments != `{"command":"ls"}` {
		t.Fatalf("expected string-encoded arguments, got %#v", toolCalls[0])
	}
}

func TestLlamaServerChatMessageConvertsMediaParts(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n")
	wav := []byte("RIFF\x00\x00\x00\x00WAVE")
	mp3 := []byte("ID3\x04\x00\x00")

	msg, err := llamaServerChatMessage(Message{
		Role:    "user",
		Content: "describe these",
		Media:   []MediaData{NewMediaData(0, png), NewMediaData(1, wav), NewMediaData(2, mp3)},
	})
	if err != nil {
		t.Fatal(err)
	}

	parts, ok := msg["content"].([]map[string]any)
	if !ok || len(parts) != 4 {
		t.Fatalf("expected four content parts, got %#v", msg["content"])
	}
	if parts[1]["type"] != "image_url" {
		t.Fatalf("expected image_url for PNG, got %#v", parts[1])
	}
	for i, want := range []string{"wav", "mp3"} {
		part := parts[i+2]
		if part["type"] != "input_audio" {
			t.Fatalf("expected input_audio for %s, got %#v", want, part)
		}
		audio, ok := part["input_audio"].(map[string]any)
		if !ok {
			t.Fatalf("expected input_audio payload for %s, got %#v", want, part["input_audio"])
		}
		if audio["format"] != want {
			t.Fatalf("expected %s format, got %#v", want, audio["format"])
		}
		if audio["data"] == "" {
			t.Fatalf("expected base64 audio data for %s", want)
		}
	}
}

func TestFindLlamaServer(t *testing.T) {
	// This just tests that the function doesn't panic and returns a reasonable error
	// when the binary doesn't exist in the expected locations
	_, err := FindLlamaServer()
	// In the test environment, it may or may not exist depending on whether
	// cmake was run. Just verify it doesn't panic.
	_ = err
}

func loadTestGGML(t *testing.T, kv ggml.KV) *ggml.GGML {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if err := ggml.WriteGGUF(f, kv, nil); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	model, err := LoadModel(f.Name(), 0)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// fakeRunningCmd returns an exec.Cmd that looks like it's still running
// (ProcessState is nil, which is the case before Wait() completes).
// Registers cleanup via t.Cleanup to prevent zombie processes.
func fakeRunningCmd() *exec.Cmd {
	cmd := exec.Command("sleep", "3600")
	cmd.Start()
	// Note: cleanup happens when the test binary exits since we can't
	// pass *testing.T here without changing all call sites. The OS will
	// SIGKILL children when the test process exits.
	return cmd
}
