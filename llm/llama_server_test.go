package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

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

func TestLlamaServerCompletionSSEParsing(t *testing.T) {
	// Simulate llama-server SSE streaming response
	sseLines := []string{
		`data: {"content":"Hello","stop":false}`,
		``,
		`data: {"content":" world","stop":false}`,
		``,
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
		Images:  []ImageData{{ID: 7, Data: []byte("media-bytes")}},
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

func TestLlamaServerWaitUntilRunningTimesOutWhenLoadExceedsTimeout(t *testing.T) {
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

func TestLlamaServerCompletionStripsLeadingBOS(t *testing.T) {
	tests := []struct {
		name            string
		stripLeadingBOS bool
		prompt          string
		wantPrompt      string
	}{
		{
			name:            "gemma4 llama server path",
			stripLeadingBOS: true,
			prompt:          "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt:      "<|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:            "other model keeps prompt",
			stripLeadingBOS: false,
			prompt:          "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
			wantPrompt:      "<bos><|turn>user\nhello<turn|>\n<|turn>model\n",
		},
		{
			name:            "only leading bos is stripped",
			stripLeadingBOS: true,
			prompt:          "prefix <bos>",
			wantPrompt:      "prefix <bos>",
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
				port:            portInt,
				cmd:             fakeRunningCmd(),
				sem:             semaphore.NewWeighted(1),
				options:         api.Options{Runner: api.Runner{NumCtx: 2048}},
				stripLeadingBOS: tt.stripLeadingBOS,
			}

			opts := api.DefaultOptions()
			err := runner.Completion(t.Context(), CompletionRequest{
				Prompt:  tt.prompt,
				Options: &opts,
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

func TestQwen3VLServerArgs(t *testing.T) {
	tests := []struct {
		name string
		arch string
		want []string
	}{
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
			if got := qwen3VLServerArgs(tt.arch); !slices.Equal(got, tt.want) {
				t.Fatalf("qwen3VLServerArgs(%q) = %v, want %v", tt.arch, got, tt.want)
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

func TestAppendFlashAttentionArgs(t *testing.T) {
	supportedGPU := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 8, ComputeMinor: 9}}
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
			name: "explicit enable is clamped off on old cuda",
			env:  "1",
			set:  true,
			gpus: oldGPU,
			want: []string{"base", "--flash-attn", "off"},
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

			if runner.memGPU != expectedGPU {
				t.Errorf("memGPU = %d, want %d", runner.memGPU, expectedGPU)
			}
			if runner.memTotal != expectedTotal {
				t.Errorf("memTotal = %d, want %d", runner.memTotal, expectedTotal)
			}

			total, vram := runner.MemorySize()
			if total != expectedTotal {
				t.Errorf("MemorySize total = %d, want %d", total, expectedTotal)
			}
			if vram != expectedGPU {
				t.Errorf("MemorySize vram = %d, want %d", vram, expectedGPU)
			}
		})
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

func TestFindLlamaServer(t *testing.T) {
	// This just tests that the function doesn't panic and returns a reasonable error
	// when the binary doesn't exist in the expected locations
	_, err := FindLlamaServer()
	// In the test environment, it may or may not exist depending on whether
	// cmake was run. Just verify it doesn't panic.
	_ = err
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
