package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
)

// malformedToolCall closes <parameter> with </function>, which the qwen3.5
// parser rejects.
const malformedToolCall = "<think>\nthinking\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nprobe.txt\n</function>\n</tool_call>"

func TestChatParseErrorMidStreamDoesNotWedge(t *testing.T) {
	gin.SetMode(gin.TestMode)

	secondChunkReturned := make(chan struct{})

	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: malformedToolCall})
			// The chunk after the failed parse is the one that wedges.
			fn(llm.CompletionResponse{Content: "trailing"})
			close(secondChunkReturned)
			fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}

	s := newServerWithMockRunner(t, &mock)
	createParserModel(t, s, "parse-wedge", "qwen3.5")

	stream := false
	done := make(chan struct{})
	go func() {
		defer close(done)
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "parse-wedge",
			Messages: []api.Message{{Role: "user", Content: "hello"}},
			Tools:    []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "write_file"}}},
			Stream:   &stream,
		})
		if w.Code != http.StatusInternalServerError {
			t.Errorf("expected 500 from parse failure, got %d: %s", w.Code, w.Body.String())
		}
	}()

	select {
	case <-secondChunkReturned:
	case <-time.After(5 * time.Second):
		t.Fatal("completion callback blocked after a parse error")
	}

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("chat handler did not return after a parse error")
	}
}

type failingStreamParser struct {
	parsers.PassthroughParser
}

func (p *failingStreamParser) Add(s string, done bool) (string, string, []api.ToolCall, error) {
	if s == "invalid" {
		return "", "", nil, errors.New("invalid stream chunk")
	}
	return p.PassthroughParser.Add(s, done)
}

func TestGenerateParseErrorMidStreamDoesNotWedge(t *testing.T) {
	gin.SetMode(gin.TestMode)
	parsers.Register("test-failing-stream", func() parsers.Parser { return &failingStreamParser{} })

	secondChunkReturned := make(chan struct{})

	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: "invalid"})
			fn(llm.CompletionResponse{Content: "trailing"})
			close(secondChunkReturned)
			fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}

	s := newServerWithMockRunner(t, &mock)
	createParserModel(t, s, "parse-wedge-gen", "test-failing-stream")

	stream := false
	done := make(chan struct{})
	go func() {
		defer close(done)
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "parse-wedge-gen",
			Prompt: "hello",
			Stream: &stream,
		})
		if w.Code != http.StatusInternalServerError {
			t.Errorf("expected 500 from parse failure, got %d: %s", w.Code, w.Body.String())
		}
	}()

	select {
	case <-secondChunkReturned:
	case <-time.After(5 * time.Second):
		t.Fatal("completion callback blocked after a parse error")
	}

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("generate handler did not return after a parse error")
	}
}

func TestChatPreservesToolMarkupWithoutTools(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const prefix = "<tool"
	const rest = "_call><function=lookup></function></tool_call>"

	for _, tc := range []struct {
		name   string
		stream bool
	}{
		{name: "non-streaming"},
		{name: "streaming", stream: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mock := mockRunner{
				CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
					fn(llm.CompletionResponse{Content: prefix})
					fn(llm.CompletionResponse{Content: rest})
					fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
					return nil
				},
			}
			s := newServerWithMockRunner(t, &mock)
			createParserModel(t, s, "literal-tool-markup", "qwen3-coder")
			w := createRequest(t, s.ChatHandler, api.ChatRequest{
				Model:    "literal-tool-markup",
				Messages: []api.Message{{Role: "user", Content: "Show the tool syntax."}},
				Stream:   &tc.stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body)
			}

			var content string
			decoder := json.NewDecoder(w.Body)
			for {
				var response api.ChatResponse
				if err := decoder.Decode(&response); err == io.EOF {
					break
				} else if err != nil {
					t.Fatal(err)
				}
				if len(response.Message.ToolCalls) != 0 {
					t.Fatalf("unexpected tool calls: %v", response.Message.ToolCalls)
				}
				content += response.Message.Content
			}
			if content != prefix+rest {
				t.Errorf("content = %q, want %q", content, prefix+rest)
			}
		})
	}
}

func createParserModel(t *testing.T, s *Server, name, parser string) {
	t.Helper()

	kv := gguftest.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}
	_, digest := createBinFile(t, kv, []*gguftest.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})

	stream := false
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  name,
		Files:  map[string]string{"file.gguf": digest},
		Parser: parser,
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("creating model: %d: %s", w.Code, w.Body.String())
	}
}
