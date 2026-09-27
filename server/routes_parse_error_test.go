package server

import (
	"bytes"
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

// A tool call whose JSON is cut off mid-value, which the qwen3-vl parser
// rejects on the chunk that carries it.
//
// These tests used a qwen3.5 call that closes <parameter> with </function>.
// qwen3.5 hands its tool calls to the qwen3-coder parser, which now returns a
// block it cannot parse as content instead of failing the request, so that
// input no longer raises an error at all. The subject here is what routes.go
// does when a parser errors mid-stream, not which inputs a parser rejects, so
// the tests use a parser that still has that error to raise.
const malformedToolCall = "<think>\nthinking\n</think>\n\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": }\n</tool_call>"

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
	createParserModel(t, s, "parse-wedge", "qwen3-vl-thinking")

	stream := false
	done := make(chan struct{})
	go func() {
		defer close(done)
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "parse-wedge",
			Messages: []api.Message{{Role: "user", Content: "hello"}},
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

func TestGenerateParseErrorMidStreamDoesNotWedge(t *testing.T) {
	gin.SetMode(gin.TestMode)

	secondChunkReturned := make(chan struct{})

	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{Content: malformedToolCall})
			fn(llm.CompletionResponse{Content: "trailing"})
			close(secondChunkReturned)
			fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
			return nil
		},
	}

	s := newServerWithMockRunner(t, &mock)
	createParserModel(t, s, "parse-wedge-gen", "qwen3-vl-thinking")

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
