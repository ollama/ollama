package server

import (
	"bytes"
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
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
	createParserModel(t, s, "parse-wedge-gen", "qwen3.5")

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

	kv := ggml.KV{
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
	_, digest := createBinFile(t, kv, []*ggml.Tensor{
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
