package llm

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// mockLlamaServer implements LlamaServer interface for testing
type mockLlamaServer struct {
	batchSupported bool
	maxBatchSize   int
	estimatedVRAM  uint64
	estimatedTotal uint64
	numParallel    int
}

func (m *mockLlamaServer) Ping(ctx context.Context) error                          { return nil }
func (m *mockLlamaServer) WaitUntilRunning(ctx context.Context) error             { return nil }
func (m *mockLlamaServer) Embedding(ctx context.Context, input string) ([]float32, error) {
	return make([]float32, 384), nil
}
func (m *mockLlamaServer) Tokenize(ctx context.Context, content string) ([]int, error) {
	return []int{1, 2, 3}, nil
}
func (m *mockLlamaServer) Detokenize(ctx context.Context, tokens []int) (string, error) {
	return "test", nil
}
func (m *mockLlamaServer) Close() error                      { return nil }
func (m *mockLlamaServer) EstimatedVRAM() uint64             { return m.estimatedVRAM }
func (m *mockLlamaServer) EstimatedTotal() uint64            { return m.estimatedTotal }
func (m *mockLlamaServer) EstimatedVRAMByGPU(gpuID string) uint64 { return m.estimatedVRAM }
func (m *mockLlamaServer) Pid() int                          { return 12345 }

func (m *mockLlamaServer) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	// Simulate completion response - first partial, then final
	fn(CompletionResponse{
		Content:            "Test response for: " + req.Prompt,
		Done:               false,
		PromptEvalCount:    10,
		PromptEvalDuration: 50 * time.Millisecond,
		EvalCount:          20,
		EvalDuration:       100 * time.Millisecond,
	})
	
	fn(CompletionResponse{
		Content:            "",
		Done:               true,
		DoneReason:         DoneReasonStop,
		PromptEvalCount:    10,
		PromptEvalDuration: 50 * time.Millisecond,
		EvalCount:          20,
		EvalDuration:       100 * time.Millisecond,
	})
	
	return nil
}

func (m *mockLlamaServer) BatchCompletion(ctx context.Context, reqs []CompletionRequest, fn func([]CompletionResponse)) error {
	if !m.batchSupported {
		return m.fallbackToIndividual(ctx, reqs, fn)
	}
	
	// Check batch size validation
	if len(reqs) > m.maxBatchSize {
		return fmt.Errorf("batch size %d exceeds maximum %d", len(reqs), m.maxBatchSize)
	}
	
	responses := make([]CompletionResponse, len(reqs))
	for i, req := range reqs {
		responses[i] = CompletionResponse{
			Content:            "Batch response for: " + req.Prompt,
			Done:               true,
			DoneReason:         DoneReasonStop,
			PromptEvalCount:    10,
			PromptEvalDuration: 50 * time.Millisecond,
			EvalCount:          20,
			EvalDuration:       100 * time.Millisecond,
		}
	}
	
	fn(responses)
	return nil
}

func (m *mockLlamaServer) fallbackToIndividual(ctx context.Context, reqs []CompletionRequest, fn func([]CompletionResponse)) error {
	responses := make([]CompletionResponse, len(reqs))
	for i, req := range reqs {
		// Capture content from the completion response
		var content string
		err := m.Completion(ctx, req, func(resp CompletionResponse) {
			if !resp.Done && resp.Content != "" {
				content = resp.Content
			}
			if resp.Done {
				responses[i] = CompletionResponse{
					Content:            content,
					Done:               true,
					DoneReason:         resp.DoneReason,
					PromptEvalCount:    resp.PromptEvalCount,
					PromptEvalDuration: resp.PromptEvalDuration,
					EvalCount:          resp.EvalCount,
					EvalDuration:       resp.EvalDuration,
				}
			}
		})
		if err != nil {
			return err
		}
	}
	fn(responses)
	return nil
}

func (m *mockLlamaServer) BatchEmbedding(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) > m.maxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds maximum %d", len(inputs), m.maxBatchSize)
	}
	
	embeddings := make([][]float32, len(inputs))
	for i := range inputs {
		embeddings[i] = make([]float32, 384)
		// Fill with some test values
		for j := range embeddings[i] {
			embeddings[i][j] = float32(i*j) * 0.1
		}
	}
	return embeddings, nil
}

func (m *mockLlamaServer) GetMaxBatchSize() int     { return m.maxBatchSize }
func (m *mockLlamaServer) SupportsBatching() bool   { return m.batchSupported }

func (m *mockLlamaServer) EstimateBatchMemory(batchSize int) uint64 {
	if batchSize <= 1 {
		return 0
	}
	baseMemory := m.EstimatedTotal()
	return uint64(float64(baseMemory) * 0.5 * float64(batchSize) / float64(m.maxBatchSize))
}

func TestBatchCompletionSupported(t *testing.T) {
	// Enable batching for this test
	os.Setenv("OLLAMA_BATCH_ENABLED", "true")
	defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   8,
		estimatedVRAM:  1024 * 1024 * 1024, // 1GB
		estimatedTotal: 2048 * 1024 * 1024, // 2GB
		numParallel:    4,
	}

	ctx := context.Background()
	reqs := []CompletionRequest{
		{Prompt: "Hello"},
		{Prompt: "World"},
		{Prompt: "Test"},
	}

	var responses []CompletionResponse
	err := server.BatchCompletion(ctx, reqs, func(resps []CompletionResponse) {
		responses = resps
	})

	if err != nil {
		t.Fatalf("BatchCompletion failed: %v", err)
	}

	if len(responses) != len(reqs) {
		t.Errorf("Expected %d responses, got %d", len(reqs), len(responses))
	}

	for i, resp := range responses {
		if !resp.Done {
			t.Errorf("Response %d not marked as done", i)
		}
		if resp.DoneReason != DoneReasonStop {
			t.Errorf("Response %d has wrong DoneReason: %v", i, resp.DoneReason)
		}
		expected := "Batch response for: " + reqs[i].Prompt
		if resp.Content != expected {
			t.Errorf("Response %d content mismatch: got %q, want %q", i, resp.Content, expected)
		}
	}
}

func TestBatchCompletionUnsupported(t *testing.T) {
	// Disable batching for this test
	os.Setenv("OLLAMA_BATCH_ENABLED", "false")
	defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

	server := &mockLlamaServer{
		batchSupported: false,
		maxBatchSize:   8,
		estimatedVRAM:  1024 * 1024 * 1024,
		estimatedTotal: 2048 * 1024 * 1024,
		numParallel:    4,
	}

	ctx := context.Background()
	reqs := []CompletionRequest{
		{Prompt: "Hello"},
		{Prompt: "World"},
	}

	var responses []CompletionResponse
	err := server.BatchCompletion(ctx, reqs, func(resps []CompletionResponse) {
		responses = resps
	})

	if err != nil {
		t.Fatalf("BatchCompletion failed: %v", err)
	}

	if len(responses) != len(reqs) {
		t.Errorf("Expected %d responses, got %d", len(reqs), len(responses))
	}

	// Verify it fell back to individual processing
	for i, resp := range responses {
		if !resp.Done {
			t.Errorf("Response %d not marked as done", i)
		}
		// Individual processing should have different content pattern
		expected := "Test response for: " + reqs[i].Prompt
		if resp.Content != expected {
			t.Errorf("Response %d content mismatch: got %q, want %q", i, resp.Content, expected)
		}
	}
}

func TestBatchEmbedding(t *testing.T) {
	os.Setenv("OLLAMA_BATCH_ENABLED", "true")
	defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   8,
		estimatedVRAM:  1024 * 1024 * 1024,
		estimatedTotal: 2048 * 1024 * 1024,
		numParallel:    4,
	}

	ctx := context.Background()
	inputs := []string{"text1", "text2", "text3"}

	embeddings, err := server.BatchEmbedding(ctx, inputs)
	if err != nil {
		t.Fatalf("BatchEmbedding failed: %v", err)
	}

	if len(embeddings) != len(inputs) {
		t.Errorf("Expected %d embeddings, got %d", len(inputs), len(embeddings))
	}

	for i, embedding := range embeddings {
		if len(embedding) != 384 {
			t.Errorf("Embedding %d has wrong dimension: got %d, want 384", i, len(embedding))
		}
	}
}

func TestGetMaxBatchSize(t *testing.T) {
	tests := []struct {
		name         string
		numParallel  int
		batchEnabled bool
		expected     int
	}{
		{"parallel_enabled", 4, true, 8}, // numParallel * 2
		{"parallel_disabled", 1, true, 8}, // fallback to envconfig
		{"batch_disabled", 4, false, 8},   // fallback to envconfig
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.batchEnabled {
				os.Setenv("OLLAMA_BATCH_ENABLED", "true")
				os.Setenv("OLLAMA_BATCH_SIZE", "8")
			} else {
				os.Setenv("OLLAMA_BATCH_ENABLED", "false")
				os.Setenv("OLLAMA_BATCH_SIZE", "8")
			}
			defer func() {
				os.Unsetenv("OLLAMA_BATCH_ENABLED")
				os.Unsetenv("OLLAMA_BATCH_SIZE")
			}()

			server := &mockLlamaServer{
				batchSupported: tt.batchEnabled,
				maxBatchSize:   tt.expected,
				numParallel:    tt.numParallel,
			}

			result := server.GetMaxBatchSize()
			if result != tt.expected {
				t.Errorf("GetMaxBatchSize() = %d, want %d", result, tt.expected)
			}
		})
	}
}

func TestSupportsBatching(t *testing.T) {
	tests := []struct {
		name     string
		enabled  bool
		expected bool
	}{
		{"enabled", true, true},
		{"disabled", false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.enabled {
				os.Setenv("OLLAMA_BATCH_ENABLED", "true")
			} else {
				os.Setenv("OLLAMA_BATCH_ENABLED", "false")
			}
			defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

			server := &mockLlamaServer{
				batchSupported: tt.expected,
			}

			result := server.SupportsBatching()
			if result != tt.expected {
				t.Errorf("SupportsBatching() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestEstimateBatchMemory(t *testing.T) {
	server := &mockLlamaServer{
		estimatedTotal: 2048 * 1024 * 1024, // 2GB
		maxBatchSize:   8,
	}

	tests := []struct {
		batchSize int
		expected  uint64
	}{
		{1, 0},                                              // no additional memory for batch size 1
		{2, uint64(2048 * 1024 * 1024 * 0.5 * 2 / 8)},     // proportional additional memory
		{4, uint64(2048 * 1024 * 1024 * 0.5 * 4 / 8)},     // proportional additional memory
		{8, uint64(2048 * 1024 * 1024 * 0.5 * 8 / 8)},     // max additional memory
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("batch_size_%d", tt.batchSize), func(t *testing.T) {
			result := server.EstimateBatchMemory(tt.batchSize)
			if result != tt.expected {
				t.Errorf("EstimateBatchMemory(%d) = %d, want %d", tt.batchSize, result, tt.expected)
			}
		})
	}
}

func TestBatchSizeExceedsMax(t *testing.T) {
	os.Setenv("OLLAMA_BATCH_ENABLED", "true")
	defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   2,
	}

	ctx := context.Background()
	reqs := []CompletionRequest{
		{Prompt: "Hello"},
		{Prompt: "World"},
		{Prompt: "Too many"},
	}

	err := server.BatchCompletion(ctx, reqs, func([]CompletionResponse) {})
	if err == nil {
		t.Error("Expected error for batch size exceeding maximum, got nil")
		return
	}

	expectedError := "batch size 3 exceeds maximum 2"
	if err.Error() != expectedError {
		t.Errorf("Expected error %q, got %q", expectedError, err.Error())
	}
}

func TestBatchCompletionWithNilOptions(t *testing.T) {
	os.Setenv("OLLAMA_BATCH_ENABLED", "true")
	defer os.Unsetenv("OLLAMA_BATCH_ENABLED")

	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   8,
	}

	ctx := context.Background()
	reqs := []CompletionRequest{
		{Prompt: "Hello", Options: nil}, // nil options should be handled gracefully
	}

	var responses []CompletionResponse
	err := server.BatchCompletion(ctx, reqs, func(resps []CompletionResponse) {
		responses = resps
	})

	if err != nil {
		t.Fatalf("BatchCompletion with nil options failed: %v", err)
	}

	if len(responses) != 1 {
		t.Errorf("Expected 1 response, got %d", len(responses))
	}
}

func BenchmarkBatchVsIndividualCompletion(b *testing.B) {
	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   8,
		estimatedVRAM:  1024 * 1024 * 1024,
		estimatedTotal: 2048 * 1024 * 1024,
		numParallel:    4,
	}

	ctx := context.Background()
	reqs := []CompletionRequest{
		{Prompt: "Write a story about AI"},
		{Prompt: "Explain quantum computing"},
		{Prompt: "Describe machine learning"},
		{Prompt: "What is deep learning?"},
	}

	b.Run("individual", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, req := range reqs {
				err := server.Completion(ctx, req, func(CompletionResponse) {})
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			err := server.BatchCompletion(ctx, reqs, func([]CompletionResponse) {})
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkBatchEmbedding(b *testing.B) {
	server := &mockLlamaServer{
		batchSupported: true,
		maxBatchSize:   16,
	}

	ctx := context.Background()
	inputs := []string{
		"The quick brown fox",
		"jumps over the lazy dog",
		"machine learning is fascinating",
		"artificial intelligence future",
	}

	b.Run("individual", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, input := range inputs {
				_, err := server.Embedding(ctx, input)
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})

	b.Run("batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := server.BatchEmbedding(ctx, inputs)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
