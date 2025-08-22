package tokenizerloader

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
)

// Mock vocabOnlyModel for testing
type mockVocabOnlyModel struct {
	modelName string
}

func (m *mockVocabOnlyModel) Tokenize(text string) ([]int, error) {
	return []int{1, 2, 3}, nil // mock tokens
}

func (m *mockVocabOnlyModel) Detokenize(tokens []int) (string, error) {
	return "mock text", nil
}

func (m *mockVocabOnlyModel) Close() error {
	return nil
}

func TestLRUEvictionDoesNotPanic(t *testing.T) {
	c := cache()
	c.capacity = 2
	m1 := &mockVocabOnlyModel{modelName: "a"}
	m2 := &mockVocabOnlyModel{modelName: "b"}
	m3 := &mockVocabOnlyModel{modelName: "c"}
	c.add("a", m1)
	c.add("b", m2)
	c.add("c", m3) // evicts "a"
	// no assertions needed, just ensure no panic and mapping sizes make sense
	if _, ok := c.items["a"]; ok {
		t.Fatalf("expected 'a' to be evicted")
	}
}

func TestConcurrencySafety(t *testing.T) {
	c := cache()
	c.capacity = 100 // increase capacity for this test

	var wg sync.WaitGroup
	numGoroutines := 10
	iterations := 100

	// Test concurrent adds
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				modelName := fmt.Sprintf("model-%d-%d", id, j)
				model := &mockVocabOnlyModel{modelName: modelName}
				c.add(modelName, model)
			}
		}(i)
	}

	wg.Wait()

	// Test concurrent gets
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				modelName := fmt.Sprintf("model-%d-%d", id, j)
				c.get(modelName)
			}
		}(i)
	}

	wg.Wait()
}

func TestDebugFlagExecution(t *testing.T) {
	// Test that both code paths execute without panic
	// We can't assert logs, just execute both code paths

	// Test vocab-only path (should return error)
	_, err := getVocabOnly("test-model")
	if err == nil {
		t.Fatalf("expected error from vocab-only path")
	}

	// Test fallback path
	_, err = newFallbackTokenizer("test-model")
	if err != nil {
		t.Fatalf("expected no error from fallback path")
	}

	// Test with debug flag set
	os.Setenv("OLLAMA_TOKENIZER_DEBUG", "1")
	defer os.Unsetenv("OLLAMA_TOKENIZER_DEBUG")

	// Should still work without panic
	_, err = getVocabOnly("test-model")
	if err == nil {
		t.Fatalf("expected error from vocab-only path with debug flag")
	}

	// Test Get function with new signature
	tk, isFallback, err := Get(context.Background(), "test-model")
	if err != nil {
		t.Fatalf("expected no error from Get function")
	}
	if !isFallback {
		t.Fatalf("expected fallback to be true")
	}
	if tk == nil {
		t.Fatalf("expected tokenizer to be returned")
	}
}
