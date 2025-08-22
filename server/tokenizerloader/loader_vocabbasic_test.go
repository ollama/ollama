package tokenizerloader

import (
	"context"
	"errors"
	"os"
	"sync/atomic"
	"testing"
)

// fake tokenizer that satisfies the current Tokenizer interface:
//   Tokenize(string) ([]int, error)
//   Detokenize([]int) (string, error)
//   Close() error
type fakeTok struct {
	tokens []int
	text   string
}

func (f *fakeTok) Tokenize(s string) ([]int, error)        { return f.tokens, nil }
func (f *fakeTok) Detokenize(ids []int) (string, error)     { return f.text, nil }
func (f *fakeTok) Close() error                             { return nil }

func TestGet_VocabOnlyPreferred_NoFallback(t *testing.T) {
	_ = os.Setenv("OLLAMA_TOKENIZER_DEBUG", "1")
	t.Cleanup(func() { _ = os.Unsetenv("OLLAMA_TOKENIZER_DEBUG") })

	ResetForTest()

	// Inject a vocab-only opener that SUCCEEDS
	wantTok := &fakeTok{tokens: []int{42, 99}, text: "ok"}
	SetOpenVocabOnlyForTest(func(ctx context.Context, model string) (Tokenizer, error) {
		return wantTok, nil
	})

	// Fallback hooks that should NOT be called
	var fallbackHit atomic.Int32
	RegisterFallbackHooks(
		func(modelName, text string) ([]int, error) {
			fallbackHit.Add(1)
			return nil, errors.New("should-not-be-called")
		},
		func(modelName string, tokens []int) (string, error) {
			fallbackHit.Add(1)
			return "", errors.New("should-not-be-called")
		},
	)

	tok, isFallback, err := Get(context.Background(), "mistral:latest")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if isFallback {
		t.Fatalf("expected vocab-only path, got fallback")
	}
	// Tokenize / Detokenize through vocab-only tokenizer:
	ids, err := tok.Tokenize("hello")
	if err != nil {
		t.Fatalf("tokenize err: %v", err)
	}
	if len(ids) != 2 || ids[0] != 42 || ids[1] != 99 {
		t.Fatalf("unexpected tokens: %v", ids)
	}
	s, err := tok.Detokenize(ids)
	if err != nil {
		t.Fatalf("detokenize err: %v", err)
	}
	if s != "ok" {
		t.Fatalf("unexpected detokenize content: %q", s)
	}
	if err := tok.Close(); err != nil {
		t.Fatalf("close err: %v", err)
	}

	if fallbackHit.Load() != 0 {
		t.Fatalf("fallback should not be hit when vocab-only succeeds")
	}
}

func TestGet_FallbackWhenVocabOnlyUnavailable(t *testing.T) {
	ResetForTest()

	// Inject a vocab-only opener that reports unavailability
	SetOpenVocabOnlyForTest(func(ctx context.Context, model string) (Tokenizer, error) {
		return nil, errVocabOnlyUnavailable
	})

	// Fallback hooks emulate scheduler behavior
	var calls atomic.Int32
	RegisterFallbackHooks(
		func(modelName, text string) ([]int, error) {
			calls.Add(1)
			return []int{7, 8, 9}, nil
		},
		func(modelName string, tokens []int) (string, error) {
			calls.Add(1)
			return "fallback", nil
		},
	)

	tok, isFallback, err := Get(context.Background(), "tinyllama:latest")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if !isFallback {
		t.Fatalf("expected fallback path when vocab-only unavailable")
	}

	ids, err := tok.Tokenize("x")
	if err != nil {
		t.Fatalf("tokenize err: %v", err)
	}
	if len(ids) != 3 || ids[0] != 7 {
		t.Fatalf("unexpected tokens via fallback: %v", ids)
	}
	s, err := tok.Detokenize(ids)
	if err != nil {
		t.Fatalf("detokenize err: %v", err)
	}
	if s != "fallback" {
		t.Fatalf("unexpected detokenize via fallback: %q", s)
	}
	if err := tok.Close(); err != nil {
		t.Fatalf("close err: %v", err)
	}

	if calls.Load() == 0 {
		t.Fatalf("expected fallback hooks to be used")
	}
}
