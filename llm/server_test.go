package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestLLMServerCompletionFormat(t *testing.T) {
	// This test was written to fix an already deployed issue. It is a bit
	// of a mess, and but it's good enough, until we can refactoring the
	// Completion method to be more testable.

	ctx, cancel := context.WithCancel(context.Background())
	s := &llmServer{
		sem: semaphore.NewWeighted(1), // required to prevent nil panic
	}

	checkInvalid := func(format string) {
		t.Helper()
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(format),
		}, nil)

		want := fmt.Sprintf("invalid format: %q; expected \"json\" or a valid JSON Schema", format)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v; want %q", err, want)
		}
	}

	checkInvalid("X")   // invalid format
	checkInvalid(`"X"`) // invalid JSON Schema

	cancel() // prevent further processing if request makes it past the format check

	checkValid := func(err error) {
		t.Helper()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Completion: err = %v; expected context.Canceled", err)
		}
	}

	valids := []string{
		// "missing"
		``,
		`""`,
		`null`,

		// JSON
		`"json"`,
		`{"type":"object"}`,
	}
	for _, valid := range valids {
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(valid),
		}, nil)
		checkValid(err)
	}

	err := s.Completion(ctx, CompletionRequest{
		Options: new(api.Options),
		Format:  nil, // missing format
	}, nil)
	checkValid(err)
}
