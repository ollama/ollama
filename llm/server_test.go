package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
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

func TestLibOllama(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	t.Run("default executable dir", func(t *testing.T) {
		got, err := libOllama()
		if err != nil {
			t.Fatal(err)
		}
		if got != filepath.Dir(exe) {
			t.Fatalf("expected %s, got %s", filepath.Dir(exe), got)
		}
	})

	t.Run("adjacent lib dir", func(t *testing.T) {
		want := filepath.Join(filepath.Dir(exe), envconfig.LibRelativeToExe(), "lib", "ollama")
		if err := os.MkdirAll(want, 0755); err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(want)

		got, err := libOllama()
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("expected %s, got %s", want, got)
		}
	})

	t.Run("build dir relative to executable", func(t *testing.T) {
		want := filepath.Join(filepath.Dir(exe), "build", "lib", "ollama")
		if err := os.MkdirAll(want, 0755); err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(want)

		got, err := libOllama()
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("expected %s, got %s", want, got)
		}
	})
}
