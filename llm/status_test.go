package llm

import (
	"fmt"
	"os"
	"sync"
	"testing"
)

// TestStatusWriterConcurrency verifies that concurrent writes and reads of
// LastErrMsg via Write() and Err() do not produce a data race.
// Run with: go test -race ./llm/ -run TestStatusWriterConcurrency
func TestStatusWriterConcurrency(t *testing.T) {
	w := NewStatusWriter(os.Stderr)

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines * 2)

	// Writers: simulate the stderr goroutine calling Write with error prefixes.
	for i := range goroutines {
		go func(n int) {
			defer wg.Done()
			payload := []byte(fmt.Sprintf("error: simulated error %d", n))
			_, _ = w.Write(payload)
		}(i)
	}

	// Readers: simulate callers reading Err() from multiple goroutines.
	for range goroutines {
		go func() {
			defer wg.Done()
			_ = w.Err()
		}()
	}

	wg.Wait()

	// After all writes, Err() must return a non-empty string (one of the writes won).
	if w.Err() == "" {
		t.Error("expected LastErrMsg to be set after concurrent writes, got empty string")
	}
}

// TestStatusWriterSetErr verifies SetErr updates LastErrMsg safely.
func TestStatusWriterSetErr(t *testing.T) {
	w := NewStatusWriter(os.Stderr)
	const msg = "this model is not supported by your version of Ollama. You may need to upgrade"
	w.SetErr(msg)
	if got := w.Err(); got != msg {
		t.Errorf("Err() = %q, want %q", got, msg)
	}
}
