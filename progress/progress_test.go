package progress

import (
	"bufio"
	"bytes"
	"strings"
	"testing"
	"time"
)

// TestNonTTYNoDuplicateLines verifies that in non-TTY mode a spinner
// message is printed exactly once regardless of how many render ticks fire.
func TestNonTTYNoDuplicateLines(t *testing.T) {
	var buf bytes.Buffer
	p := &Progress{w: bufio.NewWriter(&buf), tty: false}
	go p.start()

	p.Add("status", NewSpinner("pulling manifest"))

	// Wait for several render ticks (ticker fires every 100ms)
	time.Sleep(350 * time.Millisecond)

	p.Stop()
	p.w.Flush()

	got := strings.TrimSpace(buf.String())
	lines := strings.Split(got, "\n")
	count := 0
	for _, l := range lines {
		if strings.Contains(l, "pulling manifest") {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected exactly 1 line containing %q, got %d:\n%s", "pulling manifest", count, buf.String())
	}
}

// TestNonTTYMultipleStates verifies each distinct status prints once in non-TTY mode.
func TestNonTTYMultipleStates(t *testing.T) {
	var buf bytes.Buffer
	p := &Progress{w: bufio.NewWriter(&buf), tty: false}
	go p.start()

	s1 := NewSpinner("pulling manifest")
	p.Add("s1", s1)
	time.Sleep(150 * time.Millisecond)

	s1.Stop()
	s2 := NewSpinner("verifying sha256 digest")
	p.Add("s2", s2)
	time.Sleep(150 * time.Millisecond)

	p.Stop()
	p.w.Flush()

	out := buf.String()
	if strings.Count(out, "pulling manifest") != 1 {
		t.Errorf("expected 1 occurrence of %q:\n%s", "pulling manifest", out)
	}
	if strings.Count(out, "verifying sha256 digest") != 1 {
		t.Errorf("expected 1 occurrence of %q:\n%s", "verifying sha256 digest", out)
	}
}

// TestPlainStringerSpinner checks that Spinner.PlainString returns just the message.
func TestPlainStringerSpinner(t *testing.T) {
	s := NewSpinner("pulling manifest")
	got := s.PlainString()
	if got != "pulling manifest" {
		t.Errorf("PlainString() = %q, want %q", got, "pulling manifest")
	}
	s.Stop()
}

// TestPlainStringerBar checks that Bar.PlainString returns message + percentage.
func TestPlainStringerBar(t *testing.T) {
	b := NewBar("pulling abc123:", 100, 0)
	b.Set(50)
	got := b.PlainString()
	if !strings.Contains(got, "pulling abc123:") || !strings.Contains(got, "50%") {
		t.Errorf("PlainString() = %q, want message and percentage", got)
	}
}
