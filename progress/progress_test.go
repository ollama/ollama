package progress

import (
	"bytes"
	"os"
	"strings"
	"testing"
	"time"
)

func TestNewProgressNonTTY(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("test", NewSpinner("loading"))

	// let the ticker fire a few times
	time.Sleep(350 * time.Millisecond)
	p.Stop()

	output := buf.String()
	if strings.Contains(output, "\033[") {
		t.Errorf("non-TTY progress should not contain ANSI escape sequences, got: %q", output)
	}
}

func TestNewProgressNonTTYStopAndClear(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("test", NewSpinner("loading"))

	time.Sleep(350 * time.Millisecond)
	p.StopAndClear()

	output := buf.String()
	if strings.Contains(output, "\033[") {
		t.Errorf("non-TTY StopAndClear should not contain ANSI escape sequences, got: %q", output)
	}
}

func TestNewProgressTTYDetection(t *testing.T) {
	// bytes.Buffer does not implement Fd(), so isTerm should be false
	var buf bytes.Buffer
	p := NewProgress(&buf)
	if p.isTerm {
		t.Error("expected isTerm=false for bytes.Buffer")
	}
	p.Stop()

	// os.Stderr implements Fd(), isTerm depends on environment
	// in CI/test it's typically not a TTY
	p2 := NewProgress(os.Stderr)
	p2.Stop()
	// just verify it doesn't panic
}
