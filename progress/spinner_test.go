package progress

import (
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestSpinnerStopRace exercises the three goroutines that touch a Spinner
// concurrently — the renderer (String), message updates (SetMessage), and
// cancellation (Stop) — across several animation ticks so the value updates
// in start() genuinely overlap the readers. Run with -race.
func TestSpinnerStopRace(t *testing.T) {
	for range 3 {
		s := NewSpinner("test")
		quit := make(chan struct{})

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			for {
				select {
				case <-quit:
					return
				default:
					_ = s.String()
				}
			}
		}()

		go func() {
			defer wg.Done()
			for {
				select {
				case <-quit:
					return
				default:
					s.SetMessage("updated")
				}
			}
		}()

		// span at least one 100ms animation tick so the goroutine's value
		// writes overlap the String reads
		time.Sleep(150 * time.Millisecond)
		s.Stop()
		close(quit)
		wg.Wait()
	}
}

// TestSpinnerConcurrentStop verifies concurrent Stop calls are safe: the
// ticker must be stopped and the done channel closed exactly once.
func TestSpinnerConcurrentStop(t *testing.T) {
	for range 200 {
		s := NewSpinner("test")

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			s.Stop()
		}()
		go func() {
			defer wg.Done()
			s.Stop()
		}()
		wg.Wait()
	}
}

// TestSpinnerStopIdempotent verifies Stop can be called multiple times
// sequentially without double-stopping the ticker or double-closing the done
// channel.
func TestSpinnerStopIdempotent(t *testing.T) {
	s := NewSpinner("test")
	s.Stop()
	s.Stop()
}

// TestSpinnerStringAfterStop verifies a stopped spinner still shows its
// message but no longer shows an animation frame.
func TestSpinnerStringAfterStop(t *testing.T) {
	s := NewSpinner("hello")

	if got := s.String(); !strings.Contains(got, "hello") {
		t.Errorf("running spinner should show its message, got %q", got)
	}

	s.Stop()

	got := s.String()
	if !strings.Contains(got, "hello") {
		t.Errorf("stopped spinner should still show its message, got %q", got)
	}
	for _, part := range s.parts {
		if strings.Contains(got, part) {
			t.Errorf("stopped spinner still shows an animation frame: %q", got)
		}
	}
}

// TestSpinnerStopStopsTicker verifies Stop actually stops the ticker: no
// tick may arrive after Stop returns. Before this fix ticker.Stop was never
// called, so the ticker kept firing for the life of the process.
func TestSpinnerStopStopsTicker(t *testing.T) {
	s := NewSpinner("test")
	time.Sleep(10 * time.Millisecond)
	s.Stop()

	// drain a tick that may already have been buffered when Stop ran
	select {
	case <-s.ticker.C:
	case <-time.After(150 * time.Millisecond):
	}

	select {
	case <-s.ticker.C:
		t.Fatal("ticker still firing after Stop")
	case <-time.After(250 * time.Millisecond):
	}
}

// TestSpinnerAdvancesFrames verifies the animation goroutine actually
// advances through the frame glyphs while running.
func TestSpinnerAdvancesFrames(t *testing.T) {
	s := NewSpinner("")
	defer s.Stop()

	seen := make(map[string]bool)
	deadline := time.Now().Add(2 * time.Second)
	for len(seen) < 3 && time.Now().Before(deadline) {
		if frame := strings.TrimSpace(s.String()); frame != "" {
			seen[frame] = true
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(seen) < 3 {
		t.Fatalf("spinner did not advance through its frames: saw %d distinct frames", len(seen))
	}
}

// TestSpinnerEmptyMessage verifies a spinner with no message renders just the
// frame glyph.
func TestSpinnerEmptyMessage(t *testing.T) {
	s := NewSpinner("")
	defer s.Stop()

	if got, want := s.String(), s.parts[0]+" "; got != want {
		t.Errorf("empty-message spinner: got %q, want %q", got, want)
	}
}

// TestSpinnerMessageWidth verifies the message truncation and padding logic.
func TestSpinnerMessageWidth(t *testing.T) {
	long := NewSpinner("this is a long message")
	defer long.Stop()
	long.messageWidth = 10
	if got, want := long.String(), "this is a "+" "+long.parts[0]+" "; got != want {
		t.Errorf("truncated message: got %q, want %q", got, want)
	}

	short := NewSpinner("hi")
	defer short.Stop()
	short.messageWidth = 10
	if got, want := short.String(), "hi"+strings.Repeat(" ", 8)+" "+short.parts[0]+" "; got != want {
		t.Errorf("padded message: got %q, want %q", got, want)
	}
}

// TestSpinnerStopReleasesGoroutine verifies the animation goroutine exits on
// Stop instead of leaking for the process lifetime.
func TestSpinnerStopReleasesGoroutine(t *testing.T) {
	before := runtime.NumGoroutine()

	for range 50 {
		NewSpinner("test").Stop()
	}

	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > before+2 {
		if time.Now().After(deadline) {
			t.Fatalf("spinner goroutines leaked: %d before, %d after", before, runtime.NumGoroutine())
		}
		time.Sleep(10 * time.Millisecond)
	}
}
