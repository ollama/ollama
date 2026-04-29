package progress

import (
	"strings"
	"testing"
	"time"
)

func TestNewSpinner(t *testing.T) {
	t.Run("creates spinner with message", func(t *testing.T) {
		s := NewSpinner("Loading")
		if s == nil {
			t.Fatal("NewSpinner returned nil")
		}
		if s.message.Load().(string) != "Loading" {
			t.Errorf("message = %q, want %q", s.message.Load(), "Loading")
		}
		if s.started.IsZero() {
			t.Error("started should be non-zero after creation")
		}
	})

	t.Run("creates spinner with empty message", func(t *testing.T) {
		s := NewSpinner("")
		if s.message.Load().(string) != "" {
			t.Errorf("message = %q, want empty", s.message.Load())
		}
	})

	t.Run("has spinner parts", func(t *testing.T) {
		s := NewSpinner("test")
		if len(s.parts) == 0 {
			t.Error("spinner parts should not be empty")
		}
	})
}

func TestSpinner_SetMessage(t *testing.T) {
	t.Run("updates message", func(t *testing.T) {
		s := NewSpinner("Loading")
		s.SetMessage("Processing")
		if s.message.Load().(string) != "Processing" {
			t.Errorf("message = %q, want %q", s.message.Load(), "Processing")
		}
	})

	t.Run("can set empty message", func(t *testing.T) {
		s := NewSpinner("Loading")
		s.SetMessage("")
		if s.message.Load().(string) != "" {
			t.Errorf("message = %q, want empty", s.message.Load())
		}
	})

	t.Run("concurrent message updates", func(t *testing.T) {
		s := NewSpinner("initial")
		done := make(chan bool)

		for i := 0; i < 10; i++ {
			go func(idx int) {
				s.SetMessage(string(rune('A' + idx%26)))
				done <- true
			}(i)
		}

		for i := 0; i < 10; i++ {
			<-done
		}

		msg := s.message.Load().(string)
		if len(msg) != 1 {
			t.Errorf("message should be single character, got %q", msg)
		}
	})
}

func TestSpinner_String(t *testing.T) {
	t.Run("contains message", func(t *testing.T) {
		s := NewSpinner("Loading")
		s.Stop()
		output := s.String()
		if !strings.Contains(output, "Loading") {
			t.Errorf("output missing message: %q", output)
		}
	})

	t.Run("running spinner shows animation character", func(t *testing.T) {
		s := NewSpinner("Processing")
		time.Sleep(150 * time.Millisecond)
		output := s.String()
		s.Stop()

		found := false
		for _, part := range s.parts {
			if strings.Contains(output, part) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("output should contain spinner character: %q", output)
		}
	})

	t.Run("stopped spinner omits animation", func(t *testing.T) {
		s := NewSpinner("Done")
		s.Stop()
		output := s.String()

		for _, part := range s.parts {
			if strings.Contains(output, part) {
				t.Errorf("stopped spinner should not contain animation: %q", output)
			}
		}
	})

	t.Run("empty message handled", func(t *testing.T) {
		s := NewSpinner("")
		s.Stop()
		output := s.String()
		if output != "" {
			t.Errorf("empty stopped spinner output = %q, want empty", output)
		}
	})
}

func TestSpinner_Stop(t *testing.T) {
	t.Run("stops the spinner", func(t *testing.T) {
		s := NewSpinner("test")
		if !s.stopped.IsZero() {
			t.Error("stopped should be zero before Stop()")
		}

		s.Stop()

		if s.stopped.IsZero() {
			t.Error("stopped should not be zero after Stop()")
		}
	})

	t.Run("idempotent stop", func(t *testing.T) {
		s := NewSpinner("test")
		s.Stop()
		stoppedTime := s.stopped

		s.Stop()

		if s.stopped != stoppedTime {
			t.Error("multiple Stop() calls should not change stopped time")
		}
	})
}