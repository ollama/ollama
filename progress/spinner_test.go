package progress

import (
	"strings"
	"testing"
	"time"
)

func TestNewSpinner(t *testing.T) {
	spinner := NewSpinner("loading")
	defer spinner.Stop()

	if spinner.started.IsZero() {
		t.Error("spinner should have a start time")
	}

	if !spinner.stopped.IsZero() {
		t.Error("spinner should not be stopped initially")
	}

	if len(spinner.parts) == 0 {
		t.Error("spinner should have animation parts")
	}
}

func TestSpinnerSetMessage(t *testing.T) {
	spinner := NewSpinner("initial")
	defer spinner.Stop()

	spinner.SetMessage("updated")

	msg, ok := spinner.message.Load().(string)
	if !ok || msg != "updated" {
		t.Errorf("message = %q, want 'updated'", msg)
	}
}

func TestSpinnerString(t *testing.T) {
	spinner := NewSpinner("loading")
	defer spinner.Stop()

	str := spinner.String()

	// Should contain the message
	if !strings.Contains(str, "loading") {
		t.Errorf("String() should contain 'loading', got %q", str)
	}

	// Should contain one of the spinner characters
	hasSpinnerChar := false
	for _, part := range spinner.parts {
		if strings.Contains(str, part) {
			hasSpinnerChar = true
			break
		}
	}
	if !hasSpinnerChar {
		t.Errorf("String() should contain a spinner character, got %q", str)
	}
}

func TestSpinnerStringEmpty(t *testing.T) {
	spinner := NewSpinner("")
	defer spinner.Stop()

	str := spinner.String()

	// Should still have spinner character even with empty message
	hasSpinnerChar := false
	for _, part := range spinner.parts {
		if strings.Contains(str, part) {
			hasSpinnerChar = true
			break
		}
	}
	if !hasSpinnerChar {
		t.Errorf("String() with empty message should still contain spinner, got %q", str)
	}
}

func TestSpinnerStop(t *testing.T) {
	spinner := NewSpinner("test")

	if !spinner.stopped.IsZero() {
		t.Error("spinner should not be stopped initially")
	}

	spinner.Stop()

	if spinner.stopped.IsZero() {
		t.Error("spinner should be stopped after Stop()")
	}
}

func TestSpinnerStopIdempotent(t *testing.T) {
	spinner := NewSpinner("test")

	spinner.Stop()
	firstStopTime := spinner.stopped

	// Small delay to ensure different time if Stop() would reset
	time.Sleep(10 * time.Millisecond)

	spinner.Stop()
	secondStopTime := spinner.stopped

	if !firstStopTime.Equal(secondStopTime) {
		t.Error("Stop() should be idempotent - stopped time should not change")
	}
}

func TestSpinnerStringAfterStop(t *testing.T) {
	spinner := NewSpinner("done")
	spinner.Stop()

	str := spinner.String()

	// Should contain message
	if !strings.Contains(str, "done") {
		t.Errorf("String() after stop should contain message, got %q", str)
	}

	// Should NOT contain spinner character after stopping
	hasSpinnerChar := false
	for _, part := range spinner.parts {
		if strings.Contains(str, part) {
			hasSpinnerChar = true
			break
		}
	}
	if hasSpinnerChar {
		t.Errorf("String() after stop should not contain spinner character, got %q", str)
	}
}

func TestSpinnerMessageWidth(t *testing.T) {
	spinner := NewSpinner("this is a very long message that should be truncated")
	defer spinner.Stop()

	spinner.messageWidth = 10

	str := spinner.String()

	// Message should be truncated
	if strings.Contains(str, "very long") {
		t.Errorf("String() should truncate message when messageWidth is set, got %q", str)
	}
}

func TestSpinnerParts(t *testing.T) {
	spinner := NewSpinner("test")
	defer spinner.Stop()

	expectedParts := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

	if len(spinner.parts) != len(expectedParts) {
		t.Errorf("parts count = %d, want %d", len(spinner.parts), len(expectedParts))
	}

	for i, part := range expectedParts {
		if spinner.parts[i] != part {
			t.Errorf("parts[%d] = %q, want %q", i, spinner.parts[i], part)
		}
	}
}

func TestSpinnerValueWraps(t *testing.T) {
	spinner := NewSpinner("test")
	defer spinner.Stop()

	// Simulate multiple ticks
	for i := 0; i < 15; i++ {
		spinner.value = (spinner.value + 1) % len(spinner.parts)
	}

	// Value should have wrapped around
	if spinner.value < 0 || spinner.value >= len(spinner.parts) {
		t.Errorf("value = %d, should be in range [0, %d)", spinner.value, len(spinner.parts))
	}
}
