package progress

import (
	"strings"
	"testing"
)

func TestNewStepBar(t *testing.T) {
	t.Run("creates stepbar with message and total", func(t *testing.T) {
		sb := NewStepBar("Generating", 10)
		if sb == nil {
			t.Fatal("NewStepBar returned nil")
		}
		if sb.message != "Generating" {
			t.Errorf("message = %q, want %q", sb.message, "Generating")
		}
		if sb.total != 10 {
			t.Errorf("total = %d, want %d", sb.total, 10)
		}
		if sb.current != 0 {
			t.Errorf("current = %d, want 0", sb.current)
		}
	})

	t.Run("creates stepbar with empty message", func(t *testing.T) {
		sb := NewStepBar("", 5)
		if sb.message != "" {
			t.Errorf("message = %q, want empty", sb.message)
		}
	})
}

func TestStepBar_Set(t *testing.T) {
	t.Run("sets current value", func(t *testing.T) {
		sb := NewStepBar("Processing", 10)
		sb.Set(5)
		if sb.current != 5 {
			t.Errorf("current = %d, want 5", sb.current)
		}
	})

	t.Run("can set to zero", func(t *testing.T) {
		sb := NewStepBar("Processing", 10)
		sb.Set(5)
		sb.Set(0)
		if sb.current != 0 {
			t.Errorf("current = %d, want 0", sb.current)
		}
	})

	t.Run("can set to total", func(t *testing.T) {
		sb := NewStepBar("Processing", 10)
		sb.Set(10)
		if sb.current != 10 {
			t.Errorf("current = %d, want 10", sb.current)
		}
	})

	t.Run("can exceed total", func(t *testing.T) {
		sb := NewStepBar("Processing", 10)
		sb.Set(15)
		if sb.current != 15 {
			t.Errorf("current = %d, want 15", sb.current)
		}
	})
}

func TestStepBar_String(t *testing.T) {
	t.Run("formats zero progress", func(t *testing.T) {
		sb := NewStepBar("Generating", 9)
		sb.Set(0)
		output := sb.String()

		if !strings.Contains(output, "Generating") {
			t.Errorf("output missing message: %q", output)
		}
		if !strings.Contains(output, "0%") {
			t.Errorf("output missing 0%%: %q", output)
		}
		if !strings.Contains(output, "0/9") {
			t.Errorf("output missing step count: %q", output)
		}
		if !strings.Contains(output, "▕") || !strings.Contains(output, "▏") {
			t.Errorf("output missing bar boundaries: %q", output)
		}
	})

	t.Run("formats partial progress", func(t *testing.T) {
		sb := NewStepBar("Generating", 10)
		sb.Set(5)
		output := sb.String()

		if !strings.Contains(output, "50%") {
			t.Errorf("output missing 50%%: %q", output)
		}
		if !strings.Contains(output, "5/10") {
			t.Errorf("output missing step count: %q", output)
		}

		completed := strings.Count(output, "█")
		if completed != 5 {
			t.Errorf("completed blocks = %d, want 5", completed)
		}
	})

	t.Run("formats complete progress", func(t *testing.T) {
		sb := NewStepBar("Generating", 5)
		sb.Set(5)
		output := sb.String()

		if !strings.Contains(output, "100%") {
			t.Errorf("output missing 100%%: %q", output)
		}
		if !strings.Contains(output, "5/5") {
			t.Errorf("output missing step count: %q", output)
		}
	})

	t.Run("formats with empty message", func(t *testing.T) {
		sb := NewStepBar("", 3)
		sb.Set(1)
		output := sb.String()

		if !strings.Contains(output, "33%") {
			t.Errorf("output missing 33%%: %q", output)
		}
		if !strings.Contains(output, "1/3") {
			t.Errorf("output missing step count: %q", output)
		}
	})

	t.Run("formats single step", func(t *testing.T) {
		sb := NewStepBar("Step", 1)
		sb.Set(1)
		output := sb.String()

		if !strings.Contains(output, "100%") {
			t.Errorf("output missing 100%%: %q", output)
		}
		if !strings.Contains(output, "1/1") {
			t.Errorf("output missing step count: %q", output)
		}
	})

	t.Run("handles zero total gracefully", func(t *testing.T) {
		sb := NewStepBar("Test", 0)
		output := sb.String()

		if output == "" {
			t.Error("output should not be empty even with zero total")
		}
	})
}