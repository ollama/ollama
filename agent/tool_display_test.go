package agent

import (
	"strings"
	"testing"
)

func TestToolInvocationLabelTruncatesLongBashCommand(t *testing.T) {
	command := strings.Repeat("a", 101)
	label := ToolInvocationLabel("bash", map[string]any{"command": command})
	want := `Bash("` + strings.Repeat("a", 100) + `...")`
	if label != want {
		t.Fatalf("label = %q, want %q", label, want)
	}
}

func TestToolInvocationLabelCountsRunes(t *testing.T) {
	command := strings.Repeat("界", 101)
	label := ToolInvocationLabel("bash", map[string]any{"command": command})
	want := `Bash("` + strings.Repeat("界", 100) + `...")`
	if label != want {
		t.Fatalf("label = %q, want %q", label, want)
	}
}
