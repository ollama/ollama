package template

import (
	"strings"
	"testing"
)

func TestFormatIndentation(t *testing.T) {
	tmpl := "{{ if .Cond }}A{{ else }}B{{ end }}"
	out, err := Format(tmpl)
	if err != nil {
		t.Fatal(err)
	}
	expectedLines := []string{
		"{{ if .Cond }}",
		"  A",
		"{{ else }}",
		"  B",
		"{{ end }}",
	}
	got := strings.Split(strings.TrimSpace(out), "\n")
	if len(got) != len(expectedLines) {
		t.Fatalf("expected %d lines, got %d: %q", len(expectedLines), len(got), out)
	}
	for i, line := range expectedLines {
		if strings.TrimSpace(got[i]) != strings.TrimSpace(line) {
			t.Errorf("line %d = %q, want %q", i, got[i], line)
		}
	}
}
