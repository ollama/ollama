package readline

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestHistoryPersistsEntries(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	entries := []string{
		"first prompt",
		"summarize this:\nline one\nline two",
		`"quoted"`,
		`a "quote" inside`,
	}

	h, err := NewHistory()
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		h.Add(e)
	}

	reloaded, err := NewHistory()
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Buf.Values(); !slices.Equal(got, entries) {
		t.Fatalf("reloaded history = %q, want %q", got, entries)
	}
}

func TestHistoryReadsPlainLines(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	path := filepath.Join(home, ".ollama", "history")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("hello\nwhy is the sky blue?\n\"unterminated\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	h, err := NewHistory()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"hello", "why is the sky blue?", `"unterminated`}
	if got := h.Buf.Values(); !slices.Equal(got, want) {
		t.Fatalf("history = %q, want %q", got, want)
	}
}
