package readline

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/emirpasic/gods/v2/lists/arraylist"
)

// newTestHistory returns a History whose file lives under a temporary home
// directory. Autosave is off so that loading does not rewrite the file.
func newTestHistory(t *testing.T, limit int) (*History, string) {
	t.Helper()

	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	h := &History{
		Buf:     arraylist.New[string](),
		Limit:   limit,
		Enabled: true,
	}
	return h, filepath.Join(home, ".ollama", "history")
}

func writeHistoryFile(t *testing.T, path, content string) {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryInitKeepsLastLineWithoutTrailingNewline(t *testing.T) {
	h, path := newTestHistory(t, 100)
	writeHistoryFile(t, path, "one\ntwo")

	if err := h.Init(); err != nil {
		t.Fatal(err)
	}

	want := []string{"one", "two"}
	if got := h.Buf.Values(); !slices.Equal(got, want) {
		t.Fatalf("history = %q, want %q", got, want)
	}
}

func TestHistoryInitSkipsBlankLinesAndHonorsLimit(t *testing.T) {
	h, path := newTestHistory(t, 2)
	writeHistoryFile(t, path, "one\n\n  \ntwo\nthree\n")

	if err := h.Init(); err != nil {
		t.Fatal(err)
	}

	want := []string{"two", "three"}
	if got := h.Buf.Values(); !slices.Equal(got, want) {
		t.Fatalf("history = %q, want %q", got, want)
	}
}

func TestHistorySaveInitRoundTrip(t *testing.T) {
	h, _ := newTestHistory(t, 100)
	h.Init()

	entries := []string{"why is the sky blue?", "hello", "/set verbose"}
	for _, e := range entries {
		h.Buf.Add(e)
	}
	if err := h.Save(); err != nil {
		t.Fatal(err)
	}

	reloaded := &History{Buf: arraylist.New[string](), Limit: 100, Enabled: true}
	if err := reloaded.Init(); err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Buf.Values(); !slices.Equal(got, entries) {
		t.Fatalf("reloaded history = %q, want %q", got, entries)
	}
}

func TestHistorySaveKeepsExistingFileOnError(t *testing.T) {
	h, path := newTestHistory(t, 100)
	const original = "keep\nme\n"
	writeHistoryFile(t, path, original)
	h.Filename = path
	h.Buf.Add("new entry")

	// A directory at the temporary path makes the write fail on every platform.
	if err := os.Mkdir(path+".tmp", 0o755); err != nil {
		t.Fatal(err)
	}

	if err := h.Save(); err == nil {
		t.Fatal("Save returned nil, want an error")
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != original {
		t.Fatalf("history file = %q, want %q", got, original)
	}
}
