//go:build windows || darwin

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestExportChats(t *testing.T) {
	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	t.Cleanup(func() { st.Close() })

	started := time.Date(2025, 8, 14, 15, 4, 0, 0, time.Local)
	user := store.NewMessage("user", "What's in this picture?", &store.MessageOptions{
		Attachments: []store.File{{Filename: "cat.png", Data: []byte("png")}},
	})
	search := store.NewMessage("assistant", "", &store.MessageOptions{
		Model:    "gpt-oss:20b",
		Thinking: "I should look this up.",
		ToolCalls: []store.ToolCall{{
			Type:     "function",
			Function: store.ToolFunction{Name: "browser.search", Arguments: `{"query":"cats"}`},
		}},
	})
	result := store.NewMessage("tool", "Cats are small mammals.", nil)
	answer := store.NewMessage("assistant", "It's a **cat**.", &store.MessageOptions{Model: "gpt-oss:20b"})
	if err := st.SetChat(store.Chat{
		ID:        "cats",
		Title:     "Cats: a study?",
		CreatedAt: started,
		Messages:  []store.Message{user, search, result, answer},
	}); err != nil {
		t.Fatal(err)
	}
	if err := st.SetChat(store.Chat{
		ID:        "untitled",
		CreatedAt: started,
		Messages:  []store.Message{store.NewMessage("user", "hello\nthere", nil)},
	}); err != nil {
		t.Fatal(err)
	}
	// Chats without messages have nothing to export.
	if err := st.SetChat(store.Chat{ID: "empty", Title: "Empty", CreatedAt: started}); err != nil {
		t.Fatal(err)
	}

	dir := filepath.Join(t.TempDir(), "Ollama Chats")
	exported, err := exportChats(st, dir)
	if err != nil {
		t.Fatal(err)
	}
	if exported != 2 {
		t.Fatalf("exported %d chats, want 2", exported)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	want := []string{
		"2025-08-14 Cats a study attachments",
		"2025-08-14 Cats a study.md",
		"2025-08-14 hello.md",
	}
	if !slices.Equal(names, want) {
		t.Fatalf("exported files = %q, want %q", names, want)
	}

	image, err := os.ReadFile(filepath.Join(dir, "2025-08-14 Cats a study attachments", "cat.png"))
	if err != nil || string(image) != "png" {
		t.Fatalf("attachment = %q, %v", image, err)
	}

	markdown, err := os.ReadFile(filepath.Join(dir, "2025-08-14 Cats a study.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range []string{
		"# Cats: a study?\n",
		"## You\n\nWhat's in this picture?\n\n![cat.png](<2025-08-14 Cats a study attachments/cat.png>)",
		"## gpt-oss:20b\n\n<details>\n<summary>Thinking</summary>\n\nI should look this up.\n",
		"<summary>Used browser.search</summary>\n\n```\n{\"query\":\"cats\"}\n```",
		"<summary>browser.search result</summary>\n\n```\nCats are small mammals.\n```",
		"It's a **cat**.\n",
	} {
		if !strings.Contains(string(markdown), part) {
			t.Errorf("markdown is missing %q:\n%s", part, markdown)
		}
	}
	if strings.Count(string(markdown), "## gpt-oss:20b") != 1 {
		t.Errorf("one assistant turn should have one heading:\n%s", markdown)
	}

	// Exporting again replaces the same files instead of adding copies.
	if _, err := exportChats(st, dir); err != nil {
		t.Fatal(err)
	}
	if again, _ := os.ReadDir(dir); len(again) != len(entries) {
		t.Fatalf("second export left %d entries, want %d", len(again), len(entries))
	}
}

func TestSanitizeFileName(t *testing.T) {
	for _, tt := range []struct{ name, want string }{
		{"Plain title", "Plain title"},
		{`a/b\c:d*e?f"g<h>i|j`, "a b c d e f g h i j"},
		{"  spaced\n\tout  ", "spaced out"},
		{"...hidden", "hidden"},
		{"trailing. ", "trailing"},
		{"???", "Fallback"},
		{strings.Repeat("é", 100), strings.Repeat("é", maxChatFileNameLength)},
	} {
		if got := sanitizeFileName(tt.name, "Fallback"); got != tt.want {
			t.Errorf("sanitizeFileName(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestUniqueFileName(t *testing.T) {
	used := map[string]bool{}
	for _, want := range []string{"Chat.md", "Chat 2.md", "chat 3.md"} {
		name := "Chat.md"
		if want == "chat 3.md" {
			name = "chat.md"
		}
		if got := uniqueFileName(name, used); got != want {
			t.Errorf("uniqueFileName(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestCodeFenceIsLongerThanContent(t *testing.T) {
	if got := codeFence("plain"); got != "```" {
		t.Errorf("codeFence(plain) = %q", got)
	}
	if got := codeFence("has ```` inside"); got != "`````" {
		t.Errorf("codeFence(4 backticks) = %q", got)
	}
}
