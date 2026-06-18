package tui

import (
	"context"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestChatStartRunAttachesDroppedImagePath(t *testing.T) {
	fp := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, cmd := m.startRun("describe " + fp)
	m = updated.(chatModel)

	if cmd == nil {
		t.Fatal("startRun should return a command")
	}
	if len(m.liveMessages) != 1 {
		t.Fatalf("liveMessages = %d, want 1", len(m.liveMessages))
	}
	if got := m.liveMessages[0].Content; got != "describe" {
		t.Fatalf("content = %q, want describe", got)
	}
	if got := len(m.liveMessages[0].Images); got != 1 {
		t.Fatalf("images = %d, want 1", got)
	}
	if len(m.entries) == 0 {
		t.Fatal("missing user transcript entry")
	}
	entry := m.entries[0].content
	if strings.Contains(entry, fp) {
		t.Fatalf("transcript entry should hide local file path: %q", entry)
	}
	if !strings.Contains(entry, "describe") || !strings.Contains(entry, "[attached 1 file]") {
		t.Fatalf("transcript entry = %q, want prompt plus attachment note", entry)
	}
}

func TestChatStartRunAttachesDroppedFileURL(t *testing.T) {
	fp := writeTestPNG(t)
	fileURL := (&url.URL{Scheme: "file", Path: fp}).String()
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, _ := m.startRun(fileURL)
	m = updated.(chatModel)

	if got := m.liveMessages[0].Content; got != "" {
		t.Fatalf("content = %q, want empty prompt after extracting file URL", got)
	}
	if got := len(m.liveMessages[0].Images); got != 1 {
		t.Fatalf("images = %d, want 1", got)
	}
	if got := m.entries[0].content; got != "[attached 1 file]" {
		t.Fatalf("transcript entry = %q, want attachment-only note", got)
	}
}

func TestChatPasteImagePathAttachesOnSubmit(t *testing.T) {
	fp := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("describe " + fp), Paste: true})
	m = updated.(chatModel)
	if got := string(m.input); got != "describe [Image #0]" {
		t.Fatalf("pasted path input = %q, want placeholder", got)
	}
	if got := m.notificationLine(); got != "attached image" {
		t.Fatalf("notification = %q, want attached image", got)
	}
	if got := string(m.input); strings.Contains(got, fp) {
		t.Fatalf("pasted path should be hidden behind placeholder, input = %q", got)
	}
	if completions := m.slashCompletions(); len(completions) != 0 {
		t.Fatalf("placeholder input should not show slash completions: %#v", completions)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("submit should start a run")
	}
	if got := m.liveMessages[0].Content; got != "describe [Image #0]" {
		t.Fatalf("content = %q, want prompt with placeholder", got)
	}
	if got := len(m.liveMessages[0].Images); got != 1 {
		t.Fatalf("images = %d, want 1", got)
	}
	if strings.Contains(m.entries[0].content, fp) {
		t.Fatalf("transcript entry should hide pasted file path: %q", m.entries[0].content)
	}
	if !strings.Contains(m.entries[0].content, "[Image #0]") {
		t.Fatalf("transcript entry should show placeholder: %q", m.entries[0].content)
	}
}

func TestChatImagePlaceholdersUseSessionNumbers(t *testing.T) {
	first := writeTestPNG(t)
	second := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(first), Paste: true})
	m = updated.(chatModel)
	if got := string(m.input); got != "[Image #0]" {
		t.Fatalf("first placeholder = %q, want [Image #0]", got)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("submit should start a run")
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(second), Paste: true})
	m = updated.(chatModel)
	if got := string(m.input); got != "[Image #1]" {
		t.Fatalf("second placeholder = %q, want [Image #1]", got)
	}
}

func TestChatAbsoluteImagePathBypassesSlashCommandParsing(t *testing.T) {
	fp := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	m.input = []rune(fp)
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	updated, cmd := m.handleSubmit()
	m = updated.(chatModel)

	if cmd == nil {
		t.Fatal("absolute image path should start a run instead of being parsed as a slash command")
	}
	if got := len(m.liveMessages[0].Images); got != 1 {
		t.Fatalf("images = %d, want 1", got)
	}
	if got := m.entries[0].role; got != "user" {
		t.Fatalf("entry role = %q, want user", got)
	}
}

func TestChatDeletingImagePlaceholderRemovesAttachment(t *testing.T) {
	fp := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("describe " + fp), Paste: true})
	m = updated.(chatModel)
	if got := len(m.inputAttachments); got != 1 {
		t.Fatalf("input attachments = %d, want 1", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(chatModel)
	if got := string(m.input); got != "describe " {
		t.Fatalf("input after backspace = %q, want image placeholder removed", got)
	}
	if got := len(m.inputAttachments); got != 0 {
		t.Fatalf("input attachments after editing placeholder = %d, want 0", got)
	}

	m.input = []rune("describe")
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("submit should start a run")
	}
	if got := len(m.liveMessages[0].Images); got != 0 {
		t.Fatalf("images = %d, want 0 after deleting placeholder", got)
	}
	if got := m.liveMessages[0].Content; got != "describe" {
		t.Fatalf("content = %q, want describe", got)
	}
}

func TestChatWordDeletingImagePlaceholderRemovesAttachment(t *testing.T) {
	fp := writeTestPNG(t)
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Model:      "test",
			Client:     chatTestClient{},
			MultiModal: true,
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("describe " + fp), Paste: true})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace, Alt: true})
	m = updated.(chatModel)

	if got := string(m.input); got != "describe " {
		t.Fatalf("input after word backspace = %q, want image placeholder removed", got)
	}
	if got := len(m.inputAttachments); got != 0 {
		t.Fatalf("input attachments after word backspace = %d, want 0", got)
	}
}

func writeTestPNG(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	fp := filepath.Join(dir, "dragged image.png")
	data := make([]byte, 600)
	copy(data, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	if err := os.WriteFile(fp, data, 0o600); err != nil {
		t.Fatalf("failed to write test image: %v", err)
	}
	return fp
}
