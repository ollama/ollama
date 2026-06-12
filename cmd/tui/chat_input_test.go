package tui

import (
	"context"

	"os"
	"path/filepath"

	"slices"

	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"

	agentskills "github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
)

func TestChatToolsCommandListsTools(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})

	m := chatModel{
		opts:  ChatOptions{Tools: registry},
		input: []rune("/tools"),
	}
	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("tools command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(fm.entries))
	}
	if !strings.Contains(fm.entries[0].content, "- **fake_tool**: does test work") {
		t.Fatalf("tools output = %q", fm.entries[0].content)
	}
}

func TestChatHelpCommandShowsCommands(t *testing.T) {
	m := chatModel{input: []rune("/help")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("help command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(fm.entries))
	}
	if !strings.Contains(fm.entries[0].content, "**Commands**") ||
		!strings.Contains(fm.entries[0].content, "- `/tools`: show available tools") ||
		!strings.Contains(fm.entries[0].content, "- `/model`: switch models") ||
		!strings.Contains(fm.entries[0].content, "- `/history`: show prompt message history") ||
		!strings.Contains(fm.entries[0].content, "- `/<skill>`: run the next message with a skill") ||
		!strings.Contains(fm.entries[0].content, "**Shortcuts**") ||
		!strings.Contains(fm.entries[0].content, "- `ctrl+o`: toggle tool output and details") ||
		!strings.Contains(fm.entries[0].content, "- `shift+tab`: toggle permission mode") {
		t.Fatalf("help output = %q", fm.entries[0].content)
	}
}

func TestChatInputAcceptsSpace(t *testing.T) {
	m := chatModel{}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("hello")})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace, Runes: []rune(" ")})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("world")})
	m = updated.(chatModel)

	if got := string(m.input); got != "hello world" {
		t.Fatalf("input = %q, want hello world", got)
	}
}

func TestChatInputAcceptsTextWhileRunning(t *testing.T) {
	m := chatModel{running: true}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("next")})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("prompt")})
	m = updated.(chatModel)

	if got := string(m.input); got != "next prompt" {
		t.Fatalf("input = %q, want queued draft text", got)
	}
}

func TestChatPromptHistoryNavigatesPreviousPrompts(t *testing.T) {
	m := chatModel{
		input:         []rune("draft"),
		promptHistory: []string{"first prompt", "second prompt"},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if got := string(m.input); got != "second prompt" {
		t.Fatalf("input after first up = %q, want second prompt", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if got := string(m.input); got != "first prompt" {
		t.Fatalf("input after second up = %q, want first prompt", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(chatModel)
	if got := string(m.input); got != "second prompt" {
		t.Fatalf("input after down = %q, want second prompt", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(chatModel)
	if got := string(m.input); got != "draft" {
		t.Fatalf("input after returning to draft = %q, want draft", got)
	}
	if m.promptActive {
		t.Fatal("prompt history should be inactive after returning to draft")
	}
}

func TestChatPromptHistoryEditsRecalledPrompt(t *testing.T) {
	m := chatModel{promptHistory: []string{"previous"}}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(" edited")})
	m = updated.(chatModel)

	if got := string(m.input); got != "previous edited" {
		t.Fatalf("edited input = %q, want previous edited", got)
	}
	if m.promptActive {
		t.Fatal("editing recalled input should leave prompt history navigation")
	}
}

func TestInitialPromptHistoryLoadsFromStore(t *testing.T) {
	store := &chatResumeTestStore{prompts: []string{"old prompt", "new prompt"}}

	history := initialPromptHistory(context.Background(), ChatOptions{
		Store:    store,
		Messages: []api.Message{{Role: "user", Content: "fallback prompt"}},
	})

	if !slices.Equal(history, []string{"old prompt", "new prompt"}) {
		t.Fatalf("history = %#v, want store prompts", history)
	}
}

func TestChatViewRendersSlashCommandSuggestions(t *testing.T) {
	m := chatModel{
		input:  []rune("/"),
		width:  80,
		height: 18,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "/clear") || !strings.Contains(view, "clear this chat") {
		t.Fatalf("view missing clear suggestion: %q", view)
	}
	if !strings.Contains(view, "/tools") || !strings.Contains(view, "show available tools") {
		t.Fatalf("view missing tools suggestion: %q", view)
	}
	if !strings.Contains(view, "/model") || !strings.Contains(view, "switch models") {
		t.Fatalf("view missing model suggestion: %q", view)
	}
	if !strings.Contains(view, "/history") || !strings.Contains(view, "show prompt message history") {
		t.Fatalf("view missing history suggestion: %q", view)
	}
	if strings.Contains(view, "/new") || strings.Contains(view, "/resume") {
		t.Fatalf("bare slash should show only top suggestions: %q", view)
	}
	if got := len(m.slashCommandLines(80)); got != maxSlashCompletions {
		t.Fatalf("slash suggestions = %d, want %d", got, maxSlashCompletions)
	}
	if !strings.Contains(view, "> /█") {
		t.Fatalf("view missing slash input row: %q", view)
	}
}

func TestChatSlashCommandSuggestionsScroll(t *testing.T) {
	m := chatModel{
		input: []rune("/"),
	}

	if got := len(m.slashCompletions()); got <= maxSlashCompletions {
		t.Fatalf("test setup needs more than %d slash completions, got %d", maxSlashCompletions, got)
	}

	for i := 0; i < 6; i++ {
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
		m = updated.(chatModel)
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if got := len(m.slashCommandLines(80)); got != maxSlashCompletions {
		t.Fatalf("rendered slash suggestions = %d, want %d", got, maxSlashCompletions)
	}
	if !strings.Contains(lines, "/resume") {
		t.Fatalf("scrolled suggestions missing later command: %q", lines)
	}
	if strings.Contains(lines, "/clear") {
		t.Fatalf("scrolled suggestions should not include first command: %q", lines)
	}
	if selected, ok := m.selectedSlashCommand(); !ok || selected != "/resume" {
		t.Fatalf("selected slash command = %q, %v; want /resume, true", selected, ok)
	}
}

func TestChatSlashCommandSuggestionsFilter(t *testing.T) {
	m := chatModel{
		input: []rune("/to"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/tools") {
		t.Fatalf("suggestions missing /tools: %q", lines)
	}
	if strings.Contains(lines, "/clear") {
		t.Fatalf("suggestions should filter out /clear: %q", lines)
	}
}

func TestChatEnterAcceptsSelectedSlashCommand(t *testing.T) {
	m := chatModel{
		input:    []rune("/"),
		complete: 1, // /tools
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("slash command should not return a command")
	}
	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(m.entries))
	}
	if m.entries[0].role == "error" || strings.Contains(m.entries[0].content, "Unknown command") {
		t.Fatalf("selected slash command should run instead of submitting slash: %#v", m.entries[0])
	}
	if !strings.Contains(m.entries[0].content, "No tools are available") {
		t.Fatalf("entry content = %q, want tools command output", m.entries[0].content)
	}
}

func TestChatSkillSlashCompletionAndTrigger(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "go-code")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, agentskills.SkillFile), []byte("---\nname: go-code\ndescription: Write idiomatic Go code.\n---\n\n# Go Code\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agentskills.Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	m := chatModel{
		ctx:   context.Background(),
		input: []rune("/go"),
		opts: ChatOptions{
			Model:  "test",
			Client: &chatCaptureClient{},
			Skills: catalog,
		},
	}
	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/go-code") || !strings.Contains(lines, "Write idiomatic Go code") {
		t.Fatalf("skill completion missing: %q", lines)
	}

	m.input = []rune("/go-code write a test")
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("skill trigger should start a run")
	}
	if len(m.entries) == 0 || m.entries[0].content != "/go-code write a test" {
		t.Fatalf("displayed entries = %#v", m.entries)
	}
	runDone := waitForRunDone(t, m.events)
	if runDone.err != nil {
		t.Fatal(runDone.err)
	}
	client := m.opts.Client.(*chatCaptureClient)
	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	reqMessages := client.requests[0].Messages
	if len(reqMessages) < 2 || reqMessages[0].Role != "system" || !strings.Contains(reqMessages[0].Content, "# Go Code") {
		t.Fatalf("request messages = %#v", reqMessages)
	}
	if got := reqMessages[1].Content; !strings.Contains(got, "Use the go-code skill") || !strings.Contains(got, "write a test") {
		t.Fatalf("user prompt = %q", got)
	}
}

func TestChatSkillsImportCommand(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	src := filepath.Join(home, ".claude", "skills", "go-code")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: go-code\ndescription: Write Go code.\n---\n\n# Go Code\n"
	if err := os.WriteFile(filepath.Join(src, agentskills.SkillFile), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	registry := coreagent.NewRegistry()
	m := chatModel{
		input: []rune("/skills import claude"),
		opts: ChatOptions{
			Tools: registry,
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("skills import should not start a model run")
	}
	if len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "imported go-code") {
		t.Fatalf("entries = %#v", m.entries)
	}
	if m.opts.Skills == nil || !m.opts.Tools.Has("skill") {
		t.Fatalf("skills/tool registry not updated: skills=%#v tools=%v", m.opts.Skills, m.opts.Tools.Names())
	}
	if _, err := os.Stat(filepath.Join(home, ".ollama", "skills", "go-code", agentskills.SkillFile)); err != nil {
		t.Fatal(err)
	}
}

func TestChatViewRendersFileMentionSuggestions(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := chatModel{
		input:  []rune("check @"),
		width:  80,
		height: 16,
		opts:   ChatOptions{WorkingDir: dir},
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "@cmd/") || !strings.Contains(view, "directory") {
		t.Fatalf("view missing directory suggestion: %q", view)
	}
	if !strings.Contains(view, "@README.md") || !strings.Contains(view, "file") {
		t.Fatalf("view missing file suggestion: %q", view)
	}
}

func TestChatFileMentionSuggestionsFilterAndComplete(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "runner.go"), []byte("package main"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := chatModel{
		input: []rune("read @REA"),
		opts:  ChatOptions{WorkingDir: dir},
	}

	lines := stripANSI(strings.Join(m.completionLines(80), "\n"))
	if !strings.Contains(lines, "@README.md") {
		t.Fatalf("suggestions missing README.md: %q", lines)
	}
	if strings.Contains(lines, "@runner.go") {
		t.Fatalf("suggestions should filter out runner.go: %q", lines)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = updated.(chatModel)
	if got := string(m.input); got != "read @README.md " {
		t.Fatalf("input = %q, want completed file mention", got)
	}
}
