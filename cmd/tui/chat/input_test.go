package chat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ollama/ollama/api"
)

func TestChatHelpCommandShowsV1Commands(t *testing.T) {
	m := chatModel{input: []rune("/help")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("help command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(fm.entries))
	}
	for _, want := range []string{
		"**Commands**",
		"- `/model`: switch models",
		"- `/think`: set thinking mode",
		"- `/compact`: summarize older context",
		"- `/help`: show commands",
		"- `/bye`: exit",
		"**Shortcuts**",
		"- `shift+enter`: insert a newline",
		"- `shift+tab`: toggle permission mode",
	} {
		if !strings.Contains(fm.entries[0].content, want) {
			t.Fatalf("help output missing %q:\n%s", want, fm.entries[0].content)
		}
	}
	for _, removed := range []string{"/history", "/raw", "/resume", "/skills", "/verbose"} {
		if strings.Contains(fm.entries[0].content, removed) {
			t.Fatalf("removed command %q should stay hidden from help:\n%s", removed, fm.entries[0].content)
		}
	}
}

func TestTruncateInputLineUsesDisplayWidth(t *testing.T) {
	line := truncateInputLine(strings.Repeat("界", 10), 10)
	if got := lipgloss.Width(line); got > 10 {
		t.Fatalf("line %q width = %d, want <= 10", line, got)
	}
}

func TestRenderInputBoxTruncationUsesSingleContinuationMarker(t *testing.T) {
	lines := renderInputBoxLines("one two three four five six seven", len("one two three four five six seven"), 16, 1, "")
	rendered := strings.Join(lines, "\n")
	if strings.Contains(rendered, "... ...") {
		t.Fatalf("input rendered duplicate continuation marker: %q", rendered)
	}
	if strings.Contains(rendered, "one two") {
		t.Fatalf("input should keep the latest truncated line: %q", rendered)
	}
}

type shiftEnterCSITestMsg string

func (m shiftEnterCSITestMsg) String() string {
	return string(m)
}

func TestChatInputHandlesShiftEnterCSIMessage(t *testing.T) {
	m := chatModel{input: []rune("line one")}

	updated, _ := m.Update(shiftEnterCSITestMsg("?CSI[49 51 59 50 117]?"))
	m = updated.(chatModel)
	if got := string(m.input); got != "line one\n" {
		t.Fatalf("input = %q, want newline inserted", got)
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

func TestChatLargePasteUsesPlaceholderAndExpandsOnSubmit(t *testing.T) {
	pasted := strings.Repeat("line\n", pastedTextPlaceholderMinLines-1) + "line"
	m := chatModel{
		ctx: context.Background(),
		opts: Options{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(pasted), Paste: true})
	m = updated.(chatModel)
	if got, want := string(m.input), "[Pasted text #1 +8 lines]"; got != want {
		t.Fatalf("input = %q, want %q", got, want)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("enter should start a run")
	}
	done := waitForRunDone(t, m.events)
	if done.err != nil {
		t.Fatal(done.err)
	}
	if done.result == nil || len(done.result.Messages) < 1 || done.result.Messages[0].Content != pasted {
		t.Fatalf("messages = %#v, want expanded pasted text", done.result)
	}
}

func TestChatBackspaceDeletesWholePastedTextPlaceholder(t *testing.T) {
	m := chatModel{
		input: []rune("use [Pasted text #1 +8 lines]"),
		inputPastedTexts: []chatInputPastedText{{
			placeholder: "[Pasted text #1 +8 lines]",
			content:     "hidden",
		}},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(chatModel)
	if got := string(m.input); got != "use " {
		t.Fatalf("input after backspace = %q, want pasted text placeholder removed", got)
	}
	if got := len(m.inputPastedTexts); got != 0 {
		t.Fatalf("pasted texts after backspace = %d, want 0", got)
	}
}

func TestInitialPromptHistoryLoadsFromMessages(t *testing.T) {
	history := initialPromptHistory(context.Background(), Options{
		Messages: []api.Message{
			{Role: "user", Content: "old prompt"},
			{Role: "assistant", Content: "answer"},
			{Role: "user", Content: "new prompt"},
		},
	})

	if got, want := strings.Join(history, "|"), "old prompt|new prompt"; got != want {
		t.Fatalf("history = %#v, want %s", history, want)
	}
}

func TestChatDeletedSlashCommandsAreUnknown(t *testing.T) {
	for _, command := range []string{"/copy", "/copy-all", "/tools", "/launch", "/system", "/history", "/raw", "/resume", "/skills", "/verbose"} {
		t.Run(command, func(t *testing.T) {
			m := chatModel{input: []rune(command)}

			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("deleted slash command should not return a command")
			}
			m = updated.(chatModel)
			if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Unknown command") {
				t.Fatalf("entries = %#v, want unknown command error", m.entries)
			}
		})
	}
}

func TestChatViewRendersSlashCommandSuggestions(t *testing.T) {
	m := chatModel{
		input:  []rune("/"),
		width:  80,
		height: 18,
	}

	view := stripANSI(m.View())
	for _, want := range []string{"/clear", "/model", "/new", "/think", "/compact"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %s suggestion: %q", want, view)
		}
	}
	for _, removed := range []string{"/copy", "/copy-all", "/tools", "/history", "/raw", "/resume", "/skills", "/verbose"} {
		if strings.Contains(view, removed) {
			t.Fatalf("bare slash should hide removed command %s: %q", removed, view)
		}
	}
	if got := len(m.slashCommandLines(80)); got != maxSlashCompletions {
		t.Fatalf("slash suggestions = %d, want %d", got, maxSlashCompletions)
	}
}

func TestChatSlashCommandSuggestionsIncludeThink(t *testing.T) {
	m := chatModel{input: []rune("/th")}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/think") || !strings.Contains(lines, "set thinking mode") {
		t.Fatalf("suggestions missing /think: %q", lines)
	}
}

func TestChatEnterAcceptsSelectedSlashCommand(t *testing.T) {
	m := chatModel{input: []rune("/th")}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("think command should not return a command")
	}
	if m.thinkPicker == nil {
		t.Fatal("selected /think command should open picker")
	}
}

func TestChatSlashCommandsDoNotQueueWhileRunning(t *testing.T) {
	m := chatModel{running: true, input: []rune("/help")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("help command should not return a command")
	}
	m = updated.(chatModel)
	if len(m.queued) != 0 {
		t.Fatalf("slash command queued while running: %#v", m.queued)
	}
	if len(m.entries) != 1 || m.entries[0].role != "slash" {
		t.Fatalf("entries = %#v, want immediate slash output", m.entries)
	}
}

func TestChatThinkCommandOpensPicker(t *testing.T) {
	m := chatModel{input: []rune("/think")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("think command should not return a command")
	}
	m = updated.(chatModel)
	if m.thinkPicker == nil {
		t.Fatal("think picker should open")
	}
	if view := stripANSI(m.renderThinkPicker(80)); !strings.Contains(view, "Thinking mode") || !strings.Contains(view, "high") {
		t.Fatalf("think picker view missing options: %q", view)
	}
}

func TestChatThinkCommandSetsModes(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  any
	}{
		{input: "/think on", want: true},
		{input: "/think off", want: false},
		{input: "/think high", want: "high"},
	} {
		t.Run(tt.input, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}
			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("think command should not return a command")
			}
			m = updated.(chatModel)
			if m.opts.Think == nil || m.opts.Think.Value != tt.want {
				t.Fatalf("think = %#v, want %#v", m.opts.Think, tt.want)
			}
		})
	}
}

func TestChatLegacySetFormatCommandsStaySupported(t *testing.T) {
	m := chatModel{input: []rune("/set format json")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set format should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Format != "json" {
		t.Fatalf("format = %q, want json", m.opts.Format)
	}

	m.input = []rune("/set noformat")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set noformat should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Format != "" {
		t.Fatalf("format = %q, want empty", m.opts.Format)
	}
}

func TestChatLegacySetParameterCommandStaySupported(t *testing.T) {
	m := chatModel{input: []rune("/set parameter temperature 0.7")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set parameter should not return a command")
	}
	m = updated.(chatModel)
	if got := m.opts.Options["temperature"]; got != float32(0.7) {
		t.Fatalf("temperature option = %#v, want 0.7", got)
	}
}

func TestChatLegacySetHistoryStaysUnsupported(t *testing.T) {
	m := chatModel{input: []rune("/set history")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set history should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Unknown command") {
		t.Fatalf("entries = %#v, want unsupported error", m.entries)
	}
}

func TestChatLegacyLoadCommandSwitchesModel(t *testing.T) {
	m := chatModel{
		input: []rune("/load qwen3"),
		opts: Options{
			Model: "llama3.2",
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy load should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Model != "qwen3" {
		t.Fatalf("model = %q, want qwen3", m.opts.Model)
	}
}

func TestChatLegacyShowCommandRendersModelInfo(t *testing.T) {
	client := &chatShowTestClient{resp: &api.ShowResponse{
		Details: api.ModelDetails{Family: "llama", ParameterSize: "8B"},
	}}
	m := chatModel{
		input: []rune("/show info"),
		opts:  Options{Client: client},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy show should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "family") || !strings.Contains(m.entries[0].content, "llama") {
		t.Fatalf("show entry = %#v", m.entries)
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
		workingDir: dir,
		input:      []rune("open @"),
		width:      80,
		height:     18,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "@cmd/") || !strings.Contains(view, "@README.md") {
		t.Fatalf("file mention suggestions missing: %q", view)
	}
}

func TestChatFileMentionSuggestionsFilterAndComplete(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "cmd"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := chatModel{
		workingDir: dir,
		input:      []rune("open @REA"),
	}

	lines := stripANSI(strings.Join(m.completionLines(80), "\n"))
	if !strings.Contains(lines, "@README.md") || strings.Contains(lines, "@cmd/") {
		t.Fatalf("filtered file suggestions = %q", lines)
	}
	m.applyCompletion()
	if got := string(m.input); got != "open @README.md " {
		t.Fatalf("completed input = %q", got)
	}
}
