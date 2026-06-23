package tui

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"

	agentskills "github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
)

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
		!strings.Contains(fm.entries[0].content, "- `/model`: switch models") ||
		!strings.Contains(fm.entries[0].content, "- `/think`: set thinking mode") ||
		!strings.Contains(fm.entries[0].content, "- `/verbose`: toggle model metrics") ||
		!strings.Contains(fm.entries[0].content, "- `/<skill>`: run the next message with a skill") ||
		!strings.Contains(fm.entries[0].content, "- `/help`: show commands") ||
		!strings.Contains(fm.entries[0].content, "- `/bye`: exit") ||
		!strings.Contains(fm.entries[0].content, "**Shortcuts**") ||
		!strings.Contains(fm.entries[0].content, "- `ctrl+o`: toggle tool output") ||
		!strings.Contains(fm.entries[0].content, "- `shift+enter`: insert a newline") ||
		!strings.Contains(fm.entries[0].content, "- `shift+tab`: toggle permission mode") ||
		!strings.Contains(fm.entries[0].content, "- `ctrl+a/e`: move to line start or end") {
		t.Fatalf("help output = %q", fm.entries[0].content)
	}
	for _, hidden := range []string{"/history", "/set think", "/set nothink"} {
		if strings.Contains(fm.entries[0].content, hidden) {
			t.Fatalf("hidden command %q should stay hidden from help: %q", hidden, fm.entries[0].content)
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
	if !strings.Contains(rendered, "...") {
		t.Fatalf("input should include continuation marker: %q", rendered)
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
		opts: ChatOptions{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(pasted), Paste: true})
	m = updated.(chatModel)
	if got, want := string(m.input), "[Pasted text #1 +8 lines]"; got != want {
		t.Fatalf("input = %q, want %q", got, want)
	}
	if got := len(m.inputPastedTexts); got != 1 {
		t.Fatalf("pasted texts = %d, want 1", got)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("submit should start a run")
	}
	if got := m.entries[0].content; got != "[Pasted text #1 +8 lines]" {
		t.Fatalf("display content = %q, want placeholder", got)
	}
	if got := m.liveMessages[0].Content; got != pasted {
		t.Fatalf("model content = %q, want pasted text", got)
	}
	if len(m.promptHistory) != 1 || m.promptHistory[0] != pasted {
		t.Fatalf("prompt history = %#v, want expanded pasted text", m.promptHistory)
	}

	m.running = false
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if got := string(m.input); got != "[Pasted text #2 +8 lines]" {
		t.Fatalf("recalled input = %q, want pasted text placeholder", got)
	}
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if got := m.liveMessages[0].Content; got != pasted {
		t.Fatalf("recalled model content = %q, want pasted text", got)
	}
}

func TestChatBackspaceDeletesWholePastedTextPlaceholder(t *testing.T) {
	pasted := strings.Repeat("long paste line\n", pastedTextPlaceholderMinLines)
	m := chatModel{}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(pasted), Paste: true})
	m = updated.(chatModel)
	if got := len(m.inputPastedTexts); got != 1 {
		t.Fatalf("pasted texts = %d, want 1", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(chatModel)
	if got := string(m.input); got != "" {
		t.Fatalf("input after backspace = %q, want empty", got)
	}
	if got := len(m.inputPastedTexts); got != 0 {
		t.Fatalf("pasted texts after backspace = %d, want 0", got)
	}
}

func TestChatWordBackspaceDeletesWholePastedTextPlaceholder(t *testing.T) {
	pasted := strings.Repeat("long paste line\n", pastedTextPlaceholderMinLines)
	m := chatModel{input: []rune("use ")}
	m.inputCursor = len(m.input)
	m.inputCursorSet = true

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(pasted), Paste: true})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace, Alt: true})
	m = updated.(chatModel)

	if got := string(m.input); got != "use " {
		t.Fatalf("input after word backspace = %q, want pasted text placeholder removed", got)
	}
	if got := len(m.inputPastedTexts); got != 0 {
		t.Fatalf("pasted texts after word backspace = %d, want 0", got)
	}
}

func TestChatEditorResultReplacesInput(t *testing.T) {
	m := chatModel{
		input:            []rune("old"),
		inputCursor:      3,
		inputCursorSet:   true,
		inputPastedTexts: []chatInputPastedText{{placeholder: "[Pasted text #1 +8 lines]", content: "hidden"}},
	}

	m.applyEditorResult(chatEditorDoneMsg{content: "new prompt"})
	if got := string(m.input); got != "new prompt" {
		t.Fatalf("input = %q, want new prompt", got)
	}
	if got := m.inputCursor; got != len("new prompt") {
		t.Fatalf("cursor = %d, want %d", got, len("new prompt"))
	}
	if got := len(m.inputPastedTexts); got != 0 {
		t.Fatalf("pasted texts = %d, want 0 after editor removed placeholder", got)
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

func TestChatInputAltEnterInsertsNewline(t *testing.T) {
	m := chatModel{input: []rune("line one")}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter, Alt: true})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("alt+enter should not submit")
	}
	if got := string(m.input); got != "line one\n" {
		t.Fatalf("input = %q, want newline appended", got)
	}
}

func TestChatInputCtrlJInsertsNewline(t *testing.T) {
	m := chatModel{input: []rune("line one")}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlJ})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("ctrl+j should not submit")
	}
	if got := string(m.input); got != "line one\n" {
		t.Fatalf("input = %q, want newline appended", got)
	}
}

func TestChatInputOptionBackspaceDeletesPreviousWord(t *testing.T) {
	m := chatModel{input: []rune("alpha beta   ")}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyBackspace, Alt: true})
	m = updated.(chatModel)
	if got := string(m.input); got != "alpha " {
		t.Fatalf("input = %q, want alpha with trailing space", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace, Alt: true})
	m = updated.(chatModel)
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want empty", got)
	}
}

func TestChatInputCtrlWDeletesPreviousWord(t *testing.T) {
	m := chatModel{input: []rune("alpha beta")}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlW})
	m = updated.(chatModel)
	if got := string(m.input); got != "alpha " {
		t.Fatalf("input = %q, want alpha with trailing space", got)
	}
}

func TestChatInputCtrlUClearsPrompt(t *testing.T) {
	m := chatModel{input: []rune("alpha beta")}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlU})
	m = updated.(chatModel)
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want empty", got)
	}
}

func TestChatInputArrowKeysEditMiddle(t *testing.T) {
	m := chatModel{input: []rune("hello world")}

	for range 5 {
		updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyLeft})
		m = updated.(chatModel)
	}
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("beautiful ")})
	m = updated.(chatModel)

	if got := string(m.input); got != "hello beautiful world" {
		t.Fatalf("input = %q, want inserted text in middle", got)
	}
	if m.inputCursor != len([]rune("hello beautiful ")) {
		t.Fatalf("cursor = %d, want after inserted text", m.inputCursor)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	m = updated.(chatModel)
	if got := string(m.input); got != "hello beautifulworld" {
		t.Fatalf("input after backspace = %q", got)
	}
}

func TestChatInputUpDownNavigateMultilineBeforeHistory(t *testing.T) {
	m := chatModel{
		input:         []rune("one\ntwo\nthree"),
		promptHistory: []string{"old prompt"},
	}
	m.inputCursor = len([]rune("one\ntwo"))
	m.inputCursorSet = true

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if got := m.inputCursor; got != len([]rune("one")) {
		t.Fatalf("cursor after up = %d, want end of first line", got)
	}
	if got := string(m.input); got != "one\ntwo\nthree" {
		t.Fatalf("input after up = %q, want unchanged", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(chatModel)
	if got := m.inputCursor; got != len([]rune("one\ntwo")) {
		t.Fatalf("cursor after down = %d, want end of second line", got)
	}
	if m.promptActive {
		t.Fatal("multiline cursor navigation should not enter prompt history")
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

func TestChatDeletedSlashCommandsAreUnknown(t *testing.T) {
	for _, command := range []string{"/copy", "/copy-all", "/tools", "/launch", "/system"} {
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
	if !strings.Contains(view, "/clear") || !strings.Contains(view, "clear this chat") {
		t.Fatalf("view missing clear suggestion: %q", view)
	}
	if !strings.Contains(view, "/model") || !strings.Contains(view, "switch models") {
		t.Fatalf("view missing model suggestion: %q", view)
	}
	if strings.Contains(view, "/copy") || strings.Contains(view, "/copy-all") || strings.Contains(view, "/tools") || strings.Contains(view, "/history") {
		t.Fatalf("bare slash should hide utility commands: %q", view)
	}
	if got := len(m.slashCommandLines(80)); got != maxSlashCompletions {
		t.Fatalf("slash suggestions = %d, want %d", got, maxSlashCompletions)
	}
	if !strings.Contains(view, "› /█") {
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

	for range 4 {
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
	if strings.Contains(lines, "/copy") {
		t.Fatalf("scrolled suggestions should not include first command: %q", lines)
	}
	if selected, ok := m.selectedSlashCommand(); !ok || selected != "/resume" {
		t.Fatalf("selected slash command = %q, %v; want /resume, true", selected, ok)
	}
}

func TestChatSlashCommandSuggestionsOmitDeletedToolsCommand(t *testing.T) {
	m := chatModel{
		input: []rune("/to"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if strings.Contains(lines, "/tools") || !strings.Contains(lines, "No matching commands") {
		t.Fatalf("deleted /tools command should not be suggested: %q", lines)
	}
}

func TestChatSlashCommandSuggestionsIncludeVerbose(t *testing.T) {
	m := chatModel{
		input: []rune("/ve"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/verbose") || !strings.Contains(lines, "toggle model metrics") {
		t.Fatalf("suggestions missing /verbose: %q", lines)
	}
}

func TestChatSlashCommandSuggestionsIncludeThink(t *testing.T) {
	m := chatModel{
		input: []rune("/th"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if !strings.Contains(lines, "/think") || !strings.Contains(lines, "set thinking mode") {
		t.Fatalf("suggestions missing /think: %q", lines)
	}
}

func TestChatSlashCommandSuggestionsOmitDeletedSystemCommand(t *testing.T) {
	m := chatModel{
		input: []rune("/sys"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if strings.Contains(lines, "/system") || !strings.Contains(lines, "No matching commands") {
		t.Fatalf("deleted /system command should not be suggested: %q", lines)
	}
}

func TestChatSlashCommandSuggestionsHideLegacySetThink(t *testing.T) {
	m := chatModel{
		input: []rune("/set"),
	}

	lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
	if strings.Contains(lines, "/set think") || strings.Contains(lines, "/set nothink") {
		t.Fatalf("legacy think commands should not be suggested: %q", lines)
	}
	if !strings.Contains(lines, "No matching commands") {
		t.Fatalf("legacy /set should remain hidden from suggestions: %q", lines)
	}
}

func TestChatSlashCommandSuggestionsHideLegacyRunCommands(t *testing.T) {
	for _, input := range []string{"/show", "/load"} {
		t.Run(input, func(t *testing.T) {
			m := chatModel{input: []rune(input)}
			lines := stripANSI(strings.Join(m.slashCommandLines(80), "\n"))
			if strings.Contains(lines, input) || !strings.Contains(lines, "No matching commands") {
				t.Fatalf("legacy command %s should stay hidden from suggestions: %q", input, lines)
			}
		})
	}
}

func TestChatEnterAcceptsSelectedSlashCommand(t *testing.T) {
	m := chatModel{
		input:    []rune("/"),
		complete: 3, // /new
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("slash command should not return a command")
	}
	if len(m.entries) != 0 || len(m.messages) != 0 {
		t.Fatalf("new chat command should reset chat, entries=%d messages=%d", len(m.entries), len(m.messages))
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

func TestChatThinkPickerSelectsMode(t *testing.T) {
	m := chatModel{}
	updated, _ := m.openThinkPicker()
	m = updated.(chatModel)

	for range 5 {
		updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
		m = updated.(chatModel)
	}
	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		t.Fatal("think picker selection should not return a command")
	}
	m = updated.(chatModel)

	if m.thinkPicker != nil {
		t.Fatal("think picker should close after selection")
	}
	if m.opts.Think == nil || m.opts.Think.Value != "high" {
		t.Fatalf("think = %#v, want high", m.opts.Think)
	}
	if m.status != "think high" {
		t.Fatalf("status = %q, want think high", m.status)
	}
}

func TestChatThinkCommandSetsModes(t *testing.T) {
	tests := []struct {
		input string
		want  any
	}{
		{input: "/think auto", want: nil},
		{input: "/think on", want: true},
		{input: "/think off", want: false},
		{input: "/think low", want: "low"},
		{input: "/think medium", want: "medium"},
		{input: "/think high", want: "high"},
		{input: "/think max", want: "max"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}
			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("think command should not return a command")
			}
			m = updated.(chatModel)
			if tt.want == nil {
				if m.opts.Think != nil {
					t.Fatalf("think = %#v, want nil", m.opts.Think)
				}
				return
			}
			if m.opts.Think == nil || m.opts.Think.Value != tt.want {
				t.Fatalf("think = %#v, want %v", m.opts.Think, tt.want)
			}
		})
	}
}

func TestChatLegacySetThinkCommandsStaySupported(t *testing.T) {
	tests := []struct {
		input string
		want  any
	}{
		{input: "/set think", want: true},
		{input: "/set think high", want: "high"},
		{input: "/set nothink", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			m := chatModel{input: []rune(tt.input)}
			updated, cmd := m.handleSubmit()
			if cmd != nil {
				t.Fatal("legacy think command should not return a command")
			}
			m = updated.(chatModel)
			if m.opts.Think == nil || m.opts.Think.Value != tt.want {
				t.Fatalf("think = %#v, want %v", m.opts.Think, tt.want)
			}
		})
	}
}

func TestChatThinkCommandRejectsInvalidMode(t *testing.T) {
	m := chatModel{input: []rune("/think turbo")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("think command should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Usage: /think") {
		t.Fatalf("entries = %#v, want usage error", m.entries)
	}
}

func TestChatVerboseCommandTogglesMetrics(t *testing.T) {
	m := chatModel{input: []rune("/verbose")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("verbose command should not return a command")
	}
	m = updated.(chatModel)
	if !m.opts.Verbose {
		t.Fatal("verbose should be enabled")
	}
	if m.status != "verbose on" {
		t.Fatalf("status = %q, want verbose on", m.status)
	}
	if footer := m.footerLine(); !strings.Contains(footer, "verbose") {
		t.Fatalf("footer should show verbose mode: %q", footer)
	}

	m.input = []rune("/verbose")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("verbose command should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Verbose {
		t.Fatal("verbose should be disabled")
	}
	if m.status != "verbose off" {
		t.Fatalf("status = %q, want verbose off", m.status)
	}
}

func TestChatVerboseCommandAcceptsExplicitState(t *testing.T) {
	m := chatModel{input: []rune("/verbose on")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("verbose command should not return a command")
	}
	m = updated.(chatModel)
	if !m.opts.Verbose {
		t.Fatal("verbose should be enabled")
	}

	m.input = []rune("/verbose off")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("verbose command should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Verbose {
		t.Fatal("verbose should be disabled")
	}
}

func TestChatVerboseCommandRejectsInvalidState(t *testing.T) {
	m := chatModel{input: []rune("/verbose maybe")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("verbose command should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Usage: /verbose") {
		t.Fatalf("entries = %#v, want usage error", m.entries)
	}
}

func TestChatLegacySetVerboseQuietCommandsStaySupported(t *testing.T) {
	m := chatModel{input: []rune("/set verbose")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set verbose should not return a command")
	}
	m = updated.(chatModel)
	if !m.opts.Verbose {
		t.Fatal("verbose should be enabled")
	}

	m.input = []rune("/set quiet")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set quiet should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Verbose {
		t.Fatal("verbose should be disabled")
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
	m := chatModel{input: []rune("/set parameter temperature 0.2")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set parameter should not return a command")
	}
	m = updated.(chatModel)
	got, ok := m.opts.Options["temperature"].(float32)
	if !ok || got != 0.2 {
		t.Fatalf("temperature = %#v, want float32(0.2)", m.opts.Options["temperature"])
	}
}

func TestChatLegacySetHistoryStaysUnsupported(t *testing.T) {
	m := chatModel{input: []rune("/set history")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy set history should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || m.entries[0].role != "error" || !strings.Contains(m.entries[0].content, "Unknown command `/set history`") {
		t.Fatalf("entries = %#v, want unsupported history error", m.entries)
	}
}

func TestChatLegacyLoadCommandSwitchesModel(t *testing.T) {
	var savedModel string
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("/load qwen3"),
		opts: ChatOptions{
			Model: "llama3.2",
			OnModelSelected: func(_ context.Context, model string) error {
				savedModel = model
				return nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy load should not return a command")
	}
	m = updated.(chatModel)
	if m.opts.Model != "qwen3" || savedModel != "qwen3" {
		t.Fatalf("model = %q saved = %q, want qwen3", m.opts.Model, savedModel)
	}
}

func TestChatLegacyShowCommandRendersModelInfo(t *testing.T) {
	client := &chatShowTestClient{resp: &api.ShowResponse{
		License:    "MIT",
		Modelfile:  "FROM llama3.2",
		Parameters: "temperature 0.7",
		System:     "model system",
		Template:   "{{ .Prompt }}",
		Details: api.ModelDetails{
			Family:            "llama",
			ParameterSize:     "3B",
			QuantizationLevel: "Q4_K_M",
			ContextLength:     8192,
		},
	}}
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("/show info"),
		opts: ChatOptions{
			Model:   "llama3.2",
			Client:  client,
			Options: map[string]any{"top_k": int64(10)},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy show should not return a command")
	}
	m = updated.(chatModel)
	if client.req == nil || client.req.Model != "llama3.2" || client.req.Options["top_k"] != int64(10) {
		t.Fatalf("show request = %#v", client.req)
	}
	if len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "Model info") || !strings.Contains(m.entries[0].content, "context length") {
		t.Fatalf("show info entry = %#v", m.entries)
	}

	m.input = []rune("/show parameters")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy show parameters should not return a command")
	}
	m = updated.(chatModel)
	if !strings.Contains(m.entries[len(m.entries)-1].content, "temperature 0.7") ||
		!strings.Contains(m.entries[len(m.entries)-1].content, "`top_k`: `10`") {
		t.Fatalf("show parameters entry = %q", m.entries[len(m.entries)-1].content)
	}
}

func TestChatLegacyHelpCommandsStaySupported(t *testing.T) {
	m := chatModel{input: []rune("/help set")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy help should not return a command")
	}
	m = updated.(chatModel)
	if len(m.entries) != 1 || !strings.Contains(m.entries[0].content, "Legacy set commands") {
		t.Fatalf("help set entry = %#v", m.entries)
	}

	m.input = []rune("/? show")
	updated, cmd = m.handleSubmit()
	if cmd != nil {
		t.Fatal("legacy show help should not return a command")
	}
	m = updated.(chatModel)
	if !strings.Contains(m.entries[len(m.entries)-1].content, "Legacy show commands") {
		t.Fatalf("help show entry = %#v", m.entries[len(m.entries)-1])
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
	if len(m.promptHistory) != 1 || m.promptHistory[0] != "/go-code write a test" {
		t.Fatalf("prompt history = %#v, want slash skill command preserved", m.promptHistory)
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
	if len(reqMessages) < 3 {
		t.Fatalf("request messages = %#v", reqMessages)
	}
	if reqMessages[0].Role == "system" {
		if strings.Contains(reqMessages[0].Content, "# Go Code") {
			t.Fatalf("system prompt should contain skill metadata only: %#v", reqMessages[0])
		}
		reqMessages = reqMessages[1:]
	}
	if got := reqMessages[0].Content; got != "write a test" {
		t.Fatalf("user prompt = %q", got)
	}
	if reqMessages[1].Role != "assistant" || len(reqMessages[1].ToolCalls) != 1 || reqMessages[1].ToolCalls[0].Function.Name != "skill" {
		t.Fatalf("manual skill should be a synthetic assistant tool call: %#v", reqMessages[1])
	}
	if reqMessages[2].Role != "tool" || reqMessages[2].ToolName != "skill" || !strings.Contains(reqMessages[2].Content, "# Go Code") {
		t.Fatalf("manual skill should include skill tool result: %#v", reqMessages[2])
	}
}

func TestChatSkillsImportCommand(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	src := filepath.Join(home, ".claude", "skills", "go-code")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: go-code\ndescription: Write Go code.\n---\n\n# Go Code\n"
	if err := os.WriteFile(filepath.Join(src, agentskills.SkillFile), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	registry := coreagent.NewRegistry()
	catalog := &agentskills.Catalog{}
	m := chatModel{
		input: []rune("/skills import claude"),
		opts: ChatOptions{
			Tools:  registry,
			Skills: catalog,
			SystemPromptForModel: func(context.Context, string, *coreagent.Registry) string {
				return "default Ollama prompt\n\n" + catalog.SystemPrompt(registry.Has("skill"))
			},
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
	if !strings.Contains(m.opts.SystemPrompt, "default Ollama prompt") || !strings.Contains(m.opts.SystemPrompt, "go-code: Write Go code.") {
		t.Fatalf("system prompt not rebuilt with default prompt and skills: %q", m.opts.SystemPrompt)
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
