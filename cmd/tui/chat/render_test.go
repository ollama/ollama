package chat

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestChatAssistantEntryHasNoLabel(t *testing.T) {
	m := chatModel{}

	prefix, _ := m.renderEntry(chatEntry{role: "assistant", content: "hello"})

	if strings.Contains(prefix, "Ollama:") {
		t.Fatalf("prefix should not include Ollama label: %q", prefix)
	}
	if prefix != "" {
		t.Fatalf("prefix = %q, want empty", prefix)
	}
}

func TestChatViewRendersEmptyPromptHint(t *testing.T) {
	m := chatModel{
		chatID: "chat-a",
		width:  80,
		height: 12,
	}

	view := stripANSI(m.View())
	lines := strings.Split(view, "\n")
	hintLine := lineIndexContaining(lines, "what changed on this branch?")
	if hintLine < 0 {
		t.Fatalf("empty chat view missing prompt hint: %q", view)
	}
	if strings.Contains(view, "Start a conversation. Use /help for commands.") {
		t.Fatalf("empty chat view should use rotating prompt hint: %q", view)
	}
}

func TestChatUserEntryHasNoLabel(t *testing.T) {
	m := chatModel{entries: []chatEntry{{role: "user", content: "hello"}}}

	prefix, body := m.renderEntry(m.entries[0])

	if prefix != "" {
		t.Fatalf("prefix = %q, want empty", prefix)
	}
	if body != "hello" {
		t.Fatalf("body = %q, want hello", body)
	}

	transcript := stripANSI(m.renderTranscript(80))
	if !strings.Contains(transcript, "  > hello") {
		t.Fatalf("user transcript should render as user block: %q", transcript)
	}
}

func TestChatSystemEntryHasNoLabel(t *testing.T) {
	m := chatModel{entries: []chatEntry{{role: "system", content: "Available tools"}}}

	prefix, body := m.renderEntry(m.entries[0])

	if prefix != "" {
		t.Fatalf("prefix = %q, want empty", prefix)
	}
	if body != "Available tools" {
		t.Fatalf("body = %q, want Available tools", body)
	}
	if transcript := stripANSI(m.renderTranscript(80)); strings.Contains(transcript, "sys ") {
		t.Fatalf("system transcript should not render sys prefix: %q", transcript)
	}
}

func TestChatMixedToolGroupUsesMixedPrefix(t *testing.T) {
	m := chatModel{}
	entry := newChatEntry(chatEntry{
		role:   "tool_group",
		label:  "Tool calls (2)",
		status: "error",
		tools: []chatEntry{
			newChatEntry(chatEntry{role: "tool", status: "done"}),
			newChatEntry(chatEntry{role: "tool", status: "error", err: "failed"}),
		},
	})

	prefix, body := m.renderEntry(entry)

	if prefix != chatToolMixedStyle.Render("•")+" " {
		t.Fatalf("prefix = %q, want mixed-styled bullet", prefix)
	}
	body = stripANSI(body)
	if strings.Contains(body, "failed") || strings.Contains(body, "succeeded") || strings.Contains(body, "done") {
		t.Fatalf("body = %q, should not show status words for mixed group", body)
	}
}

func TestChatToolStatusLineDoesNotUseDisclosureGlyph(t *testing.T) {
	startedAt := time.Date(2026, 6, 22, 13, 0, 0, 0, time.UTC)
	entry := newChatEntry(chatEntry{
		role:       "tool",
		label:      `Web Search("who is parth sareen")`,
		status:     "done",
		content:    "hidden result",
		startedAt:  startedAt,
		finishedAt: startedAt.Add(812 * time.Millisecond),
	})

	line := stripANSI(toolStatusLine(entry))
	if strings.Contains(line, "▸") || strings.Contains(line, "▾") {
		t.Fatalf("tool status line should not include disclosure glyph: %q", line)
	}
	if line != `Web Search("who is parth sareen")` {
		t.Fatalf("tool status line = %q, want label only", line)
	}
	for _, word := range []string{"done", "in 812ms", "812ms"} {
		if strings.Contains(line, word) {
			t.Fatalf("tool status line should not include %q: %q", word, line)
		}
	}
}

func TestChatCompletedToolStatusLineUsesResultStyle(t *testing.T) {
	entry := newChatEntry(chatEntry{
		role:   "tool",
		detail: "bash",
		label:  `Bash("pwd")`,
		status: "done",
	})

	if line := toolStatusLine(entry); line != `Bash("pwd")` {
		t.Fatalf("tool status line = %q, want command label", line)
	}
	prefix, _ := (chatModel{}).renderEntry(entry)
	if prefix != chatToolDoneStyle.Render("•")+" " {
		t.Fatalf("tool prefix = %q, want done-styled marker", prefix)
	}
}

func TestChatToolStatusMarkerUsesStateColors(t *testing.T) {
	if got, want := chatToolRunningStyle.GetForeground(), lipgloss.Color(chatAnsiYellow); got != want {
		t.Fatalf("running tool foreground = %v, want %v", got, want)
	}
	if got, want := chatToolDoneStyle.GetForeground(), lipgloss.Color(chatAnsiGreen); got != want {
		t.Fatalf("done tool foreground = %v, want %v", got, want)
	}

	running := newChatEntry(chatEntry{
		role:   "tool",
		detail: "bash",
		label:  `Bash("pwd")`,
		status: "running",
	})
	if line := toolStatusLine(running); line != `Bash("pwd")` {
		t.Fatalf("running tool status line = %q, want command label", line)
	}
	prefix, _ := (chatModel{}).renderEntry(running)
	if prefix != chatToolRunningStyle.Render("•")+" " {
		t.Fatalf("running tool prefix = %q, want running-styled marker", prefix)
	}

	child := newChatEntry(chatEntry{
		role:   "tool",
		detail: "bash",
		label:  `Bash("pwd")`,
		status: "done",
	})
	if line := toolGroupChildStatusLine(child); line != boldToolInvocationName(`Bash("pwd")`) {
		t.Fatalf("tool group child line = %q, want neutral invocation", line)
	}
}

func TestChatViewRendersInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, inputBoxTopBorderLine(40)) || !strings.Contains(view, inputBoxBottomBorderLine(40)) {
		t.Fatalf("prompt input should render box borders: %q", view)
	}
	if !strings.Contains(view, "│ hello") {
		t.Fatalf("view missing prompt input row: %q", view)
	}
}

func TestChatViewRendersSentUserPromptPrefix(t *testing.T) {
	m := chatModel{
		width:  40,
		height: 12,
		entries: []chatEntry{
			{role: "user", content: "hello"},
		},
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "> hello") {
		t.Fatalf("submitted user message should include prompt prefix: %q", view)
	}
	if strings.Contains(view, "│ >") {
		t.Fatalf("active input should not include prompt prefix: %q", view)
	}
}

func TestChatFlowViewStartsAtInputWhenEmpty(t *testing.T) {
	m := chatModel{
		input: []rune("hello"),
		width: 40,
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	if len(lines) == 0 || !strings.Contains(lines[0], inputBoxTopBorderLine(40)) {
		t.Fatalf("empty flow view should start at input box:\n%s", strings.Join(lines, "\n"))
	}
}

func TestFlowTranscriptChangedPrefixStartDetectsToolGrouping(t *testing.T) {
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	beforeModel := chatModel{
		width: 120,
		entries: []chatEntry{
			{
				role:   "tool",
				label:  `Bash("pwd")`,
				detail: "bash",
				status: "done",
				toolID: "call-1",
				args:   firstArgs,
			},
		},
	}
	before := beforeModel.transcriptLines(120)

	afterModel := beforeModel
	afterModel.entries = groupCompletedToolEntries([]chatEntry{
		beforeModel.entries[0],
		{
			role:   "tool",
			label:  `Bash("ls")`,
			detail: "bash",
			status: "done",
			toolID: "call-2",
			args:   secondArgs,
		},
	})

	after := afterModel.transcriptLines(120)
	if start := flowTranscriptChangedPrefixStart(before, after, len(before)); start != 0 {
		t.Fatalf("changed prefix start = %d, want 0; before=%q after=%q", start, before, after)
	}
	sequence := flowTranscriptRewriteSequence(len(before), after)
	if !strings.HasPrefix(sequence, "\x1b[1A\r\x1b[J") {
		t.Fatalf("rewrite sequence = %q, want cursor-up clear-below prefix", sequence)
	}
	if !strings.Contains(sequence, "Ran 2 commands") {
		t.Fatalf("grouping should invalidate already printed tool row; before=%q after=%q", before, afterModel.transcriptLines(120))
	}
}

func TestFlowTranscriptChangedPrefixStartIgnoresAppendedLines(t *testing.T) {
	beforeModel := chatModel{
		width: 120,
		entries: []chatEntry{
			{role: "assistant", content: "hello"},
		},
	}
	before := beforeModel.transcriptLines(120)
	afterModel := beforeModel
	afterModel.entries = append(afterModel.entries, chatEntry{role: "assistant", content: "world"})

	if start := flowTranscriptChangedPrefixStart(before, afterModel.transcriptLines(120), len(before)); start >= 0 {
		t.Fatalf("appended transcript lines should not invalidate already printed prefix")
	}
}

func TestChatViewRendersCursorWithEmptyPlaceholder(t *testing.T) {
	m := chatModel{
		chatID: "placeholder-chat",
		width:  80,
		height: 12,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "summarize this file and suggest edits") {
		t.Fatalf("empty placeholder should show hint: %q", view)
	}
}

func TestChatViewRendersModelUnderInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  48,
		height: 12,
		opts: Options{
			Model: "kimi-k2.7-code:cloud",
		},
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	inputLine := lineIndexContaining(lines, "│ hello")
	modelLine := lineIndexContaining(lines, "kimi-k2.7-code:cloud")
	if inputLine < 0 || modelLine < 0 {
		t.Fatalf("view missing input or model line:\n%s", strings.Join(lines, "\n"))
	}
	if modelLine != inputLine+2 {
		t.Fatalf("model line should sit directly under input: input=%d model=%d\n%s", inputLine, modelLine, strings.Join(lines, "\n"))
	}
	if strings.Contains(lines[modelLine], "model ") {
		t.Fatalf("model line should not include a label:\n%s", strings.Join(lines, "\n"))
	}
}

func TestChatViewExpandsInputBoxForLongPrompt(t *testing.T) {
	m := chatModel{
		input:  []rune(strings.Repeat("long prompt ", 8)),
		width:  32,
		height: 14,
	}

	view := stripANSI(m.View())
	if got := inputPromptLineCount(t, view); got < 2 {
		t.Fatalf("input body lines = %d, want wrapped prompt:\n%s", got, view)
	}
	if !strings.Contains(view, "█") {
		t.Fatalf("view missing cursor: %q", view)
	}
}

func TestChatViewCapsTallInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune(strings.Repeat("pasted text ", 80)),
		width:  32,
		height: 12,
	}

	view := stripANSI(m.View())
	if got := inputPromptLineCount(t, view); got > maxInputBoxBodyLines {
		t.Fatalf("input body lines = %d, want <= %d:\n%s", got, maxInputBoxBodyLines, view)
	}
	if strings.Contains(view, "... ... ") {
		t.Fatalf("truncated pasted prompt should not duplicate omission markers:\n%s", view)
	}
}

func TestChatViewWrapsNotificationWhenNarrow(t *testing.T) {
	m := chatModel{
		input:         []rune("hello"),
		width:         28,
		height:        14,
		status:        "cache will break by turning system prompt off",
		allowAllTools: true,
		opts: Options{
			ContextWindowTokens: 262144,
		},
		contextTokens:   12345,
		contextEstimate: true,
	}

	view := stripANSI(m.View())
	for _, want := range []string{
		"cache will break by",
		"system prompt off",
	} {
		if !strings.Contains(view, want) {
			t.Fatalf("wrapped notification missing %q:\n%s", want, view)
		}
	}
	for _, hidden := range []string{"enter", "send", "/model", "full", "access"} {
		if strings.Contains(view, hidden) {
			t.Fatalf("view should not render footer chrome %q:\n%s", hidden, view)
		}
	}
	if strings.Contains(view, "ctx") {
		t.Fatalf("view should hide distant context pressure:\n%s", view)
	}
	if strings.Contains(view, "ctrl+g") {
		t.Fatalf("view should not include ctrl+g hint:\n%s", view)
	}
	for _, line := range strings.Split(view, "\n") {
		if len([]rune(line)) > 28 {
			t.Fatalf("line width = %d, want <= 28: %q\n%s", len([]rune(line)), line, view)
		}
	}
}

func TestPromptTokenTextDoesNotUseCompactionFallback(t *testing.T) {
	m := chatModel{}
	if got := m.promptTokenText(702); got != "702 tokens" {
		t.Fatalf("promptTokenText without model context = %q, want bare token count", got)
	}

	m.opts.ContextWindowTokens = 262144
	if got := m.promptTokenText(702); got != "702 / 262144 tokens" {
		t.Fatalf("promptTokenText with model context = %q", got)
	}

	m.opts.Options = map[string]any{"num_ctx": 131072}
	if got := m.promptTokenText(702); got != "702 / 131072 tokens" {
		t.Fatalf("promptTokenText with num_ctx = %q", got)
	}
}

func TestPreloadDoneUpdatesContextWindowTokens(t *testing.T) {
	compactor := &coreagent.SimpleCompactor{}
	m := chatModel{
		preloadingModel: "ornith",
		opts: Options{
			ContextWindowTokens: 32768,
			Compactor:           compactor,
		},
	}

	next, _ := m.Update(chatModelPreloadDoneMsg{model: "ornith", contextWindowTokens: 262144})
	got := next.(chatModel)
	if got.opts.ContextWindowTokens != 262144 {
		t.Fatalf("context window = %d, want 262144", got.opts.ContextWindowTokens)
	}
	if compactor.Options.ContextWindowTokens != 262144 {
		t.Fatalf("compactor context window = %d, want 262144", compactor.Options.ContextWindowTokens)
	}
}

func TestChatViewRendersNotificationAboveInput(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
		status: "copied latest output",
	}

	view := stripANSI(m.View())
	lines := strings.Split(view, "\n")
	borderLine := lineIndexContaining(lines, inputBoxTopBorderLine(40))
	if borderLine < 0 {
		t.Fatalf("view missing input box:\n%s", view)
	}
	if borderLine < 1 || !strings.Contains(lines[borderLine-1], "copied latest output") {
		t.Fatalf("notification should sit directly above input box:\n%s", view)
	}
}

func TestChatNotificationsUseSecondaryStyle(t *testing.T) {
	if !chatNotificationStyle.GetFaint() {
		t.Fatal("notification style should use secondary/faint styling")
	}
}

func TestChatSubmittedPromptUsesThemeSecondaryGrey(t *testing.T) {
	if got, want := chatUserBlockStyle.GetForeground(), lipgloss.Color(chatAnsiBrightBlack); got != want {
		t.Fatalf("submitted prompt foreground = %v, want %v", got, want)
	}
	if chatUserBlockStyle.GetFaint() {
		t.Fatal("submitted prompt should use ANSI secondary grey, not faint styling")
	}
}

func TestChatInlineCodeDoesNotLookSelected(t *testing.T) {
	if chatInlineCodeStyle.GetReverse() {
		t.Fatal("inline code should not use reverse-video styling")
	}
	if !chatInlineCodeStyle.GetBold() {
		t.Fatal("inline code should still have lightweight emphasis")
	}
}

func inputPromptLineCount(t *testing.T, view string) int {
	t.Helper()
	count := 0
	inInputBox := false
	for _, line := range strings.Split(view, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "╭") {
			inInputBox = true
			continue
		}
		if strings.HasPrefix(trimmed, "╰") {
			inInputBox = false
			continue
		}
		if inInputBox && strings.Contains(trimmed, "│") {
			count++
		}
	}
	if count == 0 {
		t.Fatalf("input prompt lines not found:\n%s", view)
	}
	return count
}

func TestChatViewKeepsInputBoxWhileRunning(t *testing.T) {
	m := chatModel{
		input:          []rune("next"),
		running:        true,
		thinking:       true,
		thinkingTokens: 42,
		width:          40,
		height:         12,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "│ next") {
		t.Fatalf("running view should keep input row: %q", view)
	}
	if strings.Contains(view, "↑/↓ scroll") || strings.Contains(view, "/new chat") || strings.Contains(view, "/clear reset") {
		t.Fatalf("footer should not include scroll/new/clear hints: %q", view)
	}
	lines := strings.Split(view, "\n")
	borderLine := lineIndexContaining(lines, inputBoxTopBorderLine(40))
	if borderLine < 0 {
		t.Fatalf("view missing input box: %q", view)
	}
	if borderLine < 1 || !strings.Contains(lines[borderLine-1], "Thinking 42 tokens") {
		t.Fatalf("thinking line should sit directly above input box:\n%s", view)
	}
}

func TestChatToolFinishedUpdatesLiveWorkingDirOnly(t *testing.T) {
	root := t.TempDir()
	subdir := filepath.Join(root, "sub")
	if err := os.Mkdir(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	m := chatModel{
		width:  140,
		height: 12,
		opts: Options{
			RootDir:    root,
			WorkingDir: root,
		},
		workingDir: root,
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		WorkingDir: subdir,
	})

	if m.workingDir != subdir {
		t.Fatalf("workingDir = %q, want %q", m.workingDir, subdir)
	}
	if m.opts.WorkingDir != root {
		t.Fatalf("opts.WorkingDir mutated to %q, want %q", m.opts.WorkingDir, root)
	}
}

func TestChatBoundedFramePinsInputAfterResize(t *testing.T) {
	m := chatModel{
		input:  []rune("next"),
		width:  60,
		height: 14,
	}
	for i := range 20 {
		m.entries = append(m.entries, chatEntry{role: "assistant", content: fmt.Sprintf("line %02d", i)})
	}
	inputLine := renderedInputLine(m.View())
	if inputLine < m.height-4 {
		t.Fatalf("bounded frame should pin input near bottom, input line=%d height=%d\n%s", inputLine, m.height, stripANSI(m.View()))
	}
}

func TestChatViewSeparatesActionStatusFromTranscript(t *testing.T) {
	m := chatModel{
		input:    []rune("next"),
		width:    72,
		height:   14,
		running:  true,
		thinking: true,
		entries: []chatEntry{
			{role: "assistant", content: "Working through it."},
		},
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	assistantLine := lineIndexContaining(lines, "Working through it.")
	activityLine := lineIndexContaining(lines, "Thinking")
	if assistantLine < 0 || activityLine < 0 {
		t.Fatalf("view missing assistant/activity lines:\n%s", strings.Join(lines, "\n"))
	}
	if gap := activityLine - assistantLine - 1; gap < 1 {
		t.Fatalf("gap between transcript and action status = %d, want at least 1:\n%s", gap, strings.Join(lines, "\n"))
	}
}

func TestChatViewDoesNotReserveIdleActionSpacerAfterResponse(t *testing.T) {
	m := chatModel{
		input:  []rune("next"),
		width:  72,
		height: 12,
		entries: []chatEntry{
			{role: "user", content: "hi"},
			{role: "assistant", content: "Hello."},
		},
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	assistantLine := lineIndexContaining(lines, "Hello.")
	inputLine := lineIndexContaining(lines, "│ next")
	if assistantLine < 0 || inputLine < 0 {
		t.Fatalf("view missing assistant/input lines:\n%s", strings.Join(lines, "\n"))
	}
	if gap := inputLine - assistantLine - 1; gap < 2 {
		t.Fatalf("gap between finished response and input body = %d, want at least 2:\n%s", gap, strings.Join(lines, "\n"))
	}
}

func TestChatViewHidesEmptyHintWhileTyping(t *testing.T) {
	m := chatModel{
		input:  []rune("next"),
		width:  72,
		height: 12,
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	hintLine := lineIndexContaining(lines, "Try:")
	inputLine := lineIndexContaining(lines, "│ next")
	if hintLine >= 0 {
		t.Fatalf("view should not show empty hint while typing:\n%s", strings.Join(lines, "\n"))
	}
	if inputLine < 0 {
		t.Fatalf("view missing input line:\n%s", strings.Join(lines, "\n"))
	}
}

func renderedInputLine(view string) int {
	for i, line := range strings.Split(stripANSI(view), "\n") {
		if strings.Contains(line, "│ next") {
			return i
		}
	}
	return -1
}

func lineIndexContaining(lines []string, needle string) int {
	for i, line := range lines {
		if strings.Contains(line, needle) {
			return i
		}
	}
	return -1
}

func TestChatScrollsTranscript(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 10,
	}
	for range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: "line"})
	}

	if m.maxScroll() == 0 {
		t.Fatal("test setup should produce scrollable transcript")
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	m = updated.(chatModel)
	if m.scroll == 0 {
		t.Fatal("page up should scroll transcript")
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("scroll = %d, want 0", m.scroll)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlHome})
	m = updated.(chatModel)
	if m.scroll != m.maxScroll() {
		t.Fatalf("scroll = %d, want max %d", m.scroll, m.maxScroll())
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlEnd})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("scroll = %d, want 0", m.scroll)
	}
}

func TestChatStreamingAssistantOutputHoldsLiveMarkdown(t *testing.T) {
	m := chatModel{
		width:   80,
		height:  12,
		running: true,
		events:  make(chan tea.Msg),
	}

	updated, _ := m.Update(chatAgentMsg{event: coreagent.Event{Type: coreagent.EventMessageDelta, Content: "generated line 00"}})
	m = updated.(chatModel)
	if !strings.Contains(stripANSI(m.View()), "generated line 00") {
		t.Fatalf("live output should be visible in flow view:\n%s", stripANSI(m.View()))
	}

	updated, _ = m.Update(chatAgentMsg{event: coreagent.Event{Type: coreagent.EventMessageDelta, Content: "\n\ngenerated line 01"}})
	m = updated.(chatModel)
	view := stripANSI(m.View())
	if !strings.Contains(view, "generated line 00") {
		t.Fatalf("live assistant output should remain visible while streaming:\n%s", view)
	}
	if !strings.Contains(view, "generated line 01") {
		t.Fatalf("latest generated line should remain visible:\n%s", view)
	}
}

func TestChatMouseWheelScrollsTranscriptWhileRunning(t *testing.T) {
	m := chatModel{
		width:         80,
		height:        10,
		running:       true,
		input:         []rune("current draft"),
		promptHistory: []string{"previous one", "previous two"},
	}
	for range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: "line"})
	}
	if m.maxScroll() == 0 {
		t.Fatal("test setup should produce scrollable transcript")
	}

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseWheelUp})
	m = updated.(chatModel)
	if m.scroll == 0 {
		t.Fatal("mouse wheel up should scroll transcript while running")
	}
	if got := string(m.input); got != "current draft" {
		t.Fatalf("mouse wheel should not navigate prompt history, input = %q", got)
	}

	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseWheelDown})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("mouse wheel down should return to bottom, got scroll %d", m.scroll)
	}
	if got := string(m.input); got != "current draft" {
		t.Fatalf("mouse wheel should leave draft alone, input = %q", got)
	}
}

func TestChatWindowsMouseWheelScrollsTranscript(t *testing.T) {
	oldGOOS := chatRuntimeGOOS
	chatRuntimeGOOS = "windows"
	defer func() {
		chatRuntimeGOOS = oldGOOS
	}()

	m := chatModel{
		width:  80,
		height: 8,
	}
	for i := range 20 {
		m.entries = append(m.entries, chatEntry{role: "user", content: fmt.Sprintf("line-%02d", i)})
	}

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseWheelUp})
	m = updated.(chatModel)
	if m.scroll == 0 {
		t.Fatal("windows mouse wheel up should scroll transcript")
	}

	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseWheelDown})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("windows mouse wheel down should return to bottom, got scroll %d", m.scroll)
	}
}

func TestChatMouseDragSelectsTranscriptWithoutAutoCopy(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 10,
		entries: []chatEntry{
			{role: "user", content: "alpha beta"},
		},
	}
	top, _ := m.transcriptLayout()

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: 2, Y: top})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: 7, Y: top})
	m = updated.(chatModel)
	if got := m.selectedTranscriptText(80); got != "alpha" {
		t.Fatalf("selected text = %q, want alpha", got)
	}
	if !m.selection.active {
		t.Fatal("selection should stay active during drag")
	}
	if !m.selection.dragging {
		t.Fatal("selection should track drag before release")
	}

	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: 7, Y: top})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("mouse release should not auto-copy selected text")
	}
	if !m.selection.active {
		t.Fatal("selection should stay visible on release")
	}
	if m.selection.dragging {
		t.Fatal("selection should stop tracking drag on release")
	}
	if got := m.selectedTranscriptText(80); got != "alpha" {
		t.Fatalf("selected text after release = %q, want alpha", got)
	}
}

func TestChatWindowsMouseDragCopiesAndClearsTranscriptSelection(t *testing.T) {
	oldGOOS := chatRuntimeGOOS
	chatRuntimeGOOS = "windows"
	defer func() {
		chatRuntimeGOOS = oldGOOS
	}()

	var copied string
	oldClipboard := writeClipboard
	writeClipboard = func(_ context.Context, text string) error {
		copied = text
		return nil
	}
	defer func() {
		writeClipboard = oldClipboard
	}()

	m := chatModel{
		width:  80,
		height: 10,
		entries: []chatEntry{
			{role: "user", content: "alpha beta"},
		},
	}
	top, _ := m.transcriptLayout()

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: 2, Y: top})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: 7, Y: top})
	m = updated.(chatModel)
	if got := m.selectedTranscriptText(80); got != "alpha" {
		t.Fatalf("selected text = %q, want alpha", got)
	}

	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: 7, Y: top})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("mouse release should copy selected text on windows")
	}
	if msg := cmd(); msg != nil {
		t.Fatalf("clipboard command message = %#v, want nil", msg)
	}
	if copied != "alpha" {
		t.Fatalf("copied text = %q, want alpha", copied)
	}
	if m.selection.active {
		t.Fatal("windows selection should clear after copy")
	}
	if m.status != "copied" {
		t.Fatalf("status = %q, want copied", m.status)
	}
}

func TestChatMouseDragSelectionUsesDisplayColumns(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 10,
		entries: []chatEntry{
			{role: "user", content: "a界b"},
		},
	}
	top, _ := m.transcriptLayout()
	contentX := 2

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: contentX, Y: top})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: contentX + 3, Y: top})
	m = updated.(chatModel)

	if got := m.selectedTranscriptText(80); got != "a界" {
		t.Fatalf("selected text = %q, want a界", got)
	}
}

func TestChatBoundedViewDoesNotRenderScrollHeader(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 6,
	}
	for i := range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: fmt.Sprintf("line-%02d", i)})
	}
	m.scroll = m.maxScroll()
	top, _ := m.transcriptLayout()
	lines := strings.Split(stripANSI(m.View()), "\n")
	if top != 0 {
		t.Fatalf("transcript top = %d, want no status header", top)
	}
	if strings.Contains(lines[0], "more") {
		t.Fatalf("view should not render scroll status header: %q", lines[0])
	}
	if strings.TrimSpace(lines[top]) == "" {
		t.Fatalf("transcript should start at layout top %d, line=%q view=%q", top, lines[top], strings.Join(lines, "\n"))
	}
}

func TestChatMouseDragSelectionUsesScrolledTranscriptCoordinates(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 8,
	}
	for i := range 10 {
		m.entries = append(m.entries, chatEntry{role: "user", content: fmt.Sprintf("line-%02d", i)})
	}
	m.scroll = m.maxScroll()
	top, _ := m.transcriptLayout()

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: 2, Y: top})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseMotion, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: 9, Y: top})
	m = updated.(chatModel)
	if got := m.selectedTranscriptText(80); got != "line-00" {
		t.Fatalf("selected text = %q, want line-00", got)
	}
	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: 9, Y: top})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("mouse release should not auto-copy selected text")
	}
	if !m.selection.active {
		t.Fatal("selection should stay visible on release")
	}
	if m.selection.dragging {
		t.Fatal("selection should stop tracking drag on release")
	}
}

func TestChatArrowKeysNavigatePromptHistoryWhenTranscriptScrollable(t *testing.T) {
	m := chatModel{
		input:         []rune("current draft"),
		promptHistory: []string{"old prompt"},
		width:         80,
		height:        10,
		running:       true,
	}
	for range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: "line"})
	}
	if m.maxScroll() == 0 {
		t.Fatal("test setup should produce scrollable transcript")
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("key up should not scroll transcript, got %d", m.scroll)
	}
	if got := string(m.input); got != "old prompt" {
		t.Fatalf("key up should recall prompt history, got %q", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyDown})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("key down should not scroll transcript, got %d", m.scroll)
	}
	if got := string(m.input); got != "current draft" {
		t.Fatalf("key down should restore draft, got %q", got)
	}
}

func TestEntriesFromMessagesSkipsToolCallOnlyAssistant(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	messages := []api.Message{
		{Role: "user", Content: "pwd"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{{
				ID: "call-1",
				Function: api.ToolCallFunction{
					Name:      "bash",
					Arguments: args,
				},
			}},
		},
		{Role: "tool", ToolName: "bash", ToolCallID: "call-1", Content: "/tmp/project\n"},
		{Role: "assistant", Content: "The current directory is /tmp/project."},
	}

	entries := entriesFromMessages(messages)
	if len(entries) != 3 {
		t.Fatalf("entries = %d, want user/tool/assistant: %#v", len(entries), entries)
	}
	if entries[1].role != "tool" || entries[1].label != "Bash(\"pwd\")" {
		t.Fatalf("tool entry = %#v", entries[1])
	}
	if line := stripANSI(toolStatusLine(entries[1])); line != `Bash("pwd")` {
		t.Fatalf("tool status line = %q, want command label", line)
	}

	transcript := stripANSI((chatModel{entries: entries}).renderTranscript(120))
	if strings.Contains(transcript, "• \n") || strings.Contains(transcript, "•\n") {
		t.Fatalf("transcript has blank assistant bullet: %q", transcript)
	}
}

func TestEntriesFromMessagesRendersDeniedCommandAsDenied(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	messages := []api.Message{
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{{
				ID: "call-1",
				Function: api.ToolCallFunction{
					Name:      "bash",
					Arguments: args,
				},
			}},
		},
		{Role: "tool", ToolName: "bash", ToolCallID: "call-1", Content: "Tool execution denied."},
	}

	entries := entriesFromMessages(messages)
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want one tool entry: %#v", len(entries), entries)
	}
	if entries[0].status != "denied" {
		t.Fatalf("tool status = %q, want denied: %#v", entries[0].status, entries[0])
	}
	if line := stripANSI(toolStatusLine(entries[0])); line != `Bash("pwd") denied` {
		t.Fatalf("tool status line = %q, want denied command label", line)
	}
}

func TestEntriesFromMessagesRendersCompactionSummaryCollapsed(t *testing.T) {
	entries := entriesFromMessages([]api.Message{
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: coreagent.CompactionToolCallID,
			Function: api.ToolCallFunction{
				Name: coreagent.CompactionToolName,
			},
		}}},
		{Role: "tool", ToolName: coreagent.CompactionToolName, ToolCallID: coreagent.CompactionToolCallID, Content: coreagent.CompactionSummaryMessagePrefix + "- old work\n- decisions"},
		{Role: "user", Content: "recent request"},
	})
	if len(entries) != 2 {
		t.Fatalf("entries = %d, want summary plus user: %#v", len(entries), entries)
	}
	if entries[0].role != "compaction_summary" || entries[0].content != "- old work\n- decisions" {
		t.Fatalf("summary entry = %#v", entries[0])
	}

	m := chatModel{entries: entries}
	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "Compacted summary") {
		t.Fatalf("collapsed summary row missing: %q", transcript)
	}
	if strings.Contains(transcript, "Compacted summary done") {
		t.Fatalf("summary row should not include done word: %q", transcript)
	}
	if strings.Contains(transcript, "old work") {
		t.Fatalf("summary body should be collapsed: %q", transcript)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	view := stripANSI(m.renderTranscript(100))
	if !strings.Contains(view, "old work") || !strings.Contains(view, "decisions") {
		t.Fatalf("inline summary body missing: %q", view)
	}
}

func TestEntriesFromMessagesHidesAutomaticCompactionInstruction(t *testing.T) {
	entries := entriesFromMessages([]api.Message{
		{
			Role:       "tool",
			ToolName:   coreagent.CompactionToolName,
			ToolCallID: coreagent.CompactionToolCallID,
			Content:    coreagent.CompactionSummaryMessagePrefix + "old work summary\n\n" + coreagent.CompactionContinueInstruction,
		},
	})
	if len(entries) != 1 || entries[0].role != "compaction_summary" || entries[0].content != "old work summary" {
		t.Fatalf("entries = %#v", entries)
	}
}

func TestEntriesFromMessagesRecognizesLegacySystemCompactionSummary(t *testing.T) {
	entries := entriesFromMessages([]api.Message{
		{Role: "system", Content: "Conversation summary:\nlegacy summary"},
	})
	if len(entries) != 1 || entries[0].role != "compaction_summary" || entries[0].content != "legacy summary" {
		t.Fatalf("entries = %#v", entries)
	}
}

func TestEntriesFromMessagesGroupsMultiToolHistoryInMiddle(t *testing.T) {
	readArgs := api.NewToolCallFunctionArguments()
	readArgs.Set("path", "feedback")
	listArgs := api.NewToolCallFunctionArguments()
	listArgs.Set("path", ".")
	messages := []api.Message{
		{Role: "user", Content: "read feedback file"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call-read",
					Function: api.ToolCallFunction{
						Name:      "read",
						Arguments: readArgs,
					},
				},
				{
					ID: "call-list",
					Function: api.ToolCallFunction{
						Name:      "list",
						Arguments: listArgs,
					},
				},
			},
		},
		{Role: "tool", ToolName: "read", ToolCallID: "call-read", Content: "Error: no such file"},
		{Role: "tool", ToolName: "list", ToolCallID: "call-list", Content: "feedback.md\n"},
		{Role: "assistant", Content: "There is a feedback.md file."},
	}

	entries := entriesFromMessages(messages)
	if len(entries) != 3 {
		t.Fatalf("entries = %d, want user/tool-group/assistant: %#v", len(entries), entries)
	}
	if entries[1].role != "tool_group" || len(entries[1].tools) != 2 {
		t.Fatalf("middle entry should be grouped tool history: %#v", entries[1])
	}
	if entries[1].tools[0].label != "Read(\"feedback\")" || entries[1].tools[1].label != "List(\".\")" {
		t.Fatalf("group labels = %#v", entries[1].tools)
	}
}

func TestChatToolCallDetectedDoesNotRenderQueuedRows(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	m := chatModel{}

	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "bash",
				Arguments: args,
			},
		}},
	})

	if len(m.entries) != 0 {
		t.Fatalf("queued tool call should not create history entries: %#v", m.entries)
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args.ToMap(),
	})
	if len(m.entries) != 1 {
		t.Fatalf("started tool should create one visible entry, got %d", len(m.entries))
	}
	if got := m.activityLabel(); got != "" {
		t.Fatalf("activityLabel = %q, want no transient tool label", got)
	}
	if line := strings.TrimSpace(stripANSI(m.activityLine())); line != "" {
		t.Fatalf("activityLine = %q, want no transient tool action line", line)
	}
}

func TestChatToolOutputIsHiddenUntilExpanded(t *testing.T) {
	fullOutput := strings.Repeat("x", maxCtrlOToolOutputRunes+25) + "tail-marker"
	m := chatModel{}
	m.applyAgentEvent(coreagent.Event{
		Type:     coreagent.EventToolFinished,
		ToolName: "bash",
		Content:  fullOutput,
	})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(m.entries))
	}
	if m.entries[0].content != fullOutput {
		t.Fatal("tool entry should keep full content before rendering")
	}

	transcript := m.renderTranscript(100)
	body := stripANSI(transcript)
	if strings.Contains(body, "tail-marker") || strings.Contains(body, strings.Repeat("x", 20)) {
		t.Fatalf("collapsed tool output should be hidden: %q", body)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)

	if !m.toolOutputMode || !m.toolOutputOpen || !m.entries[0].expanded {
		t.Fatalf("ctrl+o should expand tool output inline: %#v", m.entries[0])
	}
	body = stripANSI(m.renderTranscript(100))
	if strings.Contains(body, "tail-marker") {
		t.Fatalf("expanded transcript should cap tool output: %q", body)
	}
	if !strings.Contains(body, "...") {
		t.Fatalf("expanded transcript should show capped output ellipsis: %q", body)
	}
	if got := strings.Count(body, "x"); got != maxCtrlOToolOutputRunes-3 {
		t.Fatalf("expanded transcript x count = %d, want %d:\n%s", got, maxCtrlOToolOutputRunes-3, body)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	body = stripANSI(m.renderTranscript(100))
	if m.toolOutputOpen || m.entries[0].expanded || strings.Contains(body, strings.Repeat("x", 20)) {
		t.Fatalf("second ctrl+o should collapse tool output: %q", body)
	}
}

func TestChatCompletedToolsGroupWhenNextStepStarts(t *testing.T) {
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	thirdArgs := map[string]any{"command": "date"}
	m := chatModel{}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-1", ToolName: "bash", Args: firstArgs, Content: "one"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolFinished, ToolCallID: "call-2", ToolName: "bash", Args: secondArgs, Content: "two"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventToolStarted, ToolCallID: "call-3", ToolName: "bash", Args: thirdArgs})

	if len(m.entries) != 2 {
		t.Fatalf("entries = %d, want grouped history plus active tool: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || len(m.entries[0].tools) != 2 {
		t.Fatalf("first entry should be grouped tool history: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}
	if m.entries[1].status != "running" || m.entries[1].label != "Bash(\"date\")" {
		t.Fatalf("second entry should be active tool: %#v", m.entries[1])
	}
	if line := stripANSI(toolStatusLine(m.entries[1])); line != `Bash("date")` {
		t.Fatalf("running command line = %q", line)
	}
}

func TestChatCompletedToolsGroupAcrossEmptyAssistantEntries(t *testing.T) {
	entries := groupCompletedToolEntries([]chatEntry{
		newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("ls")`, status: "done", content: "listed"}),
		newChatEntry(chatEntry{role: "assistant"}),
		newChatEntry(chatEntry{role: "tool", detail: "read", label: `Read("agent")`, status: "error", err: "is directory", content: "is directory"}),
		newChatEntry(chatEntry{role: "assistant", content: "   "}),
		newChatEntry(chatEntry{role: "tool", detail: "read", label: `Read("AGENTS.md")`, status: "done", content: "instructions"}),
		newChatEntry(chatEntry{role: "assistant", content: "The files were listed."}),
	})

	if len(entries) != 2 {
		t.Fatalf("entries = %d, want grouped tools plus assistant: %#v", len(entries), entries)
	}
	if entries[0].role != "tool_group" || len(entries[0].tools) != 3 {
		t.Fatalf("first entry should group tools across empty assistant entries: %#v", entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(entries[0])); line != "Ran 1 command and read 2 files" {
		t.Fatalf("grouped tool line = %q", line)
	}
	if entries[1].role != "assistant" || entries[1].content != "The files were listed." {
		t.Fatalf("second entry should keep real assistant content: %#v", entries[1])
	}
}

func TestChatCompletedToolsGroupAcrossCollapsedThinking(t *testing.T) {
	entries := groupCompletedToolEntries([]chatEntry{
		newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("pwd")`, status: "done", content: "one"}),
		newChatEntry(chatEntry{role: "thinking", label: "Thinking", content: "choose next tool", status: "done"}),
		newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("ls")`, status: "done", content: "two"}),
		newChatEntry(chatEntry{role: "assistant"}),
		newChatEntry(chatEntry{role: "tool", detail: "read", label: `Read("AGENTS.md")`, status: "done", content: "instructions"}),
	})

	if len(entries) != 1 {
		t.Fatalf("entries = %d, want one grouped tool entry: %#v", len(entries), entries)
	}
	if entries[0].role != "tool_group" || len(entries[0].tools) != 3 {
		t.Fatalf("tools should group across collapsed thinking: %#v", entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(entries[0])); line != "Ran 2 commands and read a file" {
		t.Fatalf("grouped tool line = %q", line)
	}
}

func TestChatCtrlOTogglesInlineToolOutput(t *testing.T) {
	m := chatModel{
		width:  100,
		height: 20,
		entries: []chatEntry{
			{role: "tool", detail: "bash", label: "Bash(\"pwd\")", status: "done", content: "one"},
			{role: "assistant", content: "between"},
			{role: "tool", detail: "read", label: "Read(\"file\")", status: "error", err: "nope", content: "two"},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	for _, index := range []int{0, 2} {
		if !m.entries[index].expanded {
			t.Fatalf("tool entry %d should be expanded inline", index)
		}
	}
	view := stripANSI(m.renderTranscript(100))
	if strings.Contains(view, "Tool details") {
		t.Fatalf("ctrl+o should keep tool output inline: %q", view)
	}
	if !strings.Contains(view, "one") || !strings.Contains(view, "two") {
		t.Fatalf("view missing inline expanded tool output: %q", view)
	}
	if !strings.Contains(view, "between") {
		t.Fatalf("inline tool output should keep surrounding chat visible: %q", view)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	for _, index := range []int{0, 2} {
		if m.entries[index].expanded {
			t.Fatalf("tool entry %d should remain collapsed", index)
		}
	}
}

func TestChatCtrlOTogglesInlineOutput(t *testing.T) {
	m := chatModel{
		width:  100,
		height: 24,
		entries: []chatEntry{
			{role: "user", content: "who is parth sareen"},
			{role: "tool", detail: "web_search", label: "Web Search(\"who is Parth Sareen\")", status: "done", content: "Search results"},
			{role: "assistant", content: "Based on public search results, Parth Sareen works on AI tooling."},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if !m.toolOutputOpen || !m.entries[1].expanded {
		t.Fatalf("tool output should be expanded inline: %#v", m.entries[1])
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if m.toolOutputOpen || m.entries[1].expanded {
		t.Fatalf("tool output should be collapsed inline: %#v", m.entries[1])
	}
}

func TestChatCtrlODoesNotExpandThinking(t *testing.T) {
	m := chatModel{
		entries: []chatEntry{
			newChatEntry(chatEntry{role: "thinking", label: "Thinking", status: "done", content: "private reasoning"}),
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if m.entries[0].expanded {
		t.Fatalf("ctrl+o should not expand thinking entries: %#v", m.entries[0])
	}
	if view := stripANSI(m.renderTranscript(100)); strings.Contains(view, "private reasoning") {
		t.Fatalf("ctrl+o should not render thinking content:\n%s", view)
	}

	m.liveMessages = []api.Message{{Role: "assistant", Thinking: "live private reasoning"}}
	m.syncThinkingEntry()
	if m.entries[1].expanded {
		t.Fatalf("live thinking should not inherit ctrl+o expansion: %#v", m.entries[1])
	}
	if view := stripANSI(m.renderTranscript(100)); strings.Contains(view, "live private reasoning") {
		t.Fatalf("ctrl+o should not render live thinking content:\n%s", view)
	}
}

func TestChatCtrlOCollapseKeepsInputPromptVisible(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 8,
		opts:   Options{Model: "gemma4"},
		entries: []chatEntry{
			{role: "user", content: "inspect this"},
			{role: "tool", detail: "bash", label: "Bash(\"pwd\")", status: "done", content: strings.Repeat("output\n", 20)},
			{role: "assistant", content: "Done."},
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)

	view := stripANSI(m.View())
	if !strings.Contains(view, "│ █") {
		t.Fatalf("input prompt disappeared after collapsing tool output:\n%s", view)
	}
}

func TestChatCtrlOShowsRunningToolOutputInline(t *testing.T) {
	args := map[string]any{"command": "pwd"}
	m := chatModel{
		width:   100,
		height:  20,
		running: true,
		opts:    Options{Model: "test-model"},
	}
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
	})

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if !m.entries[0].expanded {
		t.Fatalf("ctrl+o should expand the running tool inline")
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, `Bash("pwd")`) || !strings.Contains(view, "$ pwd") {
		t.Fatalf("expanded transcript missing running tool details:\n%s", view)
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
		Content:    "/tmp/project\n",
	})

	view = stripANSI(m.View())
	if !strings.Contains(view, "/tmp/project") {
		t.Fatalf("finished tool output should be visible inline: %q", view)
	}
	if !strings.Contains(view, "│ █") {
		t.Fatalf("input prompt disappeared while tool output is expanded:\n%s", view)
	}
	if !strings.Contains(view, "test-model") {
		t.Fatalf("footer/model line disappeared while tool output is expanded:\n%s", view)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	view = stripANSI(m.View())
	if !strings.Contains(view, "│ █") {
		t.Fatalf("input prompt disappeared after collapsing tool output:\n%s", view)
	}
	if !strings.Contains(view, "test-model") {
		t.Fatalf("footer/model line disappeared after collapsing tool output:\n%s", view)
	}
}

func TestChatLongCommandLabelTruncatesAndCtrlOShowsFullCommand(t *testing.T) {
	command := strings.Repeat("x", 120)
	m := chatModel{
		width:  180,
		height: 20,
	}
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       map[string]any{"command": command},
	})

	wantLabel := `Bash("` + strings.Repeat("x", 100) + `...")`
	if m.entries[0].label != wantLabel {
		t.Fatalf("label = %q, want %q", m.entries[0].label, wantLabel)
	}
	if strings.Contains(m.entries[0].label, strings.Repeat("x", 101)) {
		t.Fatalf("collapsed label should truncate command: %q", m.entries[0].label)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	view := stripANSI(m.renderTranscript(180))
	if !strings.Contains(view, "$ "+command) {
		t.Fatalf("expanded ctrl+o view should show full command:\n%s", view)
	}
}

func TestChatCtrlOInlineOutputSurvivesToolGrouping(t *testing.T) {
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	m := chatModel{
		width:  100,
		height: 24,
		entries: []chatEntry{
			newChatEntry(chatEntry{role: "tool", detail: "bash", label: "Bash(\"pwd\")", status: "done", content: "one", args: firstArgs}),
			newChatEntry(chatEntry{role: "tool", detail: "bash", label: "Bash(\"ls\")", status: "done", content: "two", args: secondArgs}),
		},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})

	if len(m.entries) != 2 {
		t.Fatalf("entries = %d, want tool group plus assistant: %#v", len(m.entries), m.entries)
	}
	if m.entries[0].role != "tool_group" || !m.entries[0].expanded {
		t.Fatalf("grouped tool history should stay expanded inline: %#v", m.entries[0])
	}
	if line := stripANSI(toolGroupStatusLine(m.entries[0])); line != "Ran 2 commands" {
		t.Fatalf("grouped command line = %q", line)
	}

	view := stripANSI(m.renderTranscript(100))
	for _, want := range []string{`Bash("pwd")`, `Bash("ls")`, "one", "two"} {
		if !strings.Contains(view, want) {
			t.Fatalf("grouped tool output missing %q:\n%s", want, view)
		}
	}
	if strings.Contains(view, "    Ran 1 command") {
		t.Fatalf("expanded grouped children should show concrete invocations, not generic summaries:\n%s", view)
	}
}

func TestChatExpandedToolGroupRendersIndentedToolBlocks(t *testing.T) {
	startedAt := time.Date(2026, 6, 11, 12, 0, 0, 0, time.UTC)
	m := chatModel{
		entries: []chatEntry{
			newChatEntry(chatEntry{
				role:     "tool_group",
				label:    "Tool calls (2)",
				status:   "done",
				expanded: true,
				tools: []chatEntry{
					newChatEntry(chatEntry{
						role:       "tool",
						detail:     "web_search",
						label:      "Web Search(\"Parth Sareen\")",
						status:     "done",
						content:    "Search results for: Parth Sareen",
						startedAt:  startedAt,
						finishedAt: startedAt.Add(823 * time.Millisecond),
					}),
					newChatEntry(chatEntry{
						role:       "tool",
						detail:     "read",
						label:      "Read(\"feedback.md\")",
						status:     "done",
						content:    "Looks good.",
						startedAt:  startedAt.Add(time.Second),
						finishedAt: startedAt.Add(3 * time.Second),
					}),
				},
			}),
		},
	}

	transcript := stripANSI(m.renderTranscript(120))
	expected := strings.Join([]string{
		"    Web Search(\"Parth Sareen\")",
		"      Search results for: Parth Sareen",
		"  ",
		"    Read(\"feedback.md\")",
		"      Looks good.",
	}, "\n")
	if !strings.Contains(transcript, expected) {
		t.Fatalf("expanded tool group did not render expected block spacing:\n%s", transcript)
	}
}

func TestChatExpandedToolGroupCapsChildOutput(t *testing.T) {
	longOutput := strings.Repeat("y", maxCtrlOToolOutputRunes+25) + "group-tail-marker"
	m := chatModel{
		entries: []chatEntry{
			newChatEntry(chatEntry{
				role:     "tool_group",
				status:   "done",
				expanded: true,
				tools: []chatEntry{
					newChatEntry(chatEntry{
						role:    "tool",
						detail:  "read",
						label:   "Read(\"big.log\")",
						status:  "done",
						content: longOutput,
					}),
				},
			}),
		},
	}

	transcript := stripANSI(m.renderTranscript(120))
	if strings.Contains(transcript, "group-tail-marker") {
		t.Fatalf("expanded grouped tool output should be capped:\n%s", transcript)
	}
	if !strings.Contains(transcript, "...") {
		t.Fatalf("expanded grouped tool output should show capped output ellipsis:\n%s", transcript)
	}
	if got := strings.Count(transcript, "y"); got != maxCtrlOToolOutputRunes-3 {
		t.Fatalf("expanded grouped tool output y count = %d, want %d:\n%s", got, maxCtrlOToolOutputRunes-3, transcript)
	}
	if got := m.entries[0].tools[0].content; got != longOutput {
		t.Fatal("ctrl+o rendering should not mutate grouped tool content")
	}
}

func TestChatToolGroupStatusOmitsResultCounts(t *testing.T) {
	startedAt := time.Date(2026, 6, 15, 17, 0, 0, 0, time.UTC)
	entry := newChatEntry(chatEntry{
		role:       "tool_group",
		label:      "Tool calls (3)",
		status:     "error",
		startedAt:  startedAt,
		finishedAt: startedAt.Add(3 * time.Second),
		tools: []chatEntry{
			newChatEntry(chatEntry{role: "tool", status: "done"}),
			newChatEntry(chatEntry{role: "tool", status: "done"}),
			newChatEntry(chatEntry{role: "tool", status: "error", err: "failed"}),
		},
	})

	line := stripANSI(toolGroupStatusLine(entry))
	for _, word := range []string{"succeeded", "failed", "done", "in 3s", "3s"} {
		if strings.Contains(line, word) {
			t.Fatalf("group status = %q, should not show status word or elapsed %q", line, word)
		}
	}
	if line != "Used 3 tools" {
		t.Fatalf("group status = %q, want action summary", line)
	}
}

func TestChatToolGroupStatusShowsAllSuccessCount(t *testing.T) {
	entry := newChatEntry(chatEntry{
		role:   "tool_group",
		label:  "Tool calls (2)",
		status: "done",
		tools: []chatEntry{
			newChatEntry(chatEntry{role: "tool", status: "done"}),
			newChatEntry(chatEntry{role: "tool", status: "done"}),
		},
	})

	line := stripANSI(toolGroupStatusLine(entry))
	if strings.Contains(line, "succeeded") || strings.Contains(line, "done") {
		t.Fatalf("group status = %q, should not show success count or done", line)
	}
	if line != "Used 2 tools" {
		t.Fatalf("group status = %q, want action summary", line)
	}
}

func TestChatToolGroupStatusRecomputesCachedLabel(t *testing.T) {
	entries := groupCompletedToolEntries([]chatEntry{
		newChatEntry(chatEntry{
			role:   "tool_group",
			label:  "Ran 1 command",
			status: "done",
			tools: []chatEntry{
				newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("pwd")`, status: "done"}),
			},
		}),
		newChatEntry(chatEntry{
			role:   "tool",
			detail: "bash",
			label:  `Bash("ls")`,
			status: "done",
		}),
	})

	if len(entries) != 1 || entries[0].role != "tool_group" {
		t.Fatalf("entries = %#v, want regrouped tool group", entries)
	}
	if line := stripANSI(toolGroupStatusLine(entries[0])); line != "Ran 2 commands" {
		t.Fatalf("group status = %q, want recomputed action summary", line)
	}
}

func TestChatToolGroupStatusKeepsHiddenDetectedCount(t *testing.T) {
	args := map[string]any{"command": "ls"}
	entry := newChatEntry(chatEntry{
		role:   "tool_group",
		label:  "Ran 2 commands",
		status: "done",
		tools: []chatEntry{
			newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("pwd")`, status: "done"}),
		},
	})
	entry.tools[0].args = map[string]any{"command": "pwd"}
	detected := []chatEntry{
		newChatEntry(chatEntry{role: "tool", detail: "bash", label: `Bash("ls")`, status: "queued", args: args}),
	}

	grouped := groupCompletedToolEntries([]chatEntry{entry}, detected...)
	if len(grouped) != 1 || grouped[0].role != "tool_group" || len(grouped[0].tools) != 1 {
		t.Fatalf("grouped entries = %#v, want visible finished child only", grouped)
	}
	if line := stripANSI(toolGroupStatusLine(grouped[0])); line != "Ran 2 commands" {
		t.Fatalf("group status = %q, want hidden detected command counted", line)
	}
}

func TestChatToolOutputRendersUnifiedDiff(t *testing.T) {
	diff := strings.Join([]string{
		"diff --git a/file.go b/file.go",
		"index 1111111..2222222 100644",
		"--- a/file.go",
		"+++ b/file.go",
		"@@ -1,3 +1,3 @@",
		" package main",
		"-var old = true",
		"+var newer = true",
	}, "\n")
	m := chatModel{
		entries: []chatEntry{{
			role:    "tool",
			detail:  "bash",
			label:   "Bash(\"git diff\")",
			status:  "done",
			content: diff,
		}},
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	rendered := m.renderTranscript(100)
	body := stripANSI(rendered)
	if !strings.Contains(body, "diff --git a/file.go b/file.go") ||
		!strings.Contains(body, "-var old = true") ||
		!strings.Contains(body, "+var newer = true") {
		t.Fatalf("rendered diff missing expected lines: %q", body)
	}
	if !looksLikeUnifiedDiff(diff) {
		t.Fatal("diff output should be detected as a unified diff")
	}
}

func TestChatToolCallRendersPrettyInvocationAndResult(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("query", "Parth Sareen Ollama software engineer")

	m := chatModel{width: 100, height: 30}
	m.applyAgentEvent(coreagent.Event{
		Type: coreagent.EventToolCallDetected,
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "web_search",
				Arguments: args,
			},
		}},
	})
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "web_search",
		Args:       args.ToMap(),
	})
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "web_search",
		Args:       args.ToMap(),
		Content:    "**Search results for:** Parth Sareen\n\n1. Parth Sareen\n   URL: https://parthsareen.com\n",
	})

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "Web Search(\"Parth Sareen Ollama software engineer\")") {
		t.Fatalf("transcript missing invocation: %q", transcript)
	}
	for _, word := range []string{"done", "in 6s", "6s"} {
		if strings.Contains(transcript, word) {
			t.Fatalf("transcript should not include status word/elapsed %q: %q", word, transcript)
		}
	}
	if strings.Contains(transcript, "https://parthsareen.com") || strings.Contains(transcript, "Search results for:") {
		t.Fatalf("tool output should be collapsed by default: %q", transcript)
	}
	if strings.Contains(transcript, "web_search") {
		t.Fatalf("transcript should use display name instead of raw tool name: %q", transcript)
	}
	if m.entries[0].status != "done" {
		t.Fatalf("tool status = %q, want done", m.entries[0].status)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	view := stripANSI(m.renderTranscript(100))
	if !strings.Contains(view, "**Search results for:**") || !strings.Contains(view, "https://parthsareen.com") {
		t.Fatalf("inline web output missing content: %q", view)
	}
}

func TestChatToolOutputHidesInternalTruncationMarker(t *testing.T) {
	entry := chatEntry{
		role:     "tool",
		detail:   "bash",
		status:   "done",
		expanded: true,
		content:  "head\n\n[tool output truncated: showing first ~10 tokens and last ~10 tokens; omitted ~25 tokens. Use a narrower command, line range, or search query if more detail is needed.]\n\ntail",
	}

	rendered := stripANSI(strings.Join(renderToolResultLines(entry, 80), "\n"))
	if strings.Contains(rendered, "tool output truncated") || strings.Contains(rendered, "omitted ~25 tokens") {
		t.Fatalf("internal truncation marker should be hidden from tool output: %q", rendered)
	}
	if !strings.Contains(rendered, "head") || !strings.Contains(rendered, "tail") {
		t.Fatalf("tool output should keep visible content: %q", rendered)
	}
}

func TestChatHistoryHidesInternalToolTruncationMarker(t *testing.T) {
	rendered := stripANSI(strings.Join(renderHistoryMessages([]api.Message{{
		Role:       "tool",
		ToolName:   "bash",
		ToolCallID: "call-1",
		Content:    "head\n\n[tool output truncated: showing first ~10 tokens and last ~10 tokens; omitted ~25 tokens. Use a narrower command, line range, or search query if more detail is needed.]\n\ntail",
	}}, 80), "\n"))
	if strings.Contains(rendered, "tool output truncated") || strings.Contains(rendered, "omitted ~25 tokens") {
		t.Fatalf("internal truncation marker should be hidden from history: %q", rendered)
	}
	if !strings.Contains(rendered, "head") || !strings.Contains(rendered, "tail") {
		t.Fatalf("history should keep visible content: %q", rendered)
	}
}

func TestChatRenderTranscriptCachesEntryLinesUntilDirty(t *testing.T) {
	m := chatModel{
		entries: []chatEntry{
			newChatEntry(chatEntry{role: "assistant", content: "**hello**"}),
		},
	}

	_ = m.renderTranscript(80)
	if len(m.entries[0].renderLines) == 0 {
		t.Fatal("rendered entry lines should be cached")
	}

	m.entries[0].renderLines = []string{"cached line"}
	if got := stripANSI(m.renderTranscript(80)); !strings.Contains(got, "cached line") {
		t.Fatalf("render should reuse cached lines for unchanged entry: %q", got)
	}

	m.entries[0].content = "**changed**"
	m.markEntryDirty(0)
	if got := stripANSI(m.renderTranscript(80)); strings.Contains(got, "cached line") || !strings.Contains(got, "changed") {
		t.Fatalf("dirty entry should re-render instead of using stale cache: %q", got)
	}
}

func TestWrapChatTextSplitsLongLines(t *testing.T) {
	lines := wrapChatText("alpha beta gamma delta", 12)
	if len(lines) < 2 {
		t.Fatalf("lines = %#v, want split text", lines)
	}
	if strings.Contains(lines[0], "delta") {
		t.Fatalf("first line was not wrapped: %#v", lines)
	}
}

func TestWrapChatTextUsesDisplayWidth(t *testing.T) {
	lines := wrapChatText(strings.Repeat("界", 20), 20)
	if len(lines) < 2 {
		t.Fatalf("lines = %#v, want full-width text split", lines)
	}
	for _, line := range lines {
		if got := lipgloss.Width(line); got > 20 {
			t.Fatalf("line %q width = %d, want <= 20", line, got)
		}
	}
}

func TestRenderMarkdownTableWrapsLongCells(t *testing.T) {
	markdown := strings.Join([]string{
		"| # | Item | Why it matters |",
		"|---|---|---|",
		"| **C** | **Bash filesystem/network confinement** | Biggest asymmetry: file tools are sandboxed via `os.Root`, bash is not and this text should survive until tail-token. |",
	}, "\n")

	rendered := renderMarkdownForView(markdown, 72)
	plain := stripANSI(rendered)
	if !strings.Contains(plain, "tail-token") {
		t.Fatalf("long table cell was truncated:\n%s", plain)
	}
	if strings.Contains(plain, "tail-toke...") {
		t.Fatalf("long table cell should wrap, not ellipsize:\n%s", plain)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if got := lipgloss.Width(line); got > 72 {
			t.Fatalf("rendered table line width = %d, want <= 72: %q\n%s", got, stripANSI(line), plain)
		}
	}
}
