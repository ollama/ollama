package tui

import (
	"context"
	"errors"
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

func TestChatStatusLineIsCompact(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})

	m := chatModel{
		chatID: "12345678-1234-1234-1234-123456789abc",
		opts:   ChatOptions{Tools: registry},
	}

	status := m.statusLine()
	if status != "" {
		t.Fatalf("statusLine = %q, want empty status", status)
	}
	if strings.Contains(status, "fake_tool") || strings.Contains(status, m.chatID) {
		t.Fatalf("statusLine should not include tool names or chat id: %q", status)
	}
}

func TestChatAssistantEntryUsesBullet(t *testing.T) {
	m := chatModel{}

	prefix, _ := m.renderEntry(chatEntry{role: "assistant", content: "hello"})

	if strings.Contains(prefix, "Ollama:") {
		t.Fatalf("prefix should not include Ollama label: %q", prefix)
	}
	if !strings.Contains(prefix, "●") {
		t.Fatalf("prefix = %q, want bullet", prefix)
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
	hintLine := lineIndexContaining(lines, `› Try: "`)
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
	if !strings.Contains(transcript, "› hello") {
		t.Fatalf("user transcript should render as prompt row: %q", transcript)
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

func TestChatMixedToolGroupUsesSuccessPrefix(t *testing.T) {
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

	if prefix != chatToolDoneStyle.Render("●")+" " {
		t.Fatalf("prefix = %q, want success-styled bullet", prefix)
	}
	if !strings.Contains(stripANSI(body), "1 succeeded, 1 failed") {
		t.Fatalf("body = %q, want mixed result counts", stripANSI(body))
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
	if !strings.Contains(line, `Web Search("who is parth sareen") done in 812ms`) {
		t.Fatalf("tool status line = %q", line)
	}
}

func TestChatViewRendersInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
	}

	view := stripANSI(m.View())
	if strings.Contains(view, inputBoxTopBorderLine(40)) || strings.Contains(view, inputBoxBottomBorderLine(40)) {
		t.Fatalf("prompt input should not render box borders: %q", view)
	}
	if !strings.Contains(view, "› hello█") {
		t.Fatalf("view missing prompt input row: %q", view)
	}
}

func TestChatViewRendersModelUnderInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  48,
		height: 12,
		opts: ChatOptions{
			Model: "kimi-k2.7-code:cloud",
		},
	}

	lines := strings.Split(stripANSI(m.View()), "\n")
	inputLine := lineIndexContaining(lines, "› hello█")
	modelLine := lineIndexContaining(lines, "kimi-k2.7-code:cloud")
	if inputLine < 0 || modelLine < 0 {
		t.Fatalf("view missing input or model line:\n%s", strings.Join(lines, "\n"))
	}
	if modelLine != inputLine+1 {
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
	if !strings.Contains(view, "... ") || strings.Contains(view, "... ... ") {
		t.Fatalf("truncated pasted prompt should show an omission marker:\n%s", view)
	}
}

func TestChatViewWrapsNotificationWhenNarrow(t *testing.T) {
	m := chatModel{
		input:          []rune("hello"),
		width:          28,
		height:         14,
		status:         "cache will break by turning system prompt off",
		permissionMode: newChatPermissionMode(true),
		opts: ChatOptions{
			Verbose:             true,
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
	if strings.Contains(m.footerLine(), "cache will break") {
		t.Fatalf("footer should not include transient notification: %q", m.footerLine())
	}
	for _, line := range strings.Split(view, "\n") {
		if len([]rune(line)) > 28 {
			t.Fatalf("line width = %d, want <= 28: %q\n%s", len([]rune(line)), line, view)
		}
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
	if strings.Contains(m.footerLine(), "copied latest output") {
		t.Fatalf("footer should not include notification: %q", m.footerLine())
	}
	lines := strings.Split(view, "\n")
	for i, line := range lines {
		if strings.Contains(line, "› hello█") {
			if i < 1 || !strings.Contains(lines[i-1], "copied latest output") {
				t.Fatalf("notification should sit directly above input:\n%s", view)
			}
			return
		}
	}
	t.Fatalf("view missing input row:\n%s", view)
}

func TestChatViewDoesNotRenderQueueStatusAsNotification(t *testing.T) {
	m := chatModel{
		input:  []rune("next"),
		queued: []string{"next"},
		width:  40,
		height: 12,
		status: "queued",
	}

	if got := m.notificationLine(); got != "" {
		t.Fatalf("notificationLine = %q, want empty", got)
	}
	if footer := m.footerLine(); !strings.Contains(footer, "queued 1") {
		t.Fatalf("queued count should still render in footer: %q", footer)
	}
}

func inputPromptLineCount(t *testing.T, view string) int {
	t.Helper()
	count := 0
	for _, line := range strings.Split(view, "\n") {
		if strings.HasPrefix(line, "› ") || strings.HasPrefix(line, "... ") {
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
	if !strings.Contains(view, "› next█") {
		t.Fatalf("running view should keep input row: %q", view)
	}
	if strings.Contains(view, "↑/↓ scroll") || strings.Contains(view, "/new chat") || strings.Contains(view, "/clear reset") {
		t.Fatalf("footer should not include scroll/new/clear hints: %q", view)
	}
	lines := strings.Split(view, "\n")
	for i, line := range lines {
		if strings.Contains(line, "› next█") {
			if i < 1 || !strings.Contains(lines[i-1], "Thinking 42 tokens") {
				t.Fatalf("thinking line should sit directly above input:\n%s", view)
			}
			return
		}
	}
	t.Fatalf("view missing input row: %q", view)
}

func TestChatViewShowsSessionCWDWhenChanged(t *testing.T) {
	root := t.TempDir()
	subdir := filepath.Join(root, "sub")
	if err := os.Mkdir(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	m := chatModel{
		width:  140,
		height: 12,
		opts: ChatOptions{
			RootDir:    root,
			WorkingDir: subdir,
		},
	}

	if footer := m.footerLine(); !strings.Contains(footer, "cwd ./sub") {
		t.Fatalf("footer metadata missing cwd status: %q", footer)
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
		opts: ChatOptions{
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
	if footer := m.footerLine(); !strings.Contains(footer, "cwd ./sub") {
		t.Fatalf("footer metadata missing cwd status: %q", footer)
	}
}

func TestChatViewDoesNotPadToTerminalHeight(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
	}

	view := m.View()
	if got := len(strings.Split(view, "\n")); got >= 12 {
		t.Fatalf("view height = %d, want less than terminal height:\n%s", got, stripANSI(view))
	}
	if !strings.Contains(stripANSI(view), "› hello█") {
		t.Fatalf("view missing input row: %q", stripANSI(view))
	}
}

func TestChatInputFlowsDownAsTranscriptGrows(t *testing.T) {
	short := chatModel{
		input:  []rune("next"),
		width:  60,
		height: 14,
		entries: []chatEntry{
			{role: "assistant", content: "hello"},
		},
	}
	shortInputLine := renderedInputLine(short.View())
	if shortInputLine < 0 {
		t.Fatalf("short view missing input:\n%s", stripANSI(short.View()))
	}

	long := chatModel{
		input:  []rune("next"),
		width:  short.width,
		height: short.height,
	}
	for i := range 20 {
		long.entries = append(long.entries, chatEntry{role: "assistant", content: fmt.Sprintf("line %02d", i)})
	}
	longInputLine := renderedInputLine(long.View())
	if longInputLine < 0 {
		t.Fatalf("long view missing input:\n%s", stripANSI(long.View()))
	}
	if shortInputLine >= longInputLine {
		t.Fatalf("input line did not move down as transcript grew: short=%d long=%d\nshort:\n%s\nlong:\n%s", shortInputLine, longInputLine, stripANSI(short.View()), stripANSI(long.View()))
	}
	if longInputLine <= long.height {
		t.Fatalf("normal terminal flow should grow past terminal height, input line=%d height=%d\n%s", longInputLine, long.height, stripANSI(long.View()))
	}
}

func TestChatFlowFlushKeepsFirstRunLayoutStable(t *testing.T) {
	empty := chatModel{
		input:  []rune("next"),
		width:  80,
		height: 16,
	}
	emptyInputLine := renderedInputLine(empty.View())
	if emptyInputLine < 0 {
		t.Fatalf("empty view missing input:\n%s", stripANSI(empty.View()))
	}

	m := chatModel{
		input:   []rune("next"),
		width:   80,
		height:  16,
		running: true,
		entries: []chatEntry{
			{role: "user", content: "first prompt"},
		},
	}

	updated, cmd := m.flowTranscriptFlushCmd()
	if cmd == nil {
		t.Fatal("submitted user prompt should flush into terminal scrollback")
	}
	if updated.flowPrintedLines == 0 {
		t.Fatal("submitted user prompt should not stay in the managed input frame")
	}
	if strings.Contains(stripANSI(updated.View()), "first prompt") {
		t.Fatalf("managed frame should not keep flushed user prompt:\n%s", stripANSI(updated.View()))
	}
	if inputLine := renderedInputLine(updated.View()); inputLine != emptyInputLine {
		t.Fatalf("input line moved after first prompt flush: empty=%d flushed=%d\nempty:\n%s\nflushed:\n%s", emptyInputLine, inputLine, stripANSI(empty.View()), stripANSI(updated.View()))
	}

	updated.entries = append(updated.entries, chatEntry{role: "tool", label: "Bash(\"pwd\")", status: "running"})
	withTool, _ := updated.flowTranscriptFlushCmd()
	if !strings.Contains(stripANSI(withTool.View()), `Bash("pwd")`) {
		t.Fatalf("active tool row should remain visible:\n%s", stripANSI(withTool.View()))
	}

	withTool.entries[len(withTool.entries)-1].status = "done"
	doneTool, cmd := withTool.flowTranscriptFlushCmd()
	if cmd == nil {
		t.Fatal("finished tool row should flush once it no longer needs live updates")
	}
	if strings.Contains(stripANSI(doneTool.View()), `Bash("pwd")`) {
		t.Fatalf("managed frame should not keep finished tool row:\n%s", stripANSI(doneTool.View()))
	}
}

func TestChatBoundedFramePinsInputAfterResize(t *testing.T) {
	m := chatModel{
		input:        []rune("next"),
		width:        60,
		height:       14,
		boundedFrame: true,
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
	if gap := activityLine - assistantLine - 1; gap < 2 {
		t.Fatalf("gap between transcript and action status = %d, want at least 2:\n%s", gap, strings.Join(lines, "\n"))
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
	inputLine := lineIndexContaining(lines, "› next")
	if assistantLine < 0 || inputLine < 0 {
		t.Fatalf("view missing assistant/input lines:\n%s", strings.Join(lines, "\n"))
	}
	if gap := inputLine - assistantLine - 1; gap != 3 {
		t.Fatalf("gap between finished response and input body = %d, want 3 including stable action slot:\n%s", gap, strings.Join(lines, "\n"))
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
	inputLine := lineIndexContaining(lines, "› next")
	if hintLine >= 0 {
		t.Fatalf("view should not show empty hint while typing:\n%s", strings.Join(lines, "\n"))
	}
	if inputLine < 0 {
		t.Fatalf("view missing input line:\n%s", strings.Join(lines, "\n"))
	}
}

func renderedInputLine(view string) int {
	for i, line := range strings.Split(stripANSI(view), "\n") {
		if strings.Contains(line, "› next") {
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
		width:        80,
		height:       10,
		boundedFrame: true,
	}
	for range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: "line"})
	}

	if m.maxScroll() == 0 {
		t.Fatal("test setup should produce scrollable transcript")
	}
	if got := m.scrollStatus(); got != "↑ more" {
		t.Fatalf("scrollStatus = %q, want ↑ more", got)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyPgUp})
	m = updated.(chatModel)
	if m.scroll == 0 {
		t.Fatal("page up should scroll transcript")
	}
	if got := m.scrollStatus(); got != "↑/↓ more" {
		t.Fatalf("scrollStatus = %q, want ↑/↓ more", got)
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
	if got := m.scrollStatus(); got != "↓ more" {
		t.Fatalf("scrollStatus = %q, want ↓ more", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlEnd})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("scroll = %d, want 0", m.scroll)
	}
}

func TestChatResizeAndScrollsLongAssistantOutput(t *testing.T) {
	var generated []string
	for i := range 80 {
		generated = append(generated, fmt.Sprintf("generated line %02d", i))
	}
	m := chatModel{
		input: []rune("next"),
		entries: []chatEntry{{
			role:    "assistant",
			content: strings.Join(generated, "\n\n"),
		}},
	}

	updated, cmd := m.Update(tea.WindowSizeMsg{Width: 80, Height: 12})
	m = updated.(chatModel)
	if cmd == nil || m.boundedFrame {
		t.Fatal("initial terminal size should keep normal terminal-flow rendering")
	}
	if m.flowPrintedLines == 0 {
		t.Fatal("initial terminal-flow render should print transcript lines into scrollback")
	}
	if strings.Contains(stripANSI(m.View()), "generated line 79") {
		t.Fatalf("printed transcript should not stay in the managed input frame:\n%s", stripANSI(m.View()))
	}

	updated, cmd = m.Update(tea.WindowSizeMsg{Width: 72, Height: 10})
	m = updated.(chatModel)
	if cmd == nil || !m.boundedFrame {
		t.Fatal("terminal resize should switch to bounded rendering and clear the stale flow view")
	}
	if !m.fullScreen {
		t.Fatal("terminal resize should enter fullscreen managed rendering")
	}
	if m.flowPrintedLines != 0 {
		t.Fatalf("resize should clear flow state, printed=%d", m.flowPrintedLines)
	}
	if m.maxScroll() == 0 {
		t.Fatal("long assistant output should be scrollable after resize")
	}
	if !strings.Contains(stripANSI(m.View()), "generated line 00") {
		t.Fatalf("bounded view should reset to earliest generated content after resize:\n%s", stripANSI(m.View()))
	}
	assertChatFrameSize(t, m.View(), 72, 10)

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlHome})
	m = updated.(chatModel)
	if !strings.Contains(stripANSI(m.View()), "generated line 00") {
		t.Fatalf("scrolling to top should show earliest generated content:\n%s", stripANSI(m.View()))
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlEnd})
	m = updated.(chatModel)
	if !strings.Contains(stripANSI(m.View()), "generated line 79") {
		t.Fatalf("scrolling back to bottom should restore latest generated content:\n%s", stripANSI(m.View()))
	}
	assertChatFrameSize(t, m.View(), 72, 10)
}

func assertChatFrameSize(t *testing.T, view string, width, height int) {
	t.Helper()
	lines := strings.Split(view, "\n")
	if len(lines) != height {
		t.Fatalf("frame rendered %d lines, want %d:\n%s", len(lines), height, stripANSI(view))
	}
	for i, line := range lines {
		if got := lipgloss.Width(line); got > width {
			t.Fatalf("frame line %d width = %d, want <= %d: %q", i, got, width, stripANSI(line))
		}
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
	if m.flowPrintedLines != 0 {
		t.Fatalf("single live line should stay managed, printed=%d", m.flowPrintedLines)
	}
	if !strings.Contains(stripANSI(m.View()), "generated line 00") {
		t.Fatalf("live output should be visible in managed frame:\n%s", stripANSI(m.View()))
	}

	updated, _ = m.Update(chatAgentMsg{event: coreagent.Event{Type: coreagent.EventMessageDelta, Content: "\n\ngenerated line 01"}})
	m = updated.(chatModel)
	if m.flowPrintedLines != 0 {
		t.Fatalf("live assistant markdown should stay managed until the message is complete, printed=%d", m.flowPrintedLines)
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, "generated line 00") {
		t.Fatalf("live assistant output should not be frozen into scrollback while streaming:\n%s", view)
	}
	if !strings.Contains(view, "generated line 01") {
		t.Fatalf("latest generated line should remain visible:\n%s", view)
	}

	m.running = false
	updated, cmd := m.flowTranscriptFlushCmd()
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("completed assistant output should flush into terminal scrollback")
	}
	if m.flowPrintedLines == 0 {
		t.Fatal("completed assistant output should be marked as flushed")
	}
}

func TestChatMouseWheelScrollsTranscriptWhileRunning(t *testing.T) {
	m := chatModel{
		width:         80,
		height:        10,
		boundedFrame:  true,
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

func TestChatMouseDragSelectsAndCopiesTranscriptText(t *testing.T) {
	var copied string
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Clipboard: func(_ context.Context, text string) error {
				copied = text
				return nil
			},
		},
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

	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: 7, Y: top})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("mouse release should return clipboard command")
	}
	if msg := cmd(); msg != nil {
		_, _ = m.Update(msg)
	}
	if copied != "alpha" {
		t.Fatalf("copied = %q, want alpha", copied)
	}
	if m.status == "selection copied" {
		t.Fatalf("selection should not surface a copied status")
	}
}

func TestChatBoundedViewHeaderMatchesTranscriptLayout(t *testing.T) {
	m := chatModel{
		width:        80,
		height:       6,
		boundedFrame: true,
	}
	for i := range 12 {
		m.entries = append(m.entries, chatEntry{role: "user", content: fmt.Sprintf("line-%02d", i)})
	}
	m.scroll = m.maxScroll()
	top, _ := m.transcriptLayout()
	lines := strings.Split(stripANSI(m.View()), "\n")
	if top <= 0 {
		t.Fatalf("transcript top = %d, want header offset", top)
	}
	if !strings.Contains(lines[0], "↓ more") {
		t.Fatalf("view should render status header at top: %q", lines[0])
	}
	if strings.TrimSpace(lines[top]) == "" {
		t.Fatalf("transcript should start at layout top %d, line=%q view=%q", top, lines[top], strings.Join(lines, "\n"))
	}
}

func TestChatMouseCopyFailureDoesNotClearRunningState(t *testing.T) {
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Clipboard: func(context.Context, string) error {
				return errors.New("copy failed")
			},
		},
		width:   80,
		height:  10,
		running: true,
		entries: []chatEntry{
			{role: "user", content: "alpha beta"},
		},
	}
	top, _ := m.transcriptLayout()

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: 2, Y: top})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: 7, Y: top})
	m = updated.(chatModel)
	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: 7, Y: top})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("mouse release should return clipboard command")
	}
	msg := cmd()
	updated, _ = m.Update(msg)
	m = updated.(chatModel)
	if !m.running {
		t.Fatal("clipboard failure should not clear running state")
	}
	if m.status != "clipboard error: copy failed" {
		t.Fatalf("status = %q, want clipboard error", m.status)
	}
}

func TestChatMouseDragSelectionUsesScrolledTranscriptCoordinates(t *testing.T) {
	var copied string
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Clipboard: func(_ context.Context, text string) error {
				copied = text
				return nil
			},
		},
		width:        80,
		height:       8,
		boundedFrame: true,
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
	fm := updated.(chatModel)
	if cmd == nil {
		t.Fatal("mouse release should return clipboard command")
	}
	if msg := cmd(); msg != nil {
		_, _ = fm.Update(msg)
	}
	if copied != "line-00" {
		t.Fatalf("copied = %q, want line-00", copied)
	}
}

func TestChatArrowKeysNavigatePromptHistoryWhenTranscriptScrollable(t *testing.T) {
	m := chatModel{
		input:         []rune("queued draft"),
		promptHistory: []string{"old prompt"},
		width:         80,
		height:        10,
		boundedFrame:  true,
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
	if got := string(m.input); got != "queued draft" {
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

	transcript := stripANSI((chatModel{entries: entries}).renderTranscript(120))
	if strings.Contains(transcript, "● \n") || strings.Contains(transcript, "●\n") {
		t.Fatalf("transcript has blank assistant bullet: %q", transcript)
	}
}

func TestEntriesFromMessagesRendersCompactionSummaryCollapsed(t *testing.T) {
	entries := entriesFromMessages([]api.Message{
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: chatCompactionToolCallID,
			Function: api.ToolCallFunction{
				Name: chatCompactionToolName,
			},
		}}},
		{Role: "tool", ToolName: chatCompactionToolName, ToolCallID: chatCompactionToolCallID, Content: "Conversation summary:\n- old work\n- decisions"},
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
	if !strings.Contains(transcript, "Compacted summary done") {
		t.Fatalf("collapsed summary row missing: %q", transcript)
	}
	if strings.Contains(transcript, "old work") {
		t.Fatalf("summary body should be collapsed: %q", transcript)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	view := stripANSI(m.View())
	if !strings.Contains(view, "old work") || !strings.Contains(view, "decisions") {
		t.Fatalf("inline summary body missing: %q", view)
	}
}

func TestEntriesFromMessagesHidesAutomaticCompactionInstruction(t *testing.T) {
	entries := entriesFromMessages([]api.Message{
		{
			Role:       "tool",
			ToolName:   chatCompactionToolName,
			ToolCallID: chatCompactionToolCallID,
			Content:    "Conversation summary:\nold work summary\n\ncontinue the task in progress. the history has been compacted, do not mention compaction to the user",
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
	var outputLines []string
	for i := range 25 {
		outputLines = append(outputLines, fmt.Sprintf("line %02d", i))
	}
	fullOutput := strings.Join(outputLines, "\n")
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
	if strings.Contains(body, "line") {
		t.Fatalf("collapsed tool output should be hidden: %q", body)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)

	if !m.toolOutputMode || !m.toolOutputOpen || !m.entries[0].expanded {
		t.Fatalf("ctrl+o should expand tool output inline: %#v", m.entries[0])
	}
	body = stripANSI(m.renderTranscript(100))
	if !strings.Contains(body, "line 00") || !strings.Contains(body, "line 24") {
		t.Fatalf("expanded transcript should show full tool output: %q", body)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	body = stripANSI(m.renderTranscript(100))
	if m.toolOutputOpen || m.entries[0].expanded || strings.Contains(body, "line 00") {
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
	if m.entries[1].status != "running" || m.entries[1].label != "Bash(\"date\")" {
		t.Fatalf("second entry should be active tool: %#v", m.entries[1])
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
	view := stripANSI(m.View())
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

func TestChatCtrlOTogglesInlineOutputWithoutLeavingFullscreen(t *testing.T) {
	m := chatModel{
		width:            100,
		height:           24,
		boundedFrame:     true,
		fullScreen:       true,
		flowPrintedLines: 4,
		entries: []chatEntry{
			{role: "user", content: "who is parth sareen"},
			{role: "tool", detail: "web_search", label: "Web Search(\"who is Parth Sareen\")", status: "done", content: "Search results"},
			{role: "assistant", content: "Based on public search results, Parth Sareen works on AI tooling."},
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("inline tool toggle should not switch screens")
	}
	if !m.fullScreen || !m.boundedFrame {
		t.Fatal("inline tool toggle should keep managed fullscreen mode")
	}
	if !m.toolOutputOpen || !m.entries[1].expanded {
		t.Fatalf("tool output should be expanded inline: %#v", m.entries[1])
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if !m.boundedFrame {
		t.Fatal("inline tool toggle should keep managed redraw mode")
	}
	if !m.fullScreen {
		t.Fatal("inline tool toggle should stay fullscreen")
	}
	if cmd != nil {
		t.Fatal("inline tool toggle should not switch screens")
	}
	if m.toolOutputOpen || m.entries[1].expanded {
		t.Fatalf("tool output should be collapsed inline: %#v", m.entries[1])
	}
}

func TestChatCtrlOShowsRunningToolOutputInline(t *testing.T) {
	args := map[string]any{"command": "pwd"}
	m := chatModel{width: 100, height: 20, running: true}
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

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
		Content:    "/tmp/project\n",
	})

	view := stripANSI(m.renderTranscript(100))
	if !strings.Contains(view, "/tmp/project") {
		t.Fatalf("finished tool output should be visible inline: %q", view)
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

	view := stripANSI(m.View())
	if !strings.Contains(view, "one") || !strings.Contains(view, "two") {
		t.Fatalf("grouped tool output should be visible inline: %q", view)
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
		"    Web Search(\"Parth Sareen\") done in 823ms",
		"      Search results for: Parth Sareen",
		"  ",
		"    Read(\"feedback.md\") done in 2s",
		"      Looks good.",
	}, "\n")
	if !strings.Contains(transcript, expected) {
		t.Fatalf("expanded tool group did not render expected block spacing:\n%s", transcript)
	}
}

func TestChatToolGroupStatusShowsResultCounts(t *testing.T) {
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
	if !strings.Contains(line, "2 succeeded, 1 failed in 3s") {
		t.Fatalf("group status = %q, want result counts", line)
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
	if !strings.Contains(line, "2 succeeded") {
		t.Fatalf("group status = %q, want success count", line)
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
	startedAt := time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC)
	finishedAt := startedAt.Add(6 * time.Second)

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
		StartedAt:  startedAt,
	})
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "web_search",
		Args:       args.ToMap(),
		Content:    "**Search results for:** Parth Sareen\n\n1. Parth Sareen\n   URL: https://parthsareen.com\n",
		FinishedAt: finishedAt,
	})

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "Web Search(\"Parth Sareen Ollama software engineer\")") {
		t.Fatalf("transcript missing invocation: %q", transcript)
	}
	if !strings.Contains(transcript, "done in 6s") {
		t.Fatalf("transcript missing status summary: %q", transcript)
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
	if strings.Contains(view, "**Search results for:**") {
		t.Fatalf("inline web output should render markdown: %q", view)
	}
	if !strings.Contains(view, "Search results for:") || !strings.Contains(view, "https://parthsareen.com") {
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
