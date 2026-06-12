package tui

import (
	"os"
	"path/filepath"

	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

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
	if !strings.Contains(transcript, "> hello") {
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

func TestChatViewRendersInputBox(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, strings.Repeat("─", 40)) {
		t.Fatalf("view missing input border: %q", view)
	}
	if !strings.Contains(view, "> hello█") {
		t.Fatalf("view missing prompt input row: %q", view)
	}
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
	if !strings.Contains(view, "> next█") {
		t.Fatalf("running view should keep input row: %q", view)
	}
	if !strings.Contains(view, "enter queue") {
		t.Fatalf("running footer should describe queue action: %q", view)
	}
	if strings.Contains(view, "↑/↓ scroll") || strings.Contains(view, "/new chat") || strings.Contains(view, "/clear reset") {
		t.Fatalf("footer should not include scroll/new/clear hints: %q", view)
	}
	lines := strings.Split(view, "\n")
	for i, line := range lines {
		if strings.Contains(line, "> next█") {
			if i < 2 || !strings.Contains(lines[i-2], "thinking 42 tokens") || !strings.Contains(lines[i-1], strings.Repeat("─", 40)) {
				t.Fatalf("thinking line should sit directly above input border:\n%s", view)
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

	view := stripANSI(m.View())
	if !strings.Contains(view, "cwd ./sub") {
		t.Fatalf("view missing cwd status: %q", view)
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
	if view := stripANSI(m.View()); !strings.Contains(view, "cwd ./sub") {
		t.Fatalf("view missing cwd status: %q", view)
	}
}

func TestChatViewPadsToTerminalHeight(t *testing.T) {
	m := chatModel{
		input:  []rune("hello"),
		width:  40,
		height: 12,
	}

	view := m.View()
	if got := len(strings.Split(view, "\n")); got != 12 {
		t.Fatalf("view height = %d, want 12:\n%s", got, stripANSI(view))
	}
	if !strings.Contains(stripANSI(view), "> hello█") {
		t.Fatalf("view missing input row: %q", stripANSI(view))
	}
}

func TestChatScrollsTranscript(t *testing.T) {
	m := chatModel{
		width:  80,
		height: 10,
	}
	for i := 0; i < 12; i++ {
		m.entries = append(m.entries, chatEntry{role: "user", content: "line"})
	}

	if m.maxScroll() == 0 {
		t.Fatal("test setup should produce scrollable transcript")
	}
	if got := m.scrollStatus(); got != "↑ more" {
		t.Fatalf("scrollStatus = %q, want ↑ more", got)
	}

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	m = updated.(chatModel)
	if m.scroll != 1 {
		t.Fatalf("scroll = %d, want 1", m.scroll)
	}
	if got := m.scrollStatus(); got != "↑/↓ more" {
		t.Fatalf("scrollStatus = %q, want ↑/↓ more", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("scroll = %d, want 0", m.scroll)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyHome})
	m = updated.(chatModel)
	if m.scroll != m.maxScroll() {
		t.Fatalf("scroll = %d, want max %d", m.scroll, m.maxScroll())
	}
	if got := m.scrollStatus(); got != "↓ more" {
		t.Fatalf("scrollStatus = %q, want ↓ more", got)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyEnd})
	m = updated.(chatModel)
	if m.scroll != 0 {
		t.Fatalf("scroll = %d, want 0", m.scroll)
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
		{Role: "user", Content: "Conversation summary:\n- old work\n- decisions"},
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
	transcript = stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "old work") || !strings.Contains(transcript, "decisions") {
		t.Fatalf("expanded summary body missing: %q", transcript)
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
	if got := m.activityLabel(); got != "using Bash(\"pwd\")" {
		t.Fatalf("activityLabel = %q, want active tool name", got)
	}
}

func TestChatToolOutputIsHiddenUntilExpanded(t *testing.T) {
	fullOutput := strings.Repeat("line\n", 25)
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

	body = stripANSI(m.renderTranscript(100))
	if got := strings.Count(body, "line"); got != 25 {
		t.Fatalf("expanded body rendered %d output lines, want 25: %q", got, body)
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

func TestChatCtrlOTogglesAllToolOutputs(t *testing.T) {
	m := chatModel{
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
			t.Fatalf("tool entry %d was not expanded", index)
		}
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	for _, index := range []int{0, 2} {
		if m.entries[index].expanded {
			t.Fatalf("tool entry %d was not collapsed", index)
		}
	}
}

func TestChatCtrlOLatchesForRunningToolOutput(t *testing.T) {
	args := map[string]any{"command": "pwd"}
	m := chatModel{running: true}
	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolStarted,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
	})

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	m = updated.(chatModel)
	if !m.toolOutputMode || !m.toolOutputOpen {
		t.Fatalf("ctrl+o should latch tool output open while tool is running")
	}
	if !m.entries[0].expanded {
		t.Fatalf("running tool entry should record expanded state")
	}

	m.applyAgentEvent(coreagent.Event{
		Type:       coreagent.EventToolFinished,
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       args,
		Content:    "/tmp/project\n",
	})

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "/tmp/project") {
		t.Fatalf("finished tool output should remain expanded after latched ctrl+o: %q", transcript)
	}
}

func TestChatCtrlOExpansionSurvivesToolGrouping(t *testing.T) {
	firstArgs := map[string]any{"command": "pwd"}
	secondArgs := map[string]any{"command": "ls"}
	m := chatModel{
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
		t.Fatalf("grouped tool history should preserve expanded state: %#v", m.entries[0])
	}

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "one") || !strings.Contains(transcript, "two") {
		t.Fatalf("expanded grouped tool output should remain visible: %q", transcript)
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
	transcript = stripANSI(m.renderTranscript(100))
	if strings.Contains(transcript, "**Search results for:**") {
		t.Fatalf("expanded web output should render markdown: %q", transcript)
	}
	if !strings.Contains(transcript, "Search results for:") || !strings.Contains(transcript, "https://parthsareen.com") {
		t.Fatalf("expanded web output missing content: %q", transcript)
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
