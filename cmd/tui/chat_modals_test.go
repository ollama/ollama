package tui

import (
	"context"

	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/chatstore"
	"github.com/ollama/ollama/api"
)

func TestChatHistoryCommandShowsPromptMessages(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	m := chatModel{
		input: []rune("/history"),
		opts:  ChatOptions{SystemPrompt: "You are Ollama."},
		messages: []api.Message{
			{Role: "user", Content: "where am i?"},
			{
				Role:     "assistant",
				Thinking: "Need to inspect cwd.",
				ToolCalls: []api.ToolCall{{
					ID: "call-1",
					Function: api.ToolCallFunction{
						Name:      "bash",
						Arguments: args,
					},
				}},
			},
			{Role: "tool", ToolName: "bash", ToolCallID: "call-1", Content: "/tmp/project\n"},
			{Role: "assistant", Content: "You are in /tmp/project."},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("history command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 0 {
		t.Fatalf("entries = %d, want 0", len(fm.entries))
	}
	if fm.historyPopup == nil {
		t.Fatal("history popup was not opened")
	}
	history := fm.historyPopup.content
	for _, want := range []string{
		"**Message History**",
		"**system**",
		"  content: You are Ollama.",
		"**user**",
		"  content: where am i?",
		"**assistant**",
		"  thinking: Need to inspect cwd.",
		"  tool calls:",
		"`call-1` Bash",
		"      args:",
		"\"command\": \"pwd\"",
		"**tool**",
		"  tool: `bash`",
		"tool call: `call-1`",
		"/tmp/project",
		"  content: You are in /tmp/project.",
	} {
		if !strings.Contains(history, want) {
			t.Fatalf("history missing %q:\n%s", want, history)
		}
	}
	if strings.Contains(history, "###") {
		t.Fatalf("history should not use numbered markdown headings:\n%s", history)
	}

	fm.width = 120
	fm.height = 40
	rendered := stripANSI(fm.View())
	if !strings.Contains(rendered, "Message history") {
		t.Fatalf("history popup missing title:\n%s", rendered)
	}
	if strings.Contains(rendered, "**assistant**") || strings.Contains(rendered, "**system**") {
		t.Fatalf("history roles should render without literal markdown markers:\n%s", rendered)
	}
	if strings.Contains(rendered, "```") {
		t.Fatalf("history renderer should hide code fences:\n%s", rendered)
	}

	updated, cmd = fm.Update(tea.KeyMsg{Type: tea.KeyEsc})
	fm = updated.(chatModel)
	if cmd != nil {
		t.Fatal("closing history should not return a command")
	}
	if fm.historyPopup != nil {
		t.Fatal("history popup should close on escape")
	}
}

func TestChatHistoryCommandHandlesEmptyHistory(t *testing.T) {
	m := chatModel{input: []rune("/history")}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("history command should not return a command")
	}

	fm := updated.(chatModel)
	if len(fm.entries) != 0 || fm.historyPopup == nil || !strings.Contains(fm.historyPopup.content, "No messages yet.") {
		t.Fatalf("history output entries=%#v popup=%#v", fm.entries, fm.historyPopup)
	}
	fm.width = 80
	fm.height = 20
	if view := stripANSI(fm.View()); !strings.Contains(view, "No messages yet.") {
		t.Fatalf("history popup view missing empty state: %q", view)
	}
}

func TestChatHistoryCommandStartsAtBottom(t *testing.T) {
	messages := make([]api.Message, 0, 18)
	for i := 0; i < 18; i++ {
		messages = append(messages, api.Message{Role: "user", Content: "prompt " + strconv.Itoa(i)})
	}
	m := chatModel{
		input:    []rune("/history"),
		messages: messages,
		width:    80,
		height:   10,
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("history command should not return a command")
	}
	fm := updated.(chatModel)
	view := stripANSI(fm.View())
	if !strings.Contains(view, "prompt 17") {
		t.Fatalf("history popup should start at latest messages:\n%s", view)
	}
	if strings.Contains(view, "prompt 0") {
		t.Fatalf("history popup started at oldest messages:\n%s", view)
	}
}

func TestChatHistoryCommandFormatsMultilineContentWithLabel(t *testing.T) {
	m := chatModel{
		input:    []rune("/history"),
		messages: []api.Message{{Role: "assistant", Content: "first\nsecond"}},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("history command should not return a command")
	}

	fm := updated.(chatModel)
	if fm.historyPopup == nil {
		t.Fatal("history popup was not opened")
	}
	history := fm.historyPopup.content
	if !strings.Contains(history, "  content:\n\n  ```text\n  first\n  second\n  ```") {
		t.Fatalf("history should label multiline content before block:\n%s", history)
	}
}

func TestChatResumeCommandOpensPicker(t *testing.T) {
	store := &chatResumeTestStore{
		chats: []chatstore.ChatSummary{{
			ID:           "chat-1",
			Title:        "Research Parth Sareen online",
			Model:        "llama3.2",
			UpdatedAt:    time.Now().Add(-time.Hour),
			MessageCount: 2,
			ApproxBytes:  18 * 1024,
		}},
		byID: map[string]*chatstore.Chat{},
	}
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/resume"),
		width:  100,
		height: 20,
		opts:   ChatOptions{Store: store},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("resume command should not return a command")
	}
	m = updated.(chatModel)
	if m.resumePicker == nil {
		t.Fatal("resume picker was not opened")
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, "Resume session") ||
		!strings.Contains(view, "Search...") ||
		!strings.Contains(view, "Research Parth Sareen online") ||
		!strings.Contains(view, "llama3.2") {
		t.Fatalf("resume picker view missing content: %q", view)
	}
}

func TestChatResumePickerFiltersAndLoadsSelection(t *testing.T) {
	store := &chatResumeTestStore{
		chats: []chatstore.ChatSummary{
			{ID: "chat-1", Title: "First chat", Model: "llama3.2", UpdatedAt: time.Now().Add(-2 * time.Hour), MessageCount: 2, ApproxBytes: 2048},
			{ID: "chat-2", Title: "Second chat", Model: "qwen3", UpdatedAt: time.Now().Add(-time.Hour), MessageCount: 2, ApproxBytes: 4096},
		},
		byID: map[string]*chatstore.Chat{
			"chat-2": {
				ID:    "chat-2",
				Title: "Second chat",
				Model: "qwen3",
				Messages: []api.Message{
					{Role: "user", Content: "resume this"},
					{Role: "assistant", Content: "loaded"},
				},
			},
		},
	}
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/resume"),
		queued: []string{"old queued prompt"},
		width:  100,
		height: 20,
		opts: ChatOptions{
			Model: "llama3.2",
			Store: store,
			ToolRegistryForModel: func(ctx context.Context, model string) *coreagent.Registry {
				if model != "qwen3" {
					t.Fatalf("tool registry model = %q, want qwen3", model)
				}
				registry := coreagent.NewRegistry()
				registry.Register(chatTestTool{})
				return registry
			},
		},
	}

	updated, _ := m.handleSubmit()
	m = updated.(chatModel)
	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("second")})
	m = updated.(chatModel)

	view := stripANSI(m.View())
	if !strings.Contains(view, "Second chat") || strings.Contains(view, "First chat") {
		t.Fatalf("filtered resume picker view = %q", view)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("loading a saved chat should not start a command")
	}
	if m.resumePicker != nil {
		t.Fatal("resume picker should close after selection")
	}
	if m.chatID != "chat-2" {
		t.Fatalf("chatID = %q, want chat-2", m.chatID)
	}
	if m.opts.Model != "qwen3" {
		t.Fatalf("model = %q, want qwen3", m.opts.Model)
	}
	if m.opts.Tools == nil || !m.opts.Tools.Has("fake_tool") {
		t.Fatalf("tools registry was not rebuilt for resumed model: %#v", m.opts.Tools)
	}
	if len(m.queued) != 0 {
		t.Fatalf("queued prompts should be cleared on resume: %#v", m.queued)
	}
	if len(m.messages) != 2 || m.messages[0].Content != "resume this" {
		t.Fatalf("messages = %#v", m.messages)
	}
	if len(m.entries) != 2 || m.entries[0].content != "resume this" {
		t.Fatalf("entries = %#v", m.entries)
	}
}
