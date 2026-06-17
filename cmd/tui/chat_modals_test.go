package tui

import (
	"context"
	"errors"
	"slices"
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
	if got := len(fm.historyPopup.messages); got != 5 {
		t.Fatalf("history messages = %d, want 5", got)
	}
	if fm.historyPopup.messages[0].Role != "system" || fm.historyPopup.messages[0].Content != "You are Ollama." {
		t.Fatalf("system prompt history message = %#v", fm.historyPopup.messages[0])
	}
	if fm.historyPopup.messages[2].Thinking != "Need to inspect cwd." || len(fm.historyPopup.messages[2].ToolCalls) != 1 {
		t.Fatalf("assistant tool history message = %#v", fm.historyPopup.messages[2])
	}

	fm.width = 120
	fm.height = 40
	rendered := stripANSI(fm.View())
	for _, want := range []string{
		"Message history",
		"system",
		"content: You are Ollama.",
		"user",
		"content: where am i?",
		"assistant",
		"thinking: Need to inspect cwd.",
		"tool calls:",
		"call-1 Bash",
		"args:",
		"\"command\": \"pwd\"",
		"tool",
		"tool: bash",
		"tool call: call-1",
		"/tmp/project",
		"content: You are in /tmp/project.",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered history missing %q:\n%s", want, rendered)
		}
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
	if len(fm.entries) != 0 || fm.historyPopup == nil || len(fm.historyPopup.messages) != 0 {
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
	for i := range 18 {
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
	if title := strings.Split(view, "\n")[0]; strings.Contains(title, "/") {
		t.Fatalf("history popup should not show scroll counter in title:\n%s", view)
	}
}

func TestChatHistoryMouseWheelScrollsPopup(t *testing.T) {
	messages := make([]api.Message, 0, 18)
	for i := range 18 {
		messages = append(messages, api.Message{Role: "user", Content: "prompt " + strconv.Itoa(i)})
	}
	m := chatModel{
		historyPopup: &chatHistoryPopup{messages: messages, stickToBottom: true},
		width:        80,
		height:       10,
	}
	maxScroll := m.historyPopupMaxScroll()
	if maxScroll == 0 {
		t.Fatal("test setup should produce scrollable history")
	}

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseWheelUp})
	m = updated.(chatModel)
	if m.historyPopup.stickToBottom {
		t.Fatal("mouse wheel should detach history popup from bottom")
	}
	if m.historyPopup.scroll >= maxScroll {
		t.Fatalf("mouse wheel up should scroll toward older history, got %d max %d", m.historyPopup.scroll, maxScroll)
	}

	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseWheelDown})
	m = updated.(chatModel)
	if m.historyPopup.scroll != maxScroll {
		t.Fatalf("mouse wheel down should scroll back toward latest history, got %d want %d", m.historyPopup.scroll, maxScroll)
	}
}

func TestChatHistoryMouseDragSelectsAndCopiesText(t *testing.T) {
	var copied string
	m := chatModel{
		ctx: context.Background(),
		opts: ChatOptions{
			Clipboard: func(_ context.Context, text string) error {
				copied = text
				return nil
			},
		},
		historyPopup: &chatHistoryPopup{
			messages:      []api.Message{{Role: "user", Content: "alpha beta"}},
			stickToBottom: true,
		},
		width:  80,
		height: 10,
	}
	top, _ := m.historyPopupLayout()
	contentY := top + 1
	contentX := len("  content: ")

	updated, _ := m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress, X: contentX, Y: contentY})
	m = updated.(chatModel)
	updated, _ = m.Update(tea.MouseMsg{Type: tea.MouseLeft, Button: tea.MouseButtonLeft, Action: tea.MouseActionMotion, X: contentX + len("alpha"), Y: contentY})
	m = updated.(chatModel)
	if got := m.selectedHistoryPopupText(); got != "alpha" {
		t.Fatalf("selected history text = %q, want alpha", got)
	}
	if !m.historyPopup.selection.active {
		t.Fatal("history selection should stay active during drag")
	}

	updated, cmd := m.Update(tea.MouseMsg{Type: tea.MouseRelease, Action: tea.MouseActionRelease, X: contentX + len("alpha"), Y: contentY})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("mouse release should return clipboard command")
	}
	if msg := cmd(); msg != nil {
		updated, _ = m.Update(msg)
		m = updated.(chatModel)
	}
	if copied != "alpha" {
		t.Fatalf("copied = %q, want alpha", copied)
	}
	if m.status != "selection copied" {
		t.Fatalf("status = %q, want selection copied", m.status)
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
	fm.width = 80
	fm.height = 20
	view := stripANSI(fm.View())
	if !strings.Contains(view, "content:") || !strings.Contains(view, "first") || !strings.Contains(view, "second") {
		t.Fatalf("history should label multiline content before block:\n%s", view)
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

func TestChatModelCommandOpensPicker(t *testing.T) {
	m := chatModel{
		ctx:    context.Background(),
		input:  []rune("/model"),
		width:  100,
		height: 20,
		opts: ChatOptions{
			Model:               "llama3.2",
			ContextWindowTokens: 131072,
			ModelOptions: func(context.Context) ([]ChatModelOption, error) {
				return []ChatModelOption{
					{Name: "kimi-k2.6:cloud", Description: "cloud coding"},
					{Name: "llama3.2", Description: "local"},
				}, nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	if m.modelPicker == nil {
		t.Fatal("model picker was not opened")
	}
	view := stripANSI(m.View())
	if !strings.Contains(view, "Switch model") ||
		!strings.Contains(view, "Search...") ||
		!strings.Contains(view, "kimi-k2.6:cloud") ||
		!strings.Contains(view, "llama3.2") ||
		!strings.Contains(view, "current") {
		t.Fatalf("model picker view missing content: %q", view)
	}
}

func TestChatModelPickerShowsRecommendedModelsFirst(t *testing.T) {
	models := normalizeModelOptions([]ChatModelOption{
		{Name: "llama3.2", Description: "local"},
		{Name: "kimi-k2.6:cloud", Description: "cloud coding", Recommended: true},
		{Name: "qwen3.5:cloud", Description: "cloud reasoning", Recommended: true},
		{Name: "gemma4", Description: "local"},
	})
	got := make([]string, 0, len(models))
	for _, model := range models {
		got = append(got, model.Name)
	}
	want := []string{"kimi-k2.6:cloud", "qwen3.5:cloud", "llama3.2", "gemma4"}
	if !slices.Equal(got, want) {
		t.Fatalf("model order = %#v, want %#v", got, want)
	}
}

func TestChatModelPickerFiltersAndSwitchesModel(t *testing.T) {
	var savedModel string
	store := &chatResumeTestStore{}
	originalMessages := []api.Message{{Role: "user", Content: "keep me"}}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		input:    []rune("/model qwen"),
		width:    100,
		height:   20,
		messages: slices.Clone(originalMessages),
		opts: ChatOptions{
			Model: "llama3.2",
			Store: store,
			ModelOptions: func(context.Context) ([]ChatModelOption, error) {
				return []ChatModelOption{
					{Name: "llama3.2", Description: "local"},
					{Name: "qwen3.5:cloud", Description: "cloud reasoning"},
				}, nil
			},
			ToolRegistryForModel: func(ctx context.Context, model string) *coreagent.Registry {
				if model != "qwen3.5:cloud" {
					t.Fatalf("tool registry model = %q, want qwen3.5:cloud", model)
				}
				registry := coreagent.NewRegistry()
				registry.Register(chatTestTool{})
				return registry
			},
			ContextWindowTokensForModel: func(ctx context.Context, model string, fallback int) int {
				if model != "qwen3.5:cloud" {
					t.Fatalf("context model = %q, want qwen3.5:cloud", model)
				}
				if fallback != 0 {
					t.Fatalf("context fallback = %d, want 0 after model switch", fallback)
				}
				return 262144
			},
			SystemPromptForModel: func(ctx context.Context, model string, registry *coreagent.Registry) string {
				if model != "qwen3.5:cloud" {
					t.Fatalf("system prompt model = %q, want qwen3.5:cloud", model)
				}
				if registry == nil || !registry.Has("fake_tool") {
					t.Fatalf("system prompt registry missing fake tool: %#v", registry)
				}
				return "system for " + model
			},
			OnModelSelected: func(ctx context.Context, model string) error {
				savedModel = model
				return nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("model command should not return a command")
	}
	m = updated.(chatModel)
	if m.modelPicker == nil || m.modelPicker.filter != "qwen" {
		t.Fatalf("model picker = %#v, want qwen filter", m.modelPicker)
	}
	if view := stripANSI(m.View()); !strings.Contains(view, "qwen3.5:cloud") || strings.Contains(view, "llama3.2") {
		t.Fatalf("filtered model picker view = %q", view)
	}

	updated, cmd = m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd != nil {
		t.Fatal("switching models should not start a command")
	}
	if m.modelPicker != nil {
		t.Fatal("model picker should close after selection")
	}
	if m.opts.Model != "qwen3.5:cloud" {
		t.Fatalf("model = %q, want qwen3.5:cloud", m.opts.Model)
	}
	if m.chatID != "chat-1" {
		t.Fatalf("chatID = %q, want chat-1", m.chatID)
	}
	if len(m.messages) != len(originalMessages) || m.messages[0].Content != originalMessages[0].Content {
		t.Fatalf("messages changed on model switch: %#v", m.messages)
	}
	if len(m.entries) != 0 {
		t.Fatalf("model switch should not append transcript entries: %#v", m.entries)
	}
	if got := store.setModels["chat-1"]; got != "qwen3.5:cloud" {
		t.Fatalf("persisted chat model = %q, want qwen3.5:cloud", got)
	}
	if savedModel != "qwen3.5:cloud" {
		t.Fatalf("saved model = %q, want qwen3.5:cloud", savedModel)
	}
	if m.opts.Tools == nil || !m.opts.Tools.Has("fake_tool") {
		t.Fatalf("tools registry was not rebuilt for model: %#v", m.opts.Tools)
	}
	if m.opts.ContextWindowTokens != 262144 {
		t.Fatalf("context window = %d, want 262144", m.opts.ContextWindowTokens)
	}
	if m.opts.SystemPrompt != "system for qwen3.5:cloud" {
		t.Fatalf("system prompt = %q", m.opts.SystemPrompt)
	}
}

func TestChatModelSelectionPersistsBeforeSwitching(t *testing.T) {
	originalTools := coreagent.NewRegistry()
	errStore := errors.New("write failed")
	m := chatModel{
		ctx:    context.Background(),
		chatID: "chat-1",
		opts: ChatOptions{
			Model:               "llama3.2",
			Store:               &chatResumeTestStore{setModelErr: errStore},
			Tools:               originalTools,
			SystemPrompt:        "system for llama3.2",
			ContextWindowTokens: 8192,
			ToolRegistryForModel: func(context.Context, string) *coreagent.Registry {
				t.Fatal("tool registry should not rebuild when model persistence fails")
				return nil
			},
			SystemPromptForModel: func(context.Context, string, *coreagent.Registry) string {
				t.Fatal("system prompt should not rebuild when model persistence fails")
				return ""
			},
		},
		messages: []api.Message{{Role: "user", Content: "history"}},
	}

	err := m.applyModelSelection("qwen3", true)
	if !errors.Is(err, errStore) {
		t.Fatalf("error = %v, want %v", err, errStore)
	}
	if m.opts.Model != "llama3.2" {
		t.Fatalf("model = %q, want llama3.2", m.opts.Model)
	}
	if m.opts.Tools != originalTools {
		t.Fatal("tools changed after failed model persistence")
	}
	if m.opts.SystemPrompt != "system for llama3.2" {
		t.Fatalf("system prompt = %q, want original", m.opts.SystemPrompt)
	}
	if m.opts.ContextWindowTokens != 8192 {
		t.Fatalf("context window = %d, want 8192", m.opts.ContextWindowTokens)
	}
	if len(m.messages) != 1 || m.messages[0].Content != "history" {
		t.Fatalf("messages changed after failed model persistence: %#v", m.messages)
	}
}

func TestChatModelSwitchNextRunKeepsHistory(t *testing.T) {
	client := &chatCaptureClient{}
	store := &chatResumeTestStore{}
	history := []api.Message{
		{Role: "user", Content: "old question"},
		{Role: "assistant", Content: "old answer"},
	}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		messages: slices.Clone(history),
		input:    []rune("continue"),
		opts: ChatOptions{
			Model:  "llama3.2",
			Store:  store,
			Client: client,
			SystemPromptForModel: func(_ context.Context, model string, _ *coreagent.Registry) string {
				return "system for " + model
			},
		},
	}
	if err := m.applyModelSelection("qwen3", true); err != nil {
		t.Fatal(err)
	}

	updated, cmd := m.handleSubmit()
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("next prompt should start a model run")
	}
	done := waitForRunDone(t, m.events)
	if done.err != nil {
		t.Fatal(done.err)
	}

	if len(client.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(client.requests))
	}
	req := client.requests[0]
	if req.Model != "qwen3" {
		t.Fatalf("request model = %q, want qwen3", req.Model)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("request messages = %#v, want system + 2 history + new user", req.Messages)
	}
	if req.Messages[0].Role != "system" || req.Messages[0].Content != "system for qwen3" {
		t.Fatalf("system message = %#v", req.Messages[0])
	}
	for i, want := range history {
		got := req.Messages[i+1]
		if got.Role != want.Role || got.Content != want.Content {
			t.Fatalf("history message %d = %#v, want %#v", i, got, want)
		}
	}
	if req.Messages[3].Role != "user" || req.Messages[3].Content != "continue" {
		t.Fatalf("new user message = %#v", req.Messages[3])
	}
	if got := store.setModels["chat-1"]; got != "qwen3" {
		t.Fatalf("persisted chat model = %q, want qwen3", got)
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
