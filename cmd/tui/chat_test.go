package tui

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/chatstore"
	agentskills "github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
)

type chatTestTool struct{}

type chatTestClient struct{}

type chatCaptureClient struct {
	requests []*api.ChatRequest
}

type chatResumeTestStore struct {
	chats   []chatstore.ChatSummary
	byID    map[string]*chatstore.Chat
	prompts []string
}

type chatTestCompactor struct {
	result   coreagent.CompactionResult
	err      error
	progress []int
	request  coreagent.CompactionRequest
}

func (chatTestClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn(api.ChatResponse{
		Message: api.Message{Role: "assistant", Content: "ok"},
		Done:    true,
	})
}

func (c *chatCaptureClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.requests = append(c.requests, req)
	return fn(api.ChatResponse{
		Message: api.Message{Role: "assistant", Content: "ok"},
		Done:    true,
	})
}

func (s *chatResumeTestStore) EnsureChat(context.Context, string, string) error {
	return nil
}

func (s *chatResumeTestStore) AppendMessage(context.Context, string, api.Message) error {
	return nil
}

func (s *chatResumeTestStore) UpdateLastMessage(context.Context, string, api.Message) error {
	return nil
}

func (s *chatResumeTestStore) ListChats(context.Context, int) ([]chatstore.ChatSummary, error) {
	return slices.Clone(s.chats), nil
}

func (s *chatResumeTestStore) Chat(_ context.Context, id string) (*chatstore.Chat, error) {
	chat, ok := s.byID[id]
	if !ok {
		return nil, sql.ErrNoRows
	}
	out := *chat
	out.Messages = slices.Clone(chat.Messages)
	return &out, nil
}

func (s *chatResumeTestStore) ListUserMessages(context.Context, int) ([]string, error) {
	return slices.Clone(s.prompts), nil
}

func (c *chatTestCompactor) MaybeCompact(_ context.Context, req coreagent.CompactionRequest) (coreagent.CompactionResult, error) {
	c.request = req
	for _, tokens := range c.progress {
		if req.Progress != nil {
			req.Progress(coreagent.CompactionProgress{Tokens: tokens})
		}
	}
	return c.result, c.err
}

func nextChatMsg(t *testing.T, ch <-chan tea.Msg) tea.Msg {
	t.Helper()
	select {
	case msg, ok := <-ch:
		if !ok {
			t.Fatal("message channel closed")
		}
		return msg
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for chat message")
		return nil
	}
}

func (chatTestTool) Name() string {
	return "fake_tool"
}

func (chatTestTool) Description() string {
	return "does test work"
}

func (chatTestTool) Schema() api.ToolFunction {
	return api.ToolFunction{
		Name:        "fake_tool",
		Description: "does test work",
		Parameters: api.ToolFunctionParameters{
			Type: "object",
		},
	}
}

func (chatTestTool) Execute(context.Context, coreagent.ToolContext, map[string]any) (coreagent.ToolResult, error) {
	return coreagent.ToolResult{Content: "ok"}, nil
}

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
		!strings.Contains(fm.entries[0].content, "- `/history`: show prompt message history") ||
		!strings.Contains(fm.entries[0].content, "- `/<skill>`: run the next message with a skill") ||
		!strings.Contains(fm.entries[0].content, "**Shortcuts**") ||
		!strings.Contains(fm.entries[0].content, "- `ctrl+o`: toggle tool output and details") ||
		!strings.Contains(fm.entries[0].content, "- `shift+tab`: toggle permission mode") {
		t.Fatalf("help output = %q", fm.entries[0].content)
	}
}

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

func TestChatApprovalPromptRendersAndApprovesOnce(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	request := coreagent.ApprovalRequest{
		ToolCallID: "call-1",
		ToolName:   "bash",
		Args:       map[string]any{"command": "git status"},
		Summary:    "Bash wants to run a command",
		Risk:       coreagent.ApprovalRiskMedium,
		Reasons:    []string{"runs shell commands"},
	}
	m := chatModel{
		width:  100,
		height: 20,
		events: make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{request: request, reply: reply})

	view := stripANSI(m.View())
	if !strings.Contains(view, "Ollama") ||
		!strings.Contains(view, "Bash wants to run a command") ||
		!strings.Contains(view, "Approve once") ||
		!strings.Contains(view, "> █") ||
		!strings.Contains(view, "waiting for approval") {
		t.Fatalf("approval view missing content: %q", view)
	}
	if len(m.entries) != 1 || m.entries[0].status != "approval" {
		t.Fatalf("approval tool entry = %#v", m.entries)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("approval should resume waiting for agent events")
	}
	result := <-reply
	if result.Decision != coreagent.ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
	}
	if m.approvalPrompt != nil {
		t.Fatal("approval prompt should close")
	}
	if m.entries[0].status != "queued" {
		t.Fatalf("tool status = %q, want queued", m.entries[0].status)
	}
}

func TestChatApprovalPromptClosesHistoryPopup(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		width:        100,
		height:       24,
		historyPopup: &chatHistoryPopup{content: "**Message History**\n\n**user**\n  content: old"},
	}

	updated, cmd := m.Update(chatApprovalPromptMsg{
		request: coreagent.ApprovalRequest{
			ToolCallID: "call-1",
			ToolName:   "bash",
			Args:       map[string]any{"command": "git status"},
			Summary:    "Bash wants to run a command",
		},
		reply: reply,
	})
	if cmd != nil {
		t.Fatal("opening approval prompt should not return a command")
	}
	m = updated.(chatModel)
	if m.historyPopup != nil {
		t.Fatal("approval prompt should close history popup")
	}
	if m.approvalPrompt == nil {
		t.Fatal("approval prompt should open")
	}

	view := stripANSI(m.View())
	if strings.Contains(view, "Message history") {
		t.Fatalf("history popup should not render over approval prompt:\n%s", view)
	}
	for _, want := range []string{"Bash wants to run a command", "Approve once", "Approve session", "Deny"} {
		if !strings.Contains(view, want) {
			t.Fatalf("approval view missing %q:\n%s", want, view)
		}
	}
}

func TestChatApprovalPromptDenyShortcut(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		events: make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{
		request: coreagent.ApprovalRequest{ToolCallID: "call-1", ToolName: "edit", Args: map[string]any{"path": "note.txt"}},
		reply:   reply,
	})

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	m = updated.(chatModel)
	result := <-reply
	if result.Decision != coreagent.ApprovalDeny {
		t.Fatalf("decision = %q, want deny", result.Decision)
	}
	if m.entries[0].status != "error" {
		t.Fatalf("tool status = %q, want error", m.entries[0].status)
	}
}

func TestChatShiftTabTogglesPermissionMode(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyShiftTab})
	if cmd != nil {
		t.Fatal("permission toggle should not start a command")
	}
	m = updated.(chatModel)
	if !m.autoApproveTools() {
		t.Fatal("shift+tab should enable auto-approve mode")
	}
	if footer := m.footerLine(); !strings.Contains(footer, "full access") || !strings.Contains(footer, "shift+tab") || strings.Contains(footer, "perm") {
		t.Fatalf("footer missing auto-approve permission mode: %q", footer)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyShiftTab})
	m = updated.(chatModel)
	if m.autoApproveTools() {
		t.Fatal("second shift+tab should return to review mode")
	}
	if footer := m.footerLine(); !strings.Contains(footer, "review") || strings.Contains(footer, "perm") {
		t.Fatalf("footer missing review permission mode: %q", footer)
	}
}

func TestChatShiftTabApprovesPendingPrompt(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		events: make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{
		request: coreagent.ApprovalRequest{ToolCallID: "call-1", ToolName: "bash", Args: map[string]any{"command": "pwd"}},
		reply:   reply,
	})

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyShiftTab})
	m = updated.(chatModel)
	if cmd == nil {
		t.Fatal("approving pending prompt should resume waiting for agent events")
	}
	result := <-reply
	if result.Decision != coreagent.ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
	}
	if !m.autoApproveTools() {
		t.Fatal("shift+tab should leave future tool calls in auto-approve mode")
	}
	if m.approvalPrompt != nil {
		t.Fatal("approval prompt should close")
	}
	if m.entries[0].status != "queued" {
		t.Fatalf("tool status = %q, want queued", m.entries[0].status)
	}
}

func TestChatPermissionApprovalHandlerReadsModeAtApprovalTime(t *testing.T) {
	mode := newChatPermissionMode(false)
	handler := chatPermissionApprovalHandler{
		mode:   mode,
		review: coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{}),
	}
	req := coreagent.ApprovalRequest{
		ToolName: "bash",
		Args:     map[string]any{"command": "pwd"},
	}

	if !handler.RequiresApproval(context.Background(), chatTestTool{}, req) {
		t.Fatal("review mode should require approval for bash")
	}
	mode.SetAutoApprove(true)
	if handler.RequiresApproval(context.Background(), chatTestTool{}, req) {
		t.Fatal("auto-approve mode should not require approval")
	}
	result, err := handler.Approve(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Decision != coreagent.ApprovalAllowOnce {
		t.Fatalf("decision = %q, want allow_once", result.Decision)
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
	if !strings.Contains(view, "/history") || !strings.Contains(view, "show prompt message history") {
		t.Fatalf("view missing history suggestion: %q", view)
	}
	if !strings.Contains(view, "/new") || !strings.Contains(view, "start a new chat") {
		t.Fatalf("view missing new suggestion: %q", view)
	}
	if !strings.Contains(view, "/resume") || !strings.Contains(view, "resume a saved chat") {
		t.Fatalf("view missing resume suggestion: %q", view)
	}
	if !strings.Contains(view, "> /█") {
		t.Fatalf("view missing slash input row: %q", view)
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

func waitForRunDone(t *testing.T, events <-chan tea.Msg) chatRunDoneMsg {
	t.Helper()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case msg, ok := <-events:
			if !ok {
				t.Fatal("events closed before run done")
			}
			if done, ok := msg.(chatRunDoneMsg); ok {
				return done
			}
		case <-timeout:
			t.Fatal("timed out waiting for run done")
		}
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

func TestChatEnterQueuesWhileRunning(t *testing.T) {
	m := chatModel{
		input:   []rune("next prompt"),
		running: true,
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	m = updated.(chatModel)

	if cmd != nil {
		t.Fatal("queueing during a run should not start another command immediately")
	}
	if !m.running {
		t.Fatal("current run should stay active")
	}
	if got := string(m.input); got != "" {
		t.Fatalf("input = %q, want cleared after queue", got)
	}
	if len(m.queued) != 1 || m.queued[0] != "next prompt" {
		t.Fatalf("queued = %#v, want next prompt", m.queued)
	}
	if !strings.Contains(stripANSI(m.View()), "queued 1") {
		t.Fatalf("view should show queued count: %q", stripANSI(m.View()))
	}
}

func TestChatRunDoneStartsQueuedMessage(t *testing.T) {
	m := chatModel{
		ctx:     context.Background(),
		running: true,
		queued:  []string{"next prompt"},
		opts: ChatOptions{
			Model:  "test",
			Client: chatTestClient{},
		},
	}

	updated, cmd := m.Update(chatRunDoneMsg{result: &coreagent.RunResult{Messages: []api.Message{{Role: "assistant", Content: "done"}}}})
	m = updated.(chatModel)

	if cmd == nil {
		t.Fatal("queued message should start a new run")
	}
	if !m.running {
		t.Fatal("queued message should be running")
	}
	if len(m.queued) != 0 {
		t.Fatalf("queued = %#v, want empty after start", m.queued)
	}
	if len(m.entries) == 0 || m.entries[len(m.entries)-1].role != "user" || m.entries[len(m.entries)-1].content != "next prompt" {
		t.Fatalf("last entry should be queued user prompt: %#v", m.entries)
	}
}

func TestChatTickAdvancesSpinnerWhileRunning(t *testing.T) {
	m := chatModel{running: true}

	updated, cmd := m.Update(chatTickMsg{})
	fm := updated.(chatModel)

	if fm.spinner != 1 {
		t.Fatalf("spinner = %d, want 1", fm.spinner)
	}
	if cmd == nil {
		t.Fatal("running tick should schedule another tick")
	}
}

func TestChatTickStopsWhenIdle(t *testing.T) {
	m := chatModel{}

	updated, cmd := m.Update(chatTickMsg{})
	fm := updated.(chatModel)

	if fm.spinner != 0 {
		t.Fatalf("spinner = %d, want 0", fm.spinner)
	}
	if cmd != nil {
		t.Fatal("idle tick should not schedule another tick")
	}
}

func TestChatActivityLabelOmitsResponding(t *testing.T) {
	m := chatModel{
		running: true,
		entries: []chatEntry{
			{role: "assistant", content: "hello"},
		},
	}

	if got := m.activityLabel(); got != "" {
		t.Fatalf("activityLabel = %q, want empty", got)
	}
}

func TestChatThinkingShowsTokenCount(t *testing.T) {
	m := chatModel{running: true}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "abcdefgh"})

	if got := m.activityLabel(); got != "thinking 2 tokens" {
		t.Fatalf("activityLabel = %q, want thinking 2 tokens", got)
	}
	if strings.Contains(stripANSI(m.renderTranscript(80)), "abcdefgh") {
		t.Fatalf("thinking text should not render in transcript: %q", stripANSI(m.renderTranscript(80)))
	}

	response := api.ChatResponse{Metrics: api.Metrics{EvalCount: 12}}
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "more", Response: &response})
	if got := m.activityLabel(); got != "thinking 12 tokens" {
		t.Fatalf("activityLabel = %q, want thinking 12 tokens", got)
	}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})
	if m.thinking || m.thinkingTokens != 0 {
		t.Fatalf("thinking state was not cleared: thinking=%v tokens=%d", m.thinking, m.thinkingTokens)
	}
}

func TestChatFooterShowsContextAndCompactionOnlyWhenNear(t *testing.T) {
	m := chatModel{
		width:           120,
		height:          24,
		contextTokens:   50,
		contextEstimate: true,
		opts: ChatOptions{
			Options:             map[string]any{"num_ctx": 100},
			CompactionThreshold: 0.75,
		},
	}

	view := stripANSI(m.View())
	if !strings.Contains(view, "ctx ~50/100 (50%)") {
		t.Fatalf("view missing context pressure: %q", view)
	}
	if strings.Contains(view, "compact at") || strings.Contains(view, "compact due") {
		t.Fatalf("view should hide distant compaction point: %q", view)
	}

	m.contextTokens = 65
	view = stripANSI(m.View())
	if !strings.Contains(view, "ctx ~65/100 (65%)") || !strings.Contains(view, "compact at 75") {
		t.Fatalf("view should show compaction point near threshold: %q", view)
	}

	m.contextTokens = 75
	view = stripANSI(m.View())
	if !strings.Contains(view, "compact due at 75") {
		t.Fatalf("view should show compaction due at threshold: %q", view)
	}
}

func TestChatStartRunEstimatesFullPrompt(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})

	systemPrompt := strings.Repeat("system prompt ", 20)
	userMsg := api.Message{Role: "user", Content: "hello"}
	m := chatModel{
		ctx:   context.Background(),
		input: []rune(userMsg.Content),
		opts: ChatOptions{
			Model:        "test",
			Client:       chatTestClient{},
			Tools:        registry,
			SystemPrompt: systemPrompt,
		},
	}

	updated, _ := m.handleSubmit()
	fm := updated.(chatModel)
	if fm.cancel != nil {
		fm.cancel()
	}

	want := estimatePromptTokenCount(systemPrompt, []api.Message{userMsg}, registry.Tools(), "")
	if fm.contextTokens != want {
		t.Fatalf("contextTokens = %d, want full prompt estimate %d", fm.contextTokens, want)
	}

	messageOnly := estimatePromptTokenCount("", []api.Message{userMsg}, nil, "")
	if fm.contextTokens <= messageOnly {
		t.Fatalf("context estimate should include system prompt and tools: got %d, message-only %d", fm.contextTokens, messageOnly)
	}
	if !fm.contextEstimate {
		t.Fatal("initial context count should be marked estimated")
	}
}

func TestChatStartRunRefreshesEffectiveContextWindow(t *testing.T) {
	compactor := coreagent.NewSimpleCompactor(nil, nil, coreagent.CompactionOptions{
		ContextWindowTokens: 262144,
	})
	m := chatModel{
		ctx:   context.Background(),
		input: []rune("hello"),
		opts: ChatOptions{
			Model:               "llama3.2",
			Client:              chatTestClient{},
			Compactor:           compactor,
			ContextWindowTokens: 262144,
			ContextWindowTokensForModel: func(_ context.Context, model string, fallback int) int {
				if model != "llama3.2" || fallback != 262144 {
					t.Fatalf("resolver called with model=%q fallback=%d", model, fallback)
				}
				return 8192
			},
		},
	}

	updated, _ := m.handleSubmit()
	fm := updated.(chatModel)
	if fm.cancel != nil {
		fm.cancel()
	}
	if fm.opts.ContextWindowTokens != 8192 {
		t.Fatalf("ContextWindowTokens = %d, want effective runner window 8192", fm.opts.ContextWindowTokens)
	}
	if compactor.Options.ContextWindowTokens != 8192 {
		t.Fatalf("compactor ContextWindowTokens = %d, want 8192", compactor.Options.ContextWindowTokens)
	}
}

func TestChatRunDoneKeepsAPIPromptEvalCount(t *testing.T) {
	registry := coreagent.NewRegistry()
	registry.Register(chatTestTool{})
	messages := []api.Message{{Role: "user", Content: "hello"}}
	m := chatModel{
		opts: ChatOptions{
			Tools:        registry,
			SystemPrompt: strings.Repeat("system prompt ", 20),
		},
	}

	updated, _ := m.Update(chatRunDoneMsg{
		result: &coreagent.RunResult{
			Messages: messages,
			Latest: api.ChatResponse{
				Metrics: api.Metrics{PromptEvalCount: 123},
			},
		},
	})

	fm := updated.(chatModel)
	if fm.contextTokens != 123 {
		t.Fatalf("contextTokens = %d, want API prompt eval count", fm.contextTokens)
	}
	if fm.contextEstimate {
		t.Fatal("API prompt eval count should not be marked estimated")
	}
}

func TestChatViewShowsRunningActivityOnce(t *testing.T) {
	m := chatModel{
		running:        true,
		thinking:       true,
		thinkingTokens: 7,
		width:          80,
		height:         24,
	}

	view := stripANSI(m.View())
	if got := strings.Count(view, "thinking 7 tokens"); got != 1 {
		t.Fatalf("thinking activity rendered %d times, want 1:\n%s", got, view)
	}
	if strings.Contains(view, "sent ") || strings.Contains(view, "received ") {
		t.Fatalf("view should not show sent/received metrics: %q", view)
	}
}

func TestChatViewShowsCompactingActivity(t *testing.T) {
	m := chatModel{
		compacting:       true,
		compactingTokens: 42,
		width:            80,
		height:           24,
	}

	view := stripANSI(m.View())
	if got := strings.Count(view, "compacting 42 tokens"); got != 1 {
		t.Fatalf("compacting activity rendered %d times, want 1:\n%s", got, view)
	}
}

func TestChatCtrlCCancelsQuietly(t *testing.T) {
	canceled := false
	m := chatModel{
		running: true,
		cancel: func() {
			canceled = true
		},
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	m = updated.(chatModel)
	if !canceled {
		t.Fatal("ctrl+c should cancel an active run")
	}
	if cmd != nil {
		t.Fatal("ctrl+c during a run should not quit")
	}
	if m.status != "canceling" {
		t.Fatalf("status = %q, want canceling", m.status)
	}

	updated, _ = m.Update(chatRunDoneMsg{err: context.Canceled})
	m = updated.(chatModel)
	if m.running {
		t.Fatal("run should no longer be active after cancellation completes")
	}
	if m.status != "Tell the model what to do instead." {
		t.Fatalf("status = %q, want friendly cancellation hint", m.status)
	}
	for _, entry := range m.entries {
		if entry.role == "error" {
			t.Fatalf("cancellation should not append an error entry: %#v", entry)
		}
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

func TestChatClearCommandResetsConversation(t *testing.T) {
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "old",
		messages: []api.Message{{Role: "user", Content: "hello"}},
		entries:  []chatEntry{{role: "user", content: "hello"}},
		queued:   []string{"later"},
		input:    []rune("/clear"),
		opts: ChatOptions{
			NewChat: func(context.Context) (string, error) {
				return "new", nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("clear command should not return a command")
	}

	fm := updated.(chatModel)
	if fm.chatID != "new" {
		t.Fatalf("chatID = %q, want new", fm.chatID)
	}
	if len(fm.messages) != 0 || len(fm.entries) != 0 || len(fm.queued) != 0 {
		t.Fatalf("conversation was not cleared: messages=%d entries=%d queued=%d", len(fm.messages), len(fm.entries), len(fm.queued))
	}
	if fm.status != "cleared" {
		t.Fatalf("status = %q, want cleared", fm.status)
	}
}

func TestChatNewCommandStartsFreshChat(t *testing.T) {
	m := chatModel{
		ctx:             context.Background(),
		chatID:          "old",
		messages:        []api.Message{{Role: "user", Content: "hello"}},
		entries:         []chatEntry{{role: "user", content: "hello"}},
		contextTokens:   42,
		contextEstimate: true,
		input:           []rune("/new"),
		opts: ChatOptions{
			NewChat: func(context.Context) (string, error) {
				return "fresh", nil
			},
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd != nil {
		t.Fatal("new command should not return a command")
	}

	fm := updated.(chatModel)
	if fm.chatID != "fresh" {
		t.Fatalf("chatID = %q, want fresh", fm.chatID)
	}
	if len(fm.messages) != 0 || len(fm.entries) != 0 {
		t.Fatalf("conversation was not reset: messages=%d entries=%d", len(fm.messages), len(fm.entries))
	}
	if fm.contextTokens != 0 {
		t.Fatalf("contextTokens = %d, want 0", fm.contextTokens)
	}
	if fm.status != "new chat" {
		t.Fatalf("status = %q, want new chat", fm.status)
	}
}

func TestChatCompactCommandShowsSummary(t *testing.T) {
	compactor := &chatTestCompactor{
		progress: []int{12},
		result: coreagent.CompactionResult{
			Messages: []api.Message{
				{Role: "user", Content: "Conversation summary:\nold work summary"},
				{Role: "user", Content: "recent request"},
			},
			Compacted: true,
			Due:       true,
			Summary:   "old work summary",
		},
	}
	m := chatModel{
		ctx:      context.Background(),
		chatID:   "chat-1",
		messages: []api.Message{{Role: "user", Content: "old request"}},
		input:    []rune("/compact"),
		opts: ChatOptions{
			Model:     "test",
			Compactor: compactor,
		},
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("compact command should return a command")
	}
	fm := updated.(chatModel)
	if fm.status != "compacting" {
		t.Fatalf("status = %q, want compacting", fm.status)
	}
	if !fm.compacting {
		t.Fatal("compact command should mark model as compacting")
	}
	if got := fm.activityLabel(); got != "compacting" {
		t.Fatalf("activityLabel = %q, want compacting", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)
	if got := fm.activityLabel(); got != "compacting 12 tokens" {
		t.Fatalf("activityLabel = %q, want compacting 12 tokens", got)
	}

	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)

	if !compactor.request.Force {
		t.Fatal("manual compaction should be forced")
	}
	if fm.compacting {
		t.Fatal("compacting should be cleared after completion")
	}
	if fm.status != "compacted" {
		t.Fatalf("status = %q, want compacted", fm.status)
	}
	if len(fm.messages) != 2 || fm.messages[0].Role != "user" || !strings.Contains(fm.messages[0].Content, "old work summary") {
		t.Fatalf("compacted messages = %#v", fm.messages)
	}
	transcript := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "Compacted summary done") {
		t.Fatalf("summary row should be visible after compacting: %q", transcript)
	}
	if strings.Contains(transcript, "old work summary") || strings.Contains(transcript, "Conversation summary:") {
		t.Fatalf("summary body should be collapsed after compacting: %q", transcript)
	}

	updated, _ = fm.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	fm = updated.(chatModel)
	transcript = stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "old work summary") {
		t.Fatalf("expanded summary should show body: %q", transcript)
	}
}

func TestChatCompactCommandSuggestsNewWhenSkipped(t *testing.T) {
	compactor := &chatTestCompactor{
		result: coreagent.CompactionResult{
			Messages: []api.Message{{Role: "user", Content: "only request"}},
			Due:      true,
			Reason:   "not enough older messages to compact",
		},
	}
	m := chatModel{
		ctx:      context.Background(),
		messages: []api.Message{{Role: "user", Content: "only request"}},
		input:    []rune("/compact"),
		opts:     ChatOptions{Compactor: compactor},
	}

	updated, cmd := m.handleSubmit()
	if cmd == nil {
		t.Fatal("compact command should return a command")
	}
	fm := updated.(chatModel)
	updated, _ = fm.Update(nextChatMsg(t, fm.compactEvents))
	fm = updated.(chatModel)

	if fm.status != "compact skipped" {
		t.Fatalf("status = %q, want compact skipped", fm.status)
	}
	transcript := stripANSI(fm.renderTranscript(100))
	if !strings.Contains(transcript, "not enough older messages to compact") || !strings.Contains(transcript, "/new") {
		t.Fatalf("skip message should explain /new: %q", transcript)
	}
}

func TestChatAgentEventsUpdateAssistantEntry(t *testing.T) {
	m := chatModel{}

	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageStarted})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventThinkingDelta, Thinking: "thinking"})
	m.applyAgentEvent(coreagent.Event{Type: coreagent.EventMessageDelta, Content: "done"})

	if len(m.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(m.entries))
	}
	entry := m.entries[0]
	if entry.role != "assistant" {
		t.Fatalf("role = %q, want assistant", entry.role)
	}
	if entry.detail != "" {
		t.Fatalf("detail = %q, want no rendered thinking", entry.detail)
	}
	if entry.content != "done" {
		t.Fatalf("content = %q, want done", entry.content)
	}
	if strings.Contains(stripANSI(m.renderTranscript(80)), "thinking") {
		t.Fatalf("thinking text should not render in transcript: %q", stripANSI(m.renderTranscript(80)))
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

func TestChatMarkdownRendersAssistantAndSystemOutput(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{
		{role: "assistant", content: "**Current role:** Software Engineer\n\n- AI agents\n- Structured outputs"},
		{role: "system", content: "**Available tools:**\n\n- bash\n- read"},
	}

	transcript := stripANSI(m.renderTranscript(100))
	if strings.Contains(transcript, "**Current role:**") || strings.Contains(transcript, "**Available tools:**") {
		t.Fatalf("markdown markers were not rendered: %q", transcript)
	}
	if !strings.Contains(transcript, "Current role:") || !strings.Contains(transcript, "Available tools:") {
		t.Fatalf("rendered markdown content missing: %q", transcript)
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

func TestChatMarkdownExposesLinks(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{{
		role:    "assistant",
		content: "Open [Ollama](https://ollama.com) for details.",
	}}

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "https://ollama.com") {
		t.Fatalf("rendered markdown should expose URL for terminal clicking: %q", transcript)
	}
}

func TestChatMarkdownRendersTableWithinWidth(t *testing.T) {
	markdown := strings.Join([]string{
		"| Tool | Description |",
		"| --- | --- |",
		"| bash | Execute shell commands, inspect files, run tests, and perform development tasks. |",
		"| web_search | Search the web for current information that may not be in the model training data. |",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 48))
	if strings.Contains(rendered, "| --- |") {
		t.Fatalf("table should render instead of preserving markdown separator: %q", rendered)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if len([]rune(line)) > 48 {
			t.Fatalf("rendered table line width = %d, want <= 48: %q\n%s", len([]rune(line)), line, rendered)
		}
	}
	if !strings.Contains(rendered, "bash") || !strings.Contains(rendered, "web_search") {
		t.Fatalf("rendered table missing content: %q", rendered)
	}
}

func TestChatMarkdownRendersTableWithoutOuterPipes(t *testing.T) {
	markdown := strings.Join([]string{
		"Here is a summary table:",
		"",
		"Category | Details",
		"-------- | -------",
		"**Current Role** | Software Engineer",
		"Company | [Ollama](https://ollama.com/)",
		"Previous Experience | Tesla, Apple, Co-founder of Extensible (LLM monitoring / agent reliability startup)",
		"Personal Interests | Latte art, Muay Thai, writing essays",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 76))
	if strings.Contains(rendered, "-------- | -------") ||
		strings.Contains(rendered, "Category | Details") ||
		strings.Contains(rendered, "Previous Experience |") {
		t.Fatalf("table should not preserve raw markdown table syntax: %q", rendered)
	}
	if !strings.Contains(rendered, "Current Role") ||
		!strings.Contains(rendered, "Software Engineer") ||
		!strings.Contains(rendered, "https://ollama.com/") {
		t.Fatalf("rendered table missing content: %q", rendered)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if len([]rune(line)) > 76 {
			t.Fatalf("rendered table line width = %d, want <= 76: %q\n%s", len([]rune(line)), line, rendered)
		}
	}
}

func TestChatMarkdownDoesNotExposeLinksInsideCodeFences(t *testing.T) {
	markdown := strings.Join([]string{
		"Open [Ollama](https://ollama.com) for details.",
		"",
		"```text",
		"[Code Link](https://code.example)",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 80))
	if !strings.Contains(rendered, "Ollama (https://ollama.com)") {
		t.Fatalf("prose link should expose URL: %q", rendered)
	}
	if strings.Contains(rendered, "Code Link (https://code.example)") {
		t.Fatalf("code fence link should remain literal: %q", rendered)
	}
	if !strings.Contains(rendered, "[Code Link](https://code.example)") {
		t.Fatalf("code fence content missing literal markdown link: %q", rendered)
	}
}

func TestChatMarkdownDoesNotRenderTablesInsideCodeFences(t *testing.T) {
	markdown := strings.Join([]string{
		"```text",
		"Name | Value",
		"--- | ---",
		"one | two",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 80))
	if !strings.Contains(rendered, "Name | Value") ||
		!strings.Contains(rendered, "--- | ---") ||
		!strings.Contains(rendered, "one | two") {
		t.Fatalf("code fence table syntax should remain literal: %q", rendered)
	}
}

func TestChatMarkdownRendersMixedBlocks(t *testing.T) {
	markdown := strings.Join([]string{
		"Intro [link](https://prose.example).",
		"",
		"Name | Link",
		"--- | ---",
		"Ollama | [site](https://ollama.com)",
		"",
		"```diff",
		"--- a/file.go",
		"+++ b/file.go",
		"@@ -1 +1 @@",
		"-old",
		"+new",
		"```",
		"",
		"```text",
		"[literal](https://literal.example)",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 90))
	for _, want := range []string{
		"link (https://prose.example)",
		"Ollama",
		"https://ollama.com",
		"-old",
		"+new",
		"[literal](https://literal.example)",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered mixed markdown missing %q: %q", want, rendered)
		}
	}
	for _, notWant := range []string{"--- | ---", "```diff", "```text", "literal (https://literal.example)"} {
		if strings.Contains(rendered, notWant) {
			t.Fatalf("rendered mixed markdown should not contain %q: %q", notWant, rendered)
		}
	}
}

func TestChatMarkdownRendersFencedDiff(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{{
		role: "assistant",
		content: strings.Join([]string{
			"Patch:",
			"",
			"```diff",
			"--- a/file.go",
			"+++ b/file.go",
			"@@ -1 +1 @@",
			"-old",
			"+new",
			"```",
		}, "\n"),
	}}

	rendered := m.renderTranscript(100)
	transcript := stripANSI(rendered)
	if strings.Contains(transcript, "```diff") || strings.Contains(transcript, "```") {
		t.Fatalf("diff fence markers should not render: %q", transcript)
	}
	if !strings.Contains(transcript, "Patch:") ||
		!strings.Contains(transcript, "-old") ||
		!strings.Contains(transcript, "+new") {
		t.Fatalf("rendered fenced diff missing content: %q", transcript)
	}
	if _, ok := renderMarkdownDiffFences(m.entries[0].content, 100); !ok {
		t.Fatal("markdown diff fence should use diff-aware rendering")
	}
}

func TestChatReadToolOutputRendersMarkdown(t *testing.T) {
	output := strings.Join([]string{
		"Name | Value",
		"--- | ---",
		"**Status** | [ok](https://example.com)",
	}, "\n")

	rendered := stripANSI(strings.Join(renderToolOutputLines(chatEntry{detail: "read"}, output, 80), "\n"))
	if strings.Contains(rendered, "--- | ---") || strings.Contains(rendered, "Name | Value") {
		t.Fatalf("read tool markdown table should render: %q", rendered)
	}
	if !strings.Contains(rendered, "Status") || !strings.Contains(rendered, "https://example.com") {
		t.Fatalf("rendered read markdown missing content: %q", rendered)
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

func stripANSI(s string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;:]*[A-Za-z]`)
	return re.ReplaceAllString(s, "")
}
