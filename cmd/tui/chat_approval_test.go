package tui

import (
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

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
		!strings.Contains(view, "$ git status") ||
		!strings.Contains(view, "1. Approve once") ||
		!strings.Contains(view, "2. Approve session") ||
		!strings.Contains(view, "3. Deny") ||
		!strings.Contains(view, "1/2/3 choose • enter select • esc deny") ||
		!strings.Contains(view, "> █") ||
		!strings.Contains(view, "review") {
		t.Fatalf("approval view missing content: %q", view)
	}
	if strings.Contains(view, "Bash wants to run a command") {
		t.Fatalf("approval view should not repeat tool summary when command detail is shown: %q", view)
	}
	if strings.Contains(view, "waiting for approval") {
		t.Fatalf("approval view should not render waiting spinner: %q", view)
	}
	if strings.Contains(view, "Approve once (o)") || strings.Contains(view, "same command in this chat") {
		t.Fatalf("approval choices should be minimal: %q", view)
	}
	if strings.Contains(view, "risk:") {
		t.Fatalf("approval view should not render risk level: %q", view)
	}
	approvalIdx := strings.LastIndex(view, "1. Approve once")
	inputIdx := strings.LastIndex(view, "> █")
	if approvalIdx < 0 || inputIdx < 0 || approvalIdx > inputIdx {
		t.Fatalf("approval picker should render above the input box:\n%s", view)
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

func TestChatApprovalPromptCtrlOExpandsToolCallDetails(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		width:  100,
		height: 24,
		events: make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{
		request: coreagent.ApprovalRequest{
			ToolCallID: "call-1",
			ToolName:   "bash",
			Args:       map[string]any{"command": "git status --short --branch --untracked-files=all"},
			Summary:    "Bash wants to run a command",
		},
		reply: reply,
	})

	transcript := stripANSI(m.renderTranscript(100))
	if strings.Contains(transcript, "$ git status") {
		t.Fatalf("tool details should start collapsed: %q", transcript)
	}

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	if cmd != nil {
		t.Fatal("ctrl+o should only toggle tool details")
	}
	m = updated.(chatModel)
	transcript = stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "▾ Bash") || !strings.Contains(transcript, "$ git status --short --branch --untracked-files=all") {
		t.Fatalf("expanded approval tool should show full command: %q", transcript)
	}
}

func TestChatApprovalPromptClosesHistoryPopup(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		width:        100,
		height:       24,
		historyPopup: &chatHistoryPopup{messages: []api.Message{{Role: "user", Content: "old"}}},
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
	for _, want := range []string{"$ git status", "1. Approve once", "2. Approve session", "3. Deny", "1/2/3 choose • enter select • esc deny"} {
		if !strings.Contains(view, want) {
			t.Fatalf("approval view missing %q:\n%s", want, view)
		}
	}
	if strings.Contains(view, "Bash wants to run a command") {
		t.Fatalf("approval view should not repeat tool summary when command detail is shown:\n%s", view)
	}
	if strings.Contains(view, "Approve once (o)") || strings.Contains(view, "same command in this chat") {
		t.Fatalf("approval choices should be minimal:\n%s", view)
	}
}

func TestChatApprovalPromptDenyNumberShortcut(t *testing.T) {
	reply := make(chan coreagent.ApprovalResult, 1)
	m := chatModel{
		events: make(chan tea.Msg),
	}
	m.openApprovalPrompt(chatApprovalPromptMsg{
		request: coreagent.ApprovalRequest{ToolCallID: "call-1", ToolName: "edit", Args: map[string]any{"path": "note.txt"}},
		reply:   reply,
	})

	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
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
	if m.status != "full access enabled" || m.notificationLine() != "full access enabled" {
		t.Fatalf("status = %q notification = %q, want full access enabled", m.status, m.notificationLine())
	}
	if footer := m.footerLine(); !strings.Contains(footer, "full access") || !strings.Contains(footer, "shift+tab") || strings.Contains(footer, "perm") {
		t.Fatalf("footer missing auto-approve permission mode: %q", footer)
	}

	updated, _ = m.Update(tea.KeyMsg{Type: tea.KeyShiftTab})
	m = updated.(chatModel)
	if m.autoApproveTools() {
		t.Fatal("second shift+tab should return to review mode")
	}
	if m.status != "review mode enabled" || m.notificationLine() != "review mode enabled" {
		t.Fatalf("status = %q notification = %q, want review mode enabled", m.status, m.notificationLine())
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
